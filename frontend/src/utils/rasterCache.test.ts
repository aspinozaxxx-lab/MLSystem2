import "fake-indexeddb/auto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RASTER_CACHE_TTL, RasterRangeCache } from "./rasterCache";

const origin = "https://grovika.test";
const url = `${origin}/api/v1/results/pseudo-markup/result/raster/scene?v=first`;
const headers = { Range: "bytes=0-3" };
let now: number;
let fetcher: ReturnType<typeof vi.fn>;
function part(version = "first", text = "TIFF") {
  return new Response(text, { status: 206, headers: { "content-length": "4", "content-range": "bytes 0-3/20", etag: `"${version}"` } });
}
function cache(limit = 20) { return new RasterRangeCache(`проверка-${crypto.randomUUID()}`, limit, () => now); }

beforeEach(() => {
  now = 1000;
  vi.stubGlobal("location", { href: origin, origin });
  fetcher = vi.fn(async () => part());
  vi.stubGlobal("fetch", fetcher);
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("Кэш фрагментов TIFF", () => {
  it("повторно использует диапазон после пересоздания загрузчика, но не продлевает сутки", async () => {
    const name = crypto.randomUUID();
    const first = new RasterRangeCache(name, 20, () => now);
    expect(await (await first.load("oleg", url, headers)).text()).toBe("TIFF");
    const reopened = new RasterRangeCache(name, 20, () => now);
    now += RASTER_CACHE_TTL - 1;
    expect(await (await reopened.load("oleg", url, headers)).text()).toBe("TIFF");
    expect(fetcher).toHaveBeenCalledTimes(1);
    now += 1;
    await reopened.load("oleg", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("разделяет версии файлов, пользователей и диапазоны", async () => {
    const store = cache();
    await store.load("oleg", url, headers);
    fetcher.mockImplementation(async () => part("second"));
    await store.load("oleg", url.replace("first", "second"), headers);
    fetcher.mockImplementation(async () => part());
    await store.load("other", url, headers);
    await store.load("oleg", url, { Range: "bytes=4-7" });
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it("вытесняет самый давно просмотренный фрагмент при достижении предела", async () => {
    const store = cache(8);
    await store.load("a", url, headers);
    now++;
    await store.load("b", url, headers);
    now++;
    await store.load("a", url, headers);
    now++;
    await store.load("c", url, headers);
    await store.load("a", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(3);
    await store.load("b", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it("соблюдает общий предел при одновременной записи из разных вкладок", async () => {
    const name = crypto.randomUUID();
    const a = new RasterRangeCache(name, 8, () => now);
    const b = new RasterRangeCache(name, 8, () => now);
    await Promise.all([a.load("a", url, headers), b.load("b", url, headers), a.load("c", url, headers)]);
    const opening = indexedDB.open(name, 1);
    const db = await new Promise<IDBDatabase>((resolve) => { opening.onsuccess = () => resolve(opening.result); });
    const query = db.transaction("ranges").objectStore("ranges").getAll();
    const rows = await new Promise<{ size: number }[]>((resolve) => { query.onsuccess = () => resolve(query.result); });
    expect(rows.reduce((sum, row) => sum + row.size, 0)).toBe(8);
    db.close();
  });

  it.each([200, 401, 403, 404, 412])("не сохраняет ответ со статусом %s", async (status) => {
    fetcher.mockImplementation(async () => new Response("TIFF", { status }));
    const store = cache();
    await store.load("oleg", url, headers);
    await store.load("oleg", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("восстанавливает запрос после обрыва и временного ответа сервера", async () => {
    fetcher.mockRejectedValueOnce(new TypeError("Обрыв сети"))
      .mockResolvedValueOnce(new Response("Временно недоступно", { status: 503 }));
    const store = cache();
    expect(await (await store.load("oleg", url, headers)).text()).toBe("TIFF");
    await store.load("oleg", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it("останавливает повторы при устойчивой ошибке и при отмене запроса", async () => {
    fetcher.mockResolvedValue(new Response("Временно недоступно", { status: 503 }));
    expect((await cache().load("oleg", url, headers)).status).toBe(503);
    expect(fetcher).toHaveBeenCalledTimes(3);
    const abort = new AbortController();
    fetcher.mockImplementationOnce(async () => { abort.abort(); throw abort.signal.reason; });
    await expect(cache().load("oleg", url, headers, abort.signal)).rejects.toThrow();
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it("не сохраняет чужой ETag, неполные байты и неправильный диапазон", async () => {
    const store = cache();
    for (const response of [part("other"), part("first", "xx"), new Response("TIFF", { status: 206,
      headers: { "content-length": "4", "content-range": "bytes 4-7/20", etag: '"first"' } })]) {
      fetcher.mockImplementation(async () => response.clone());
      await store.load("oleg", url, headers);
      await store.load("oleg", url, headers);
    }
    expect(fetcher).toHaveBeenCalledTimes(6);
  });

  it("очищается при выходе и не принимает запоздавший ответ после очистки", async () => {
    const store = cache();
    await store.load("oleg", url, headers);
    await store.clear();
    let release!: (response: Response) => void;
    fetcher.mockImplementationOnce(() => new Promise<Response>((resolve) => { release = resolve; }));
    const pending = store.load("oleg", url, headers);
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    await store.clear();
    release(part());
    await pending;
    await store.load("oleg", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it("не требует IndexedDB и не возвращает данные отменённому запросу", async () => {
    vi.stubGlobal("indexedDB", undefined);
    const store = cache();
    expect(await (await store.load("oleg", url, headers)).text()).toBe("TIFF");
    const abort = new AbortController();
    abort.abort();
    await expect(store.load("oleg", url, headers, abort.signal)).rejects.toThrow();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("снижает предел по квоте браузера и переживает её исчерпание", async () => {
    vi.stubGlobal("navigator", { storage: { estimate: async () => ({ quota: 80, usage: 0 }) } });
    const store = new RasterRangeCache(crypto.randomUUID(), undefined, () => now++);
    await store.load("a", url, headers);
    await store.load("b", url, headers);
    await store.load("c", url, headers);
    await store.load("a", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(4);
    vi.spyOn(IDBObjectStore.prototype, "put").mockImplementationOnce(() => { throw new DOMException("Недостаточно места", "QuotaExceededError"); });
    expect(await (await store.load("d", url, headers)).text()).toBe("TIFF");
    await store.load("d", url, headers);
    expect(fetcher).toHaveBeenCalledTimes(6);
  });
});
