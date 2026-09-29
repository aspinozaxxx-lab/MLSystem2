// Кэш сжатых фрагментов TIFF: общий предел для всех снимков и вкладок этого сайта.
export const RASTER_CACHE_BYTES = 512 * 1024 * 1024;
export const RASTER_CACHE_TTL = 24 * 60 * 60 * 1000;
const MAX_ENTRY_BYTES = 8 * 1024 * 1024;
const TRANSIENT_STATUSES = new Set([408, 429, 500, 502, 503, 504]);
type CachedRange = { key: string; data: ArrayBuffer; headers: [string, string][]; size: number; expires: number; used: number };

function request<T>(value: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    value.onsuccess = () => resolve(value.result);
    value.onerror = () => reject(value.error);
  });
}

function completed(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = tx.onerror = () => reject(tx.error);
  });
}

export class RasterRangeCache {
  private database: Promise<IDBDatabase | null> | undefined;
  private generation = 0;
  private limit: Promise<number>;

  constructor(private name = "grovika-raster-ranges-v1", limit?: number, private now = Date.now) {
    this.limit = limit === undefined ? this.storageLimit() : Promise.resolve(limit);
  }

  private async storageLimit(): Promise<number> {
    try {
      const { quota, usage } = await navigator.storage.estimate();
      return Math.max(0, Math.floor(Math.min(RASTER_CACHE_BYTES, (quota ?? Infinity) / 10,
        ((quota ?? Infinity) - (usage ?? 0)) / 2)));
    } catch { return RASTER_CACHE_BYTES; }
  }

  private open(): Promise<IDBDatabase | null> {
    this.database ??= new Promise((resolve) => {
      try {
        const opening = indexedDB.open(this.name, 1);
        opening.onupgradeneeded = () => {
          const ranges = opening.result.createObjectStore("ranges", { keyPath: "key" });
          ranges.createIndex("used", "used");
          ranges.createIndex("expires", "expires");
          opening.result.createObjectStore("meta");
        };
        opening.onsuccess = () => {
          opening.result.onversionchange = () => opening.result.close();
          resolve(opening.result);
        };
        opening.onerror = opening.onblocked = () => resolve(null);
      } catch { resolve(null); }
    });
    return this.database;
  }

  private async get(key: string): Promise<CachedRange | undefined> {
    try {
      const db = await this.open();
      if (!db) return;
      const tx = db.transaction("ranges", "readwrite");
      const done = completed(tx);
      // Обработчик уже установлен, даже если отдельный запрос завершится ошибкой.
      void done.catch(() => {});
      const store = tx.objectStore("ranges");
      const row: CachedRange | undefined = await request(store.get(key));
      if (row && row.expires > this.now()) {
        row.used = this.now();
        store.put(row);
        await done;
        return row;
      }
      await done;
    } catch { /* Недоступное хранилище не мешает обычной загрузке карты. */ }
  }

  private async put(row: CachedRange, generation: number): Promise<void> {
    try {
      const [db, limit] = await Promise.all([this.open(), this.limit]);
      if (!db || row.size > limit || generation !== this.generation) return;
      const tx = db.transaction(["ranges", "meta"], "readwrite");
      const done = completed(tx);
      void done.catch(() => {});
      const ranges = tx.objectStore("ranges");
      const meta = tx.objectStore("meta");
      let total: number = (await request(meta.get("bytes"))) ?? 0;
      const old: CachedRange | undefined = await request(ranges.get(row.key));
      if (old) { total -= old.size; ranges.delete(row.key); }
      const evict = (index: string, shouldRemove: (item: CachedRange) => boolean) => new Promise<void>((resolve, reject) => {
        const cursor = ranges.index(index).openCursor();
        cursor.onerror = () => reject(cursor.error);
        cursor.onsuccess = () => {
          const item = cursor.result;
          if (!item || !shouldRemove(item.value)) { resolve(); return; }
          total -= (item.value as CachedRange).size;
          item.delete();
          item.continue();
        };
      });
      await evict("expires", (item) => item.expires <= this.now());
      await evict("used", () => total + row.size > limit);
      ranges.put(row);
      meta.put(total + row.size, "bytes");
      await done;
    } catch {
      // При переполнении квоты освобождаем только собственный кэш, запрос уже выполнен.
      await this.clear();
    }
  }

  async clear(): Promise<void> {
    this.generation++;
    try {
      const db = await this.open();
      if (!db) return;
      const tx = db.transaction(["ranges", "meta"], "readwrite");
      tx.objectStore("ranges").clear();
      tx.objectStore("meta").clear();
      await completed(tx);
    } catch { /* В приватном режиме постоянный кэш может быть недоступен. */ }
  }

  async load(owner: string, url: string, headers: HeadersInit, signal?: AbortSignal): Promise<Response> {
    signal?.throwIfAborted();
    const parsed = new URL(url, location.href);
    const range = new Headers(headers).get("range");
    const version = parsed.searchParams.get("v");
    const cacheable = owner && parsed.origin === location.origin && version && /^bytes=\d+-\d+$/.test(range ?? "");
    const key = JSON.stringify([owner, parsed.href, range]);
    const generation = this.generation;
    const cached = cacheable ? await this.get(key) : undefined;
    signal?.throwIfAborted();
    if (cached && generation === this.generation) return new Response(cached.data, { status: 206, headers: cached.headers });
    // Повторяем обрыв сети и временный ответ сервера, но не отмену, отсутствие файла или смену версии.
    for (let attempt = 0; ; attempt++) {
      try {
        const response = await fetch(url, { headers, signal, credentials: "same-origin", cache: "no-store" });
        if (TRANSIENT_STATUSES.has(response.status) && attempt < 2) {
          await response.body?.cancel();
        } else {
          await this.cacheResponse(response, cacheable, version, range, key, generation, signal);
          return response;
        }
      } catch (error) {
        if (signal?.aborted || attempt >= 2) throw error;
      }
      await new Promise<void>((resolve, reject) => {
        const abort = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); reject(signal!.reason); };
        const timer = setTimeout(() => { signal?.removeEventListener("abort", abort); resolve(); }, attempt ? 600 : 200);
        signal?.addEventListener("abort", abort, { once: true });
        if (signal?.aborted) abort();
      });
      signal?.throwIfAborted();
    }
  }

  private async cacheResponse(response: Response, cacheable: string | boolean | null, version: string | null,
    range: string | null, key: string, generation: number, signal?: AbortSignal): Promise<void> {
    const size = Number(response.headers.get("content-length"));
    const actual = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get("content-range") ?? "");
    const requested = /^bytes=(\d+)-(\d+)$/.exec(range ?? "");
    if (cacheable && response.status === 206 && response.headers.get("etag") === `"${version}"`
      && size > 0 && size <= MAX_ENTRY_BYTES && actual && requested
      && actual[1] === requested[1] && Number(actual[2]) === Math.min(Number(requested[2]), Number(actual[3]) - 1)
      && size === Number(actual[2]) - Number(actual[1]) + 1) {
      const data = await response.clone().arrayBuffer();
      if (data.byteLength === size && !signal?.aborted) {
        await this.put({ key, data, size, headers: Array.from(response.headers.entries()),
          used: this.now(), expires: this.now() + RASTER_CACHE_TTL }, generation);
      }
    }
  }
}

export const rasterCache = new RasterRangeCache();
