import { afterEach, describe, expect, it, vi } from "vitest";
import DataTile from "ol/DataTile";
import DataTileSource from "ol/source/DataTile";
import GeoTIFF from "ol/source/GeoTIFF";
import { get as getProjection } from "ol/proj";
import { rasterRequestError, watchRasterLoading } from "./rasterLoading";

afterEach(() => vi.useRealTimers());

function fixture(loader: ConstructorParameters<typeof DataTileSource>[0]["loader"]) {
  vi.useFakeTimers();
  const source = new DataTileSource({ loader });
  const report = vi.fn();
  const stop = watchRasterLoading(source, { active: () => true, describe: () => "Ошибка чтения", report });
  const tile = (x = 0) => source.getTile(2, x, 0, 1, getProjection("EPSG:3857")!) as DataTile;
  return { source, report, stop, tile };
}

describe("Восстановление загрузки тайлов", () => {
  it("отмена чтения тайла не означает отсутствующий файл", async () => {
    const f = fixture(async () => { throw new DOMException("Запрос отменён", "AbortError"); });
    const rawErrors = vi.fn();
    f.source.on("tileloaderror", rawErrors);
    const tile = f.tile();
    tile.load();
    await vi.runAllTimersAsync();
    expect(rawErrors).toHaveBeenCalledOnce();
    expect(f.report).not.toHaveBeenCalled();
    f.stop(); f.source.dispose();
  });

  it("повторно загружает сбойный тайл, который OpenLayers оставил бы в ERROR", async () => {
    const loader = vi.fn().mockRejectedValueOnce(new Error("Обрыв сети")).mockResolvedValue(new Uint8Array(4));
    const f = fixture(loader);
    f.tile().load();
    await vi.runAllTimersAsync();
    expect(loader).toHaveBeenCalledTimes(2);
    expect(f.report).not.toHaveBeenCalled();
    f.stop(); f.source.dispose();
  });

  it("ограничивает повторы и не прячет ошибку одного тайла при успехе соседнего", async () => {
    let recovered = false;
    const loader = vi.fn(async (_z, x) => {
      if (x === 0 && !recovered) throw new Error("Ошибка чтения");
      return new Uint8Array(4);
    });
    const f = fixture(loader);
    const bad = f.tile();
    bad.load();
    await vi.runAllTimersAsync();
    expect(loader).toHaveBeenCalledTimes(3);
    expect(f.report).toHaveBeenLastCalledWith("Ошибка чтения");
    f.tile(1).load();
    await vi.runAllTimersAsync();
    expect(f.report).toHaveBeenCalledTimes(1);
    recovered = true;
    bad.load();
    await vi.runAllTimersAsync();
    expect(f.report).toHaveBeenLastCalledWith(null);
    f.stop(); f.source.dispose();
  });

  it("не возобновляет запросы уже удалённого источника", async () => {
    const loader = vi.fn().mockRejectedValue(new Error("Обрыв сети"));
    const f = fixture(loader);
    f.tile().load();
    await vi.advanceTimersByTimeAsync(0);
    f.stop();
    await vi.runAllTimersAsync();
    expect(loader).toHaveBeenCalledOnce();
    expect(f.report).not.toHaveBeenCalled();
    f.source.dispose();
  });

  it("убирает предупреждение после вытеснения сбойного тайла", async () => {
    const f = fixture(async () => { throw new Error("Ошибка чтения"); });
    const tile = f.tile();
    tile.load();
    await vi.runAllTimersAsync();
    expect(f.report).toHaveBeenLastCalledWith("Ошибка чтения");
    tile.dispose();
    expect(f.report).toHaveBeenLastCalledWith(null);
    f.stop(); f.source.dispose();
  });

  it("сообщает об отсутствии TIFF при начальном чтении заголовка, не ожидая getView", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const source = new GeoTIFF({ sources: [{ url: "/missing.tif", loader: async () => new Response(null, { status: 404 }) }] });
    const report = vi.fn();
    const stop = watchRasterLoading(source, { active: () => true, describe: () => rasterRequestError(404), report });
    try {
      await vi.runAllTimersAsync();
      expect(source.getState()).toBe("error");
      expect(report).toHaveBeenLastCalledWith("Файл снимка не найден на сервере.");
    } finally { stop(); source.dispose(); consoleError.mockRestore(); }
  });

  it("отличает отсутствие файла, смену ревизии и истёкший сеанс", () => {
    expect(rasterRequestError(404)).toContain("не найден");
    expect(rasterRequestError(412)).toContain("изменился");
    expect(rasterRequestError(401)).toContain("Войдите");
  });
});
