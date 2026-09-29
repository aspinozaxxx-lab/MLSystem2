import DataTile, { disposedError } from "ol/DataTile";
import type DataTileSource from "ol/source/DataTile";
import TileState from "ol/TileState";
import { unByKey } from "ol/Observable";
import { listen, type EventsKey } from "ol/events";

export function rasterRequestError(status: number): string {
  if (status === 401 || status === 403) return "Сеанс завершён или нет доступа. Войдите в Гровику заново.";
  if (status === 404 || status === 410) return "Файл снимка не найден на сервере.";
  if (status === 412) return "Снимок изменился на сервере. Откройте просмотр заново.";
  return `Сервер не отдал фрагмент снимка (код ${status}).`;
}

export function rasterLoadCancelled(error: unknown): boolean {
  return error === disposedError || (error instanceof Error && error.name === "AbortError");
}

export async function rasterResponseError(response: Response): Promise<string> {
  if (response.status === 400) {
    const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
    if (typeof payload?.detail === "string") return payload.detail;
  }
  return rasterRequestError(response.status);
}

// OpenLayers оставляет неудавшийся тайл в ERROR и сам его повторно не запрашивает.
// Повторяем тот же тайл, а сообщение держим только для ещё не восстановленных тайлов.
export function watchRasterLoading(source: DataTileSource, options: {
  active: () => boolean;
  describe: () => string;
  report: (message: string | null) => void;
}) {
  const failed = new Map<DataTile, EventsKey>();
  const attempts = new WeakMap<DataTile, number>();
  const timers = new Map<DataTile, ReturnType<typeof setTimeout>>();
  let disposed = false;
  let sourceFailed = false;
  const active = () => !disposed && options.active();
  const publish = () => { if (active()) options.report(sourceFailed || failed.size ? options.describe() : null); };
  const recovered = (tile: DataTile) => {
    const key = failed.get(tile);
    if (!key) return;
    unByKey(key);
    failed.delete(tile);
    publish();
  };
  const keys = [source.on("change", () => {
    // Ошибка начального HTTP Range меняет state, но GeoTIFF.getView() может остаться pending.
    const next = source.getState() === "error";
    if (next === sourceFailed) return;
    sourceFailed = next;
    publish();
  }), source.on("tileloaderror", ({ tile: rawTile }) => {
    const tile = rawTile as DataTile;
    if (!active() || rasterLoadCancelled(tile.getError())) return;
    const count = attempts.get(tile) ?? 0;
    if (count < 2) {
      if (timers.has(tile)) return;
      attempts.set(tile, count + 1);
      timers.set(tile, setTimeout(() => {
        timers.delete(tile);
        if (active() && tile.getState() === TileState.ERROR && !rasterLoadCancelled(tile.getError())) tile.load();
      }, count ? 1000 : 250));
    } else {
      if (!failed.has(tile)) failed.set(tile, listen(tile, "change", () => {
        // Вытесненный из кэша тайл больше не относится к текущему изображению.
        if (tile.getState() === TileState.EMPTY || tile.getState() === TileState.LOADED) recovered(tile);
      }));
      publish();
    }
  }), source.on("tileloadend", ({ tile }) => {
    recovered(tile as DataTile);
  })];
  return () => {
    disposed = true;
    unByKey(keys);
    timers.forEach(clearTimeout);
    timers.clear();
    failed.forEach(unByKey);
    failed.clear();
  };
}
