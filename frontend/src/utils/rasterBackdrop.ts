import type OLMap from "ol/Map";
import Layer from "ol/layer/Layer";
import { multiply } from "ol/transform";
import { unByKey } from "ol/Observable";

// Один готовый кадр под тайлами сохраняет изображение даже при скачке через несколько масштабов.
// Снимок берётся синхронно после WebGL-отрисовки: позже браузер вправе очистить его буфер.
export function rasterBackdrop() {
  let saved = document.createElement("canvas");
  let pending = document.createElement("canvas");
  const canvas = document.createElement("canvas");
  canvas.className = "ol-layer pseudo-raster-backdrop";
  canvas.style.position = "absolute";
  let savedTransform: number[] | undefined;
  let pendingTransform: number[] | undefined;
  const layer = new Layer({
    render: (frame) => {
      canvas.width = frame.size[0];
      canvas.height = frame.size[1];
      const context = canvas.getContext("2d")!;
      if (savedTransform) {
        const transform = multiply(frame.coordinateToPixelTransform.slice(), savedTransform);
        context.setTransform(...transform as [number, number, number, number, number, number]);
        context.drawImage(saved, 0, 0);
      }
      return canvas;
    },
  });
  const reset = () => { savedTransform = pendingTransform = undefined; layer.changed(); };
  const attach = (map: OLMap) => {
    const capture = map.on("postrender", (event) => {
      const frame = event.frameState;
      if (!frame || !layer.getVisible() || map.getView().getAnimating() || map.getView().getInteracting()) return;
      pending.width = frame.size[0];
      pending.height = frame.size[1];
      const context = pending.getContext("2d")!;
      // Слои трёх схем каналов используют общий WebGL-холст.
      map.getViewport().querySelectorAll<HTMLCanvasElement>("canvas.pseudo-raster").forEach((source) => {
        if (source.width && source.height) context.drawImage(source, 0, 0, pending.width, pending.height);
      });
      pendingTransform = frame.pixelToCoordinateTransform.slice();
    });
    const commit = map.on("rendercomplete", () => {
      if (!pendingTransform) return;
      [saved, pending] = [pending, saved];
      savedTransform = pendingTransform;
      pendingTransform = undefined;
    });
    return () => { unByKey([capture, commit]); reset(); saved.width = pending.width = canvas.width = 0; layer.dispose(); };
  };
  return { layer, attach, reset };
}
