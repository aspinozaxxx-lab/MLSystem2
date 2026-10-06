import { afterEach, beforeEach, expect, it, vi } from "vitest";
import Observable from "ol/Observable";
import type OLMap from "ol/Map";
import type { FrameState } from "ol/Map";
import { rasterBackdrop } from "./rasterBackdrop";

type TestCanvas = { width: number; height: number; style: object; getContext: () => { drawImage: ReturnType<typeof vi.fn>; setTransform: ReturnType<typeof vi.fn> } };
let canvases: TestCanvas[];
beforeEach(() => {
  canvases = [];
  vi.stubGlobal("document", { createElement: () => {
    const context = { drawImage: vi.fn(), setTransform: vi.fn() };
    const canvas = { width: 100, height: 100, style: {}, getContext: () => context };
    canvases.push(canvas);
    return canvas;
  } });
});
afterEach(() => vi.unstubAllGlobals());

it("увеличивает готовый кадр и перемещает его вместе с координатами объектов, пока новые тайлы не готовы", () => {
  const map = Object.assign(new Observable(), {
    getView: () => ({ getAnimating: () => false, getInteracting: () => false }),
    getViewport: () => ({ querySelectorAll: () => [{ width: 800, height: 600 }] }),
  });
  const fallback = rasterBackdrop();
  const detach = fallback.attach(map as unknown as OLMap);
  const frame = { size: [800, 600], pixelToCoordinateTransform: [2, 0, 0, -2, 100, 300] };
  map.dispatchEvent({ type: "postrender", frameState: frame } as never);
  map.dispatchEvent("rendercomplete");
  const zoomed = { size: [1000, 700], coordinateToPixelTransform: [1, 0, 0, -1, -120, 310] } as FrameState;
  fallback.layer.render(zoomed, null as never);
  const display = canvases[2].getContext();
  expect(display.setTransform).toHaveBeenLastCalledWith(2, 0, 0, 2, -20, 10);
  expect(display.drawImage).toHaveBeenLastCalledWith(canvases[1], 0, 0);
  expect(canvases[2].width).toBe(1000);

  // Частичная загрузка следующего кадра не заменяет готовое изображение.
  map.dispatchEvent({ type: "postrender", frameState: frame } as never);
  fallback.layer.render(zoomed, null as never);
  expect(display.drawImage).toHaveBeenLastCalledWith(canvases[1], 0, 0);
  map.dispatchEvent("rendercomplete");
  fallback.layer.render(zoomed, null as never);
  expect(display.drawImage).toHaveBeenLastCalledWith(canvases[0], 0, 0);

  // Смена каналов удаляет прежний кадр, чтобы под NRG не оставалось RGB.
  fallback.reset();
  display.drawImage.mockClear();
  fallback.layer.render(zoomed, null as never);
  expect(display.drawImage).not.toHaveBeenCalled();
  detach();
  expect(canvases.every((canvas) => canvas.width === 0)).toBe(true);
});

it("сохраняет последнюю подложку при кадре нулевого размера во время полного экрана", () => {
  const map = Object.assign(new Observable(), {
    getView: () => ({ getAnimating: () => false, getInteracting: () => false }),
    getViewport: () => ({ querySelectorAll: () => [{ width: 800, height: 600 }] }),
  });
  const fallback = rasterBackdrop();
  const detach = fallback.attach(map as unknown as OLMap);
  const transform = [1, 0, 0, 1, 0, 0];
  map.dispatchEvent({ type: "postrender", frameState: { size: [800, 600], pixelToCoordinateTransform: transform } } as never);
  map.dispatchEvent("rendercomplete");
  map.dispatchEvent({ type: "postrender", frameState: { size: [0, 600], pixelToCoordinateTransform: transform } } as never);
  map.dispatchEvent("rendercomplete");
  const frame = { size: [800, 600], coordinateToPixelTransform: transform } as FrameState;
  fallback.layer.render(frame, null as never);
  expect(canvases[2].getContext().drawImage).toHaveBeenLastCalledWith(canvases[1], 0, 0);
  expect(canvases[1].width).toBe(800);
  detach();
});
