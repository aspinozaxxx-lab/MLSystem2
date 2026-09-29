import type { Style as WebGLTileStyle } from "ol/layer/WebGLTile";
import { getRotatedViewport, intersects, type Extent } from "ol/extent";
import Polygon from "ol/geom/Polygon";
import { BAND_CHANNELS, RASTER_CONTRAST, type BandMode } from "./datasetEditor";

export type PseudoProperties = Record<string, unknown>;

export function pseudoRasterCacheSizes(sceneCounts: number[]) {
  const total = sceneCounts.reduce((sum, count) => sum + count, 0);
  const base = sceneCounts.map((count) => count ? 32 : 2);
  const remaining = 256 - base.reduce((sum, size) => sum + size, 0);
  // OpenLayers использует половину cacheSize как длину массива старых ключей.
  // Поэтому каждый бюджет должен быть чётным, включая пока пустые слои.
  return sceneCounts.map((count, index) => base[index] + (total ? 2 * Math.floor(remaining * count / total / 2) : 0));
}

export function pseudoRasterScenes<T extends { bounds: Extent }>(scenes: T[], extent: Extent, resolution: number): T[] {
  return resolution === Number.MAX_SAFE_INTEGER ? scenes.slice(0, 1) : scenes.filter((scene) => intersects(extent, scene.bounds));
}

export function pseudoViewportScenes<T extends { bounds: Extent }>(
  scenes: T[],
  view: { center: number[]; resolution: number; rotation: number },
  size: number[],
): T[] {
  const corners = getRotatedViewport(view.center, view.resolution, view.rotation, size);
  const viewport = new Polygon([Array.from({ length: corners.length / 2 }, (_, index) => corners.slice(index * 2, index * 2 + 2))]);
  return scenes.filter((scene) => viewport.intersectsExtent(scene.bounds));
}

export function pseudoRasterStyle(mode: BandMode, hasAlpha: boolean, hasNir: boolean): WebGLTileStyle {
  const [red, green, blue] = BAND_CHANNELS[hasNir && !hasAlpha ? mode : "RGB"];
  return {
    color: ["color", ["*", ["band", red], 255], ["*", ["band", green], 255], ["*", ["band", blue], 255],
      hasAlpha ? ["*", ["band", 4], ["band", 5]] : ["band", hasNir ? 5 : 4]],
    contrast: RASTER_CONTRAST,
  };
}

export function pseudoClass(value?: PseudoProperties | null) {
  const properties = value ?? {};
  const key = String(properties.object_type_slug || properties.object_type_id || "objects");
  const color = typeof properties.object_type_color === "string" && /^#[0-9a-f]{6}$/i.test(properties.object_type_color)
    ? properties.object_type_color.toUpperCase() : "#22D3EE";
  return { key, color, name: String(properties.object_type_name || properties.object_type_slug || "Объекты") };
}

export function pseudoClasses(features: { properties?: PseudoProperties | null }[]) {
  const groups = new Map<string, ReturnType<typeof pseudoClass> & { count: number }>();
  for (const feature of features) {
    const item = pseudoClass(feature.properties);
    const group = groups.get(item.key);
    if (group) group.count += 1; else groups.set(item.key, { ...item, count: 1 });
  }
  return [...groups.values()].sort((left, right) => left.name.localeCompare(right.name, "ru"));
}
