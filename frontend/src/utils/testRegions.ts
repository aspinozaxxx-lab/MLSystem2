import Polygon from "ol/geom/Polygon";
import MultiPolygon from "ol/geom/MultiPolygon";
import type Geometry from "ol/geom/Geometry";
import { getCenter } from "ol/extent";
import type { JsonObject } from "./datasetEditor";

export const TEST_REGION_SIZES = Array.from({ length: 16 }, (_, index) => (index + 1) * 512);

export function testRegions(payload: JsonObject): JsonObject[] {
  return ((payload.features || []) as JsonObject[]).filter((feature) =>
    (feature.properties as JsonObject | undefined)?._mlsystem2_role === "annotation_zone");
}

export function replaceTestRegionLabels(previous: JsonObject, labels: JsonObject): JsonObject {
  const positives = ((labels.features || []) as JsonObject[]).filter((feature) =>
    (feature.properties as JsonObject | undefined)?._mlsystem2_role !== "annotation_zone");
  return { ...labels, features: [...positives, ...testRegions(previous)] };
}

export function squareTestRegion(center: number[], size: number, grid: number[]): Polygon {
  const [a, b, c, d, e, f] = grid;
  const determinant = a * e - b * d;
  if (!determinant || !Number.isFinite(determinant)) throw new Error("Не удалось определить пиксельную сетку снимка.");
  const px = (e * (center[0] - c) - b * (center[1] - f)) / determinant;
  const py = (-d * (center[0] - c) + a * (center[1] - f)) / determinant;
  const x = Math.floor(px - size / 2 + 1e-7), y = Math.floor(py - size / 2 + 1e-7);
  const coordinates = [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
    .map(([column, row]) => [a * column + b * row + c, d * column + e * row + f]);
  return new Polygon([coordinates]);
}

/** Размер уже сохранённой квадратной рамки; отверстия и произвольные контуры дают 0. */
export function testRegionSize(geometry: Geometry | undefined, grid: number[]): number {
  const polygons = geometry instanceof Polygon ? [geometry.getCoordinates()]
    : geometry instanceof MultiPolygon ? geometry.getCoordinates() : [];
  if (polygons.length !== 1 || polygons[0].length !== 1 || polygons[0][0].length !== 5) return 0;
  const [a, b, c, d, e, f] = grid, determinant = a * e - b * d;
  if (!determinant) return 0;
  const points = polygons[0][0].map(([x, y]) => [
    (e * (x - c) - b * (y - f)) / determinant, (-d * (x - c) + a * (y - f)) / determinant,
  ]);
  const close = (left: number, right: number) => Math.abs(left - right) < 1e-5;
  const width = Math.max(...points.map(p => p[0])) - Math.min(...points.map(p => p[0]));
  const height = Math.max(...points.map(p => p[1])) - Math.min(...points.map(p => p[1]));
  const size = TEST_REGION_SIZES.find(value => close(width, value) && close(height, value));
  if (!size || points.some(p => p.some(value => !close(value, Math.round(value))))) return 0;
  if (new Set(points.slice(0, 4).map(p => p.map(Math.round).join(","))).size !== 4) return 0;
  if (!points[0].every((value, i) => close(value, points[4][i]))) return 0;
  return points.slice(0, 4).every((p, i) => {
    const next = points[i + 1];
    return (close(p[0], next[0]) && close(Math.abs(p[1] - next[1]), size))
      || (close(p[1], next[1]) && close(Math.abs(p[0] - next[0]), size));
  }) ? size : 0;
}

export function resizeTestRegion(geometry: Geometry, size: number, grid: number[]): Polygon {
  return squareTestRegion(getCenter(geometry.getExtent()), size, grid);
}

/** Перенос на целое число исходных пикселей: размеры, кольца и отверстия неизменны. */
export function translateTestRegion(geometry: Geometry, delta: number[], grid: number[]): Geometry {
  const [a, b, , d, e] = grid, determinant = a * e - b * d;
  if (!determinant) throw new Error("Не удалось определить пиксельную сетку снимка.");
  const column = Math.round((e * delta[0] - b * delta[1]) / determinant);
  const row = Math.round((-d * delta[0] + a * delta[1]) / determinant);
  const translated = geometry.clone();
  translated.translate(a * column + b * row, d * column + e * row);
  return translated;
}
