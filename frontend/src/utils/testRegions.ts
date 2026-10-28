import Polygon from "ol/geom/Polygon";
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
  const x = Math.floor(px - size / 2), y = Math.floor(py - size / 2);
  const coordinates = [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
    .map(([column, row]) => [a * column + b * row + c, d * column + e * row + f]);
  return new Polygon([coordinates]);
}
