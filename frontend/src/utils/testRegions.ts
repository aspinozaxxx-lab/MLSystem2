import Polygon from "ol/geom/Polygon";
import MultiPolygon from "ol/geom/MultiPolygon";
import type Geometry from "ol/geom/Geometry";
import Point from "ol/geom/Point";
import type Feature from "ol/Feature";
import type VectorSource from "ol/source/Vector";
import type { VectorSourceEvent } from "ol/source/Vector";
import { unByKey } from "ol/Observable";
import { containsExtent, getCenter, type Extent } from "ol/extent";
import { Fill, Stroke, Style, Text } from "ol/style";
import RBush from "ol/structs/RBush";
import polygonClipping, { type MultiPolygon as ClipMultiPolygon } from "polygon-clipping";
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

function polygonCoordinates(geometry: Geometry | undefined): ClipMultiPolygon {
  return geometry instanceof Polygon ? [geometry.getCoordinates()] as ClipMultiPolygon
    : geometry instanceof MultiPolygon ? geometry.getCoordinates() as ClipMultiPolygon : [];
}

function rectangleExtent(polygons: ClipMultiPolygon): Extent | null {
  if (polygons.length !== 1 || polygons[0].length !== 1 || polygons[0][0].length !== 5) return null;
  const ring = polygons[0][0];
  const left = Math.min(...ring.map(p => p[0])), right = Math.max(...ring.map(p => p[0]));
  const bottom = Math.min(...ring.map(p => p[1])), top = Math.max(...ring.map(p => p[1]));
  return right > left && top > bottom
    && new Set(ring.slice(0, 4).map(([x, y]) => `${x},${y}`)).size === 4
    && ring.every(([x, y]) => (x === left || x === right) && (y === bottom || y === top))
    && ring[0].every((value, i) => value === ring[4][i])
    && ring.slice(0, 4).every((p, i) => p[0] === ring[i + 1][0] || p[1] === ring[i + 1][1])
    ? [left, bottom, right, top] : null;
}

/** Считает отдельные площадные части после обрезки, не объединяя соседние объекты. */
export function createTestRegionObjectCounter(source: VectorSource<Feature<Geometry>>, footprint: Geometry | null) {
  const boundary = polygonCoordinates(footprint || undefined);
  const rectangularFootprint = rectangleExtent(boundary);
  const edges = footprint && !rectangularFootprint ? new RBush<Extent>() : null;
  if (edges) {
    const segments = boundary.flatMap(polygon => polygon.flatMap(ring => ring.slice(1).map((end, i) => [
      Math.min(ring[i][0], end[0]), Math.min(ring[i][1], end[1]),
      Math.max(ring[i][0], end[0]), Math.max(ring[i][1], end[1]),
    ])));
    edges.load(segments, segments);
  }
  const prepared = new WeakMap<Geometry, { revision: number; polygons: ClipMultiPolygon; rectangle: Extent | null }>();
  const counts = new WeakMap<Geometry, { revision: number; objectsRevision: number; count: number }>();
  const positives = new WeakSet(source.getFeatures().filter(feature =>
    (feature.get("_mlsystem2_role") || "positive") === "positive"));
  let objectsRevision = 0;
  const objectsChanged = (event: VectorSourceEvent<Feature<Geometry>>) => {
    const feature = event.feature;
    const positive = feature && (feature.get("_mlsystem2_role") || "positive") === "positive";
    if (!feature || positive || positives.has(feature)) objectsRevision += 1;
    if (feature) {
      if (positive) positives.add(feature);
      else positives.delete(feature);
    }
  };
  const listeners = (["addfeature", "removefeature", "changefeature", "clear"] as const)
    .map(type => source.on(type, objectsChanged));
  return {
    count(region: Geometry | undefined): number {
      if (!(region instanceof Polygon || region instanceof MultiPolygon)) return 0;
      const revision = region.getRevision(), cached = counts.get(region);
      if (cached?.revision === revision && cached.objectsRevision === objectsRevision) return cached.count;
      const coordinates = polygonCoordinates(region), extent = region.getExtent(), rectangular = rectangleExtent(coordinates);
      let count = 0;
      try {
        // Используем уже существующий пространственный индекс слоя, а не обходим весь снимок.
        for (const feature of source.getFeaturesInExtent(extent)) {
          if ((feature.get("_mlsystem2_role") || "positive") !== "positive") continue;
          const geometry = feature.getGeometry();
          if (!(geometry instanceof Polygon || geometry instanceof MultiPolygon)) continue;
          let object = prepared.get(geometry);
          if (object?.revision !== geometry.getRevision()) {
            let polygons = geometry.getArea() > 0 ? polygonCoordinates(geometry) : [];
            if (footprint && !(rectangularFootprint && containsExtent(footprint.getExtent(), geometry.getExtent()))) {
              // Сложную маску тоже не пересекаем заново с каждым внутренним объектом.
              if (edges && !edges.forEachInExtent(geometry.getExtent(), () => true)) {
                if (!footprint.intersectsCoordinate(getCenter(geometry.getExtent()))) polygons = [];
              } else polygons = boundary.length ? polygonClipping.intersection(polygons, boundary) : [];
            }
            object = { revision: geometry.getRevision(), polygons, rectangle: rectangleExtent(polygons) };
            prepared.set(geometry, object);
          }
          // Внутри прямоугольника топология объекта не меняется; обрезаем только пограничные.
          if (rectangular && containsExtent(extent, geometry.getExtent())) count += object.polygons.length;
          else if (rectangular && object.rectangle) {
            const box = object.rectangle;
            if (Math.max(extent[0], box[0]) < Math.min(extent[2], box[2])
              && Math.max(extent[1], box[1]) < Math.min(extent[3], box[3])) count += 1;
          } else if (object.polygons.length) count += polygonClipping.intersection(object.polygons, coordinates).length;
        }
      } catch {
        // Во время правки свободного контура возможна временно некорректная геометрия.
        return cached?.count ?? 0;
      }
      counts.set(region, { revision, objectsRevision, count });
      return count;
    },
    dispose(): void { unByKey(listeners); },
  };
}

/** Только число, внутри правого верхнего угла, цветом текущего контура. */
export function testRegionObjectCountStyle(geometry: Geometry, count: number, color: string): Style {
  const extent = geometry.getExtent();
  return new Style({
    geometry: new Point([extent[2], extent[3]]),
    zIndex: 30,
    text: new Text({
      text: String(count), font: "600 16px sans-serif", textAlign: "right", textBaseline: "top",
      offsetX: -7, offsetY: 7,
      fill: new Fill({ color }), stroke: new Stroke({ color: "rgba(0, 0, 0, 0.8)", width: 3 }),
    }),
  });
}
