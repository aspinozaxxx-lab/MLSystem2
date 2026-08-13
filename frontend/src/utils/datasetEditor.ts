import { containsCoordinate, type Extent } from "ol/extent";
import type Geometry from "ol/geom/Geometry";
import type MultiPolygon from "ol/geom/MultiPolygon";
import type Polygon from "ol/geom/Polygon";

export type JsonObject = Record<string, unknown>;
export type SortDirection = "ascending" | "descending";

export const RASTER_MAX_SCALE = 10;
export const RASTER_CONTRAST = 0.15;

export function preventMapMiddleButtonDefault(
  event: Pick<MouseEvent, "button" | "preventDefault">,
): boolean {
  if (event.button !== 1) return false;
  event.preventDefault();
  return true;
}

export type DraftSnapshot = {
  geojson: JsonObject;
  newFeatureIndexes: number[];
};

export type DraftState = {
  baseline: DraftSnapshot;
  current: DraftSnapshot;
  history: DraftSnapshot[];
};

export type EditableVertex = {
  polygonIndex: number;
  ringIndex: number;
  vertexIndex: number;
  coordinate: number[];
};

export type VertexDeletionResult = {
  geometry: Geometry;
  removedCount: number;
  blockedRingCount: number;
};

export type CountedScene = {
  annotation_name: string;
  total_count: number;
  positive_count: number;
  hard_negative_count: number;
  class_counts?: Record<string, number>;
};

export type PublishableDraft = DraftState & {
  scene: CountedScene & { revision: string };
};

export function cloneSnapshot(snapshot: DraftSnapshot): DraftSnapshot {
  return {
    geojson: JSON.parse(JSON.stringify(snapshot.geojson)) as JsonObject,
    newFeatureIndexes: [...snapshot.newFeatureIndexes],
  };
}

export function snapshotsEqual(left: DraftSnapshot, right: DraftSnapshot): boolean {
  return (
    canonicalJson(left.geojson) === canonicalJson(right.geojson) &&
    left.newFeatureIndexes.length === right.newFeatureIndexes.length &&
    left.newFeatureIndexes.every((value, index) => value === right.newFeatureIndexes[index])
  );
}

export function extendRasterResolutions(
  viewResolutions: number[],
  nativeResolution: number,
  maximumScale = RASTER_MAX_SCALE,
): number[] {
  if (
    !viewResolutions.length ||
    !Number.isFinite(nativeResolution) ||
    nativeResolution <= 0 ||
    !Number.isFinite(maximumScale) ||
    maximumScale <= 1
  ) {
    return [...viewResolutions];
  }
  const minimumResolution = nativeResolution / maximumScale;
  const resolutionTolerance = nativeResolution * 1e-12;
  const result = viewResolutions.filter(
    (resolution) => resolution + resolutionTolerance >= minimumResolution,
  );
  const factors = [2, 4, 8, maximumScale]
    .filter((factor) => factor <= maximumScale)
    .sort((left, right) => left - right);
  for (const factor of factors) {
    const resolution = nativeResolution / factor;
    if (
      !result.some(
        (existing) => Math.abs(existing - resolution) <= resolutionTolerance,
      )
    ) {
      result.push(resolution);
    }
  }
  return result.sort((left, right) => right - left);
}

export function geometryInsideFootprint(
  geometry: Geometry | null | undefined,
  footprint: Geometry | null,
): boolean {
  if (!geometry || !footprint) return Boolean(geometry);
  const simpleGeometry = geometry as Geometry & {
    getFlatCoordinates?: () => number[] | null;
    getStride?: () => number;
  };
  const coordinates = simpleGeometry.getFlatCoordinates?.();
  const stride = simpleGeometry.getStride?.() || 2;
  if (!coordinates) return false;
  const extent = footprint.getExtent();
  const footprintScale = Math.max(1, ...extent.map((value) => Math.abs(value)));
  for (let index = 0; index < coordinates.length; index += stride) {
    const coordinate = [coordinates[index], coordinates[index + 1]];
    if (!coordinate.every(Number.isFinite)) return false;
    if (footprint.intersectsCoordinate(coordinate)) continue;
    const closest = footprint.getClosestPoint(coordinate);
    if (!closest.slice(0, 2).every(Number.isFinite)) return false;
    const coordinateScale = Math.max(
      footprintScale,
      Math.abs(coordinate[0]),
      Math.abs(coordinate[1]),
    );
    const tolerance = coordinateScale * Number.EPSILON * 64;
    const deltaX = closest[0] - coordinate[0];
    const deltaY = closest[1] - coordinate[1];
    if (deltaX * deltaX + deltaY * deltaY > tolerance * tolerance) return false;
  }
  return true;
}

export function editableVertexCoordinates(
  geometry: Geometry | null | undefined,
): number[][] {
  return editableVertices(geometry).map((vertex) => vertex.coordinate);
}

export function editableVertices(
  geometry: Geometry | null | undefined,
): EditableVertex[] {
  if (!geometry) return [];
  if (geometry.getType() === "Polygon") {
    return polygonVertices((geometry as Polygon).getCoordinates(), 0);
  }
  if (geometry.getType() === "MultiPolygon") {
    return (geometry as MultiPolygon)
      .getCoordinates()
      .flatMap((polygon, polygonIndex) => polygonVertices(polygon, polygonIndex));
  }
  return [];
}

export function editableVerticesInExtent(
  geometry: Geometry | null | undefined,
  extent: Extent,
): EditableVertex[] {
  return editableVertices(geometry).filter((vertex) =>
    containsCoordinate(extent, vertex.coordinate),
  );
}

export function deleteEditableVertices(
  geometry: Geometry,
  selectedVertices: EditableVertex[],
): VertexDeletionResult {
  const geometryType = geometry.getType();
  if (geometryType !== "Polygon" && geometryType !== "MultiPolygon") {
    return { geometry: geometry.clone(), removedCount: 0, blockedRingCount: 0 };
  }

  const selectedIndexes = new Map<string, Set<number>>();
  for (const vertex of selectedVertices) {
    const ringKey = `${vertex.polygonIndex}:${vertex.ringIndex}`;
    const indexes = selectedIndexes.get(ringKey) || new Set<number>();
    indexes.add(vertex.vertexIndex);
    selectedIndexes.set(ringKey, indexes);
  }

  let removedCount = 0;
  let blockedRingCount = 0;
  const sourcePolygons = geometryType === "Polygon"
    ? [(geometry as Polygon).getCoordinates()]
    : (geometry as MultiPolygon).getCoordinates();
  const coordinates = sourcePolygons.map((polygon, polygonIndex) =>
    polygon.map((ring, ringIndex) => {
      const openRing = ringWithoutClosingVertex(ring);
      const selected = selectedIndexes.get(`${polygonIndex}:${ringIndex}`);
      const removable = selected
        ? new Set([...selected].filter((index) => index >= 0 && index < openRing.length))
        : new Set<number>();
      if (!removable.size) return cloneRing(ring);
      if (openRing.length - removable.size < 3) {
        blockedRingCount += 1;
        return cloneRing(ring);
      }
      const remaining = openRing
        .filter((_, vertexIndex) => !removable.has(vertexIndex))
        .map((coordinate) => [...coordinate]);
      removedCount += removable.size;
      return [...remaining, [...remaining[0]]];
    }),
  );

  const result = geometry.clone();
  if (geometryType === "Polygon") {
    (result as Polygon).setCoordinates(coordinates[0]);
  } else {
    (result as MultiPolygon).setCoordinates(coordinates);
  }
  return { geometry: result, removedCount, blockedRingCount };
}

function polygonVertices(polygon: number[][][], polygonIndex: number): EditableVertex[] {
  return polygon.flatMap((ring, ringIndex) =>
    ringWithoutClosingVertex(ring).map((coordinate, vertexIndex) => ({
      polygonIndex,
      ringIndex,
      vertexIndex,
      coordinate: [...coordinate],
    })),
  );
}

function ringWithoutClosingVertex(ring: number[][]): number[][] {
  if (ring.length < 2) return ring;
  const first = ring[0];
  const last = ring.at(-1);
  return last && coordinatesEqual(first, last) ? ring.slice(0, -1) : ring;
}

function coordinatesEqual(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function cloneRing(ring: number[][]): number[][] {
  return ring.map((coordinate) => [...coordinate]);
}

export function featureCounts(geojson: JsonObject): {
  total: number;
  positive: number;
  hardNegative: number;
} {
  const features = Array.isArray(geojson.features) ? geojson.features : [];
  let positive = 0;
  let hardNegative = 0;
  for (const feature of features) {
    if (!feature || typeof feature !== "object") continue;
    const properties = (feature as JsonObject).properties;
    const role =
      properties && typeof properties === "object"
        ? (properties as JsonObject)._mlsystem2_role
        : undefined;
    if (role === "hard_negative") hardNegative += 1;
    else positive += 1;
  }
  return { total: positive + hardNegative, positive, hardNegative };
}

export function featureClassCounts(geojson: JsonObject): Record<string, number> {
  const result: Record<string, number> = {};
  const features = Array.isArray(geojson.features) ? geojson.features : [];
  for (const feature of features) {
    if (!feature || typeof feature !== "object") continue;
    const properties = (feature as JsonObject).properties;
    if (!properties || typeof properties !== "object") continue;
    const values = properties as JsonObject;
    if (values._mlsystem2_role === "hard_negative") continue;
    const slug = values._mlsystem2_class;
    if (typeof slug === "string" && slug) result[slug] = (result[slug] || 0) + 1;
  }
  return result;
}

export function draftChanged(draft: DraftState): boolean {
  return canonicalJson(draft.baseline.geojson) !== canonicalJson(draft.current.geojson);
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortObjectKeys(value));
}

function sortObjectKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortObjectKeys);
  if (!value || typeof value !== "object") return value;
  const sorted = Object.fromEntries(
    Object.entries(value as JsonObject)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortObjectKeys(item)]),
  ) as JsonObject;
  if (sorted.type === "FeatureCollection" && Array.isArray(sorted.features)) {
    sorted.features = [...sorted.features].sort((left, right) => {
      const leftJson = JSON.stringify(left);
      const rightJson = JSON.stringify(right);
      return leftJson < rightJson ? -1 : leftJson > rightJson ? 1 : 0;
    });
  }
  return sorted;
}

export function appendHistory(
  history: DraftSnapshot[],
  snapshot: DraftSnapshot,
  limit = 100,
): DraftSnapshot[] {
  return [...history, cloneSnapshot(snapshot)].slice(-limit);
}

export function undoDraft(draft: DraftState): DraftState | null {
  const previous = draft.history.at(-1);
  if (!previous) return null;
  return {
    ...draft,
    current: cloneSnapshot(previous),
    history: draft.history.slice(0, -1),
  };
}

export function sceneCounts(
  scene: CountedScene,
  draft: DraftState | undefined,
): { total: number; positive: number; hardNegative: number } {
  return draft
    ? featureCounts(draft.current.geojson)
    : {
        total: scene.total_count,
        positive: scene.positive_count,
        hardNegative: scene.hard_negative_count,
      };
}

export function sceneClassCounts(
  scene: CountedScene,
  draft: DraftState | undefined,
): Record<string, number> {
  return draft
    ? featureClassCounts(draft.current.geojson)
    : { ...(scene.class_counts || {}) };
}

export function sortEditorScenes<T extends CountedScene>(
  scenes: T[],
  drafts: Record<string, DraftState>,
  direction: SortDirection,
): T[] {
  const factor = direction === "ascending" ? 1 : -1;
  return [...scenes].sort((left, right) => {
    const difference =
      sceneCounts(left, drafts[left.annotation_name]).total -
      sceneCounts(right, drafts[right.annotation_name]).total;
    return (
      difference * factor ||
      left.annotation_name.localeCompare(right.annotation_name, "ru", { sensitivity: "base" })
    );
  });
}

export function publishScenes(
  drafts: Record<string, PublishableDraft>,
): Array<{ annotation_name: string; revision: string; geojson: JsonObject }> {
  return Object.values(drafts)
    .filter(draftChanged)
    .sort((left, right) =>
      left.scene.annotation_name.localeCompare(right.scene.annotation_name, "ru", {
        sensitivity: "base",
      }),
    )
    .map((draft) => ({
      annotation_name: draft.scene.annotation_name,
      revision: draft.scene.revision,
      geojson: draft.current.geojson,
    }));
}

export function acceptPublishedDraft<T extends PublishableDraft>(
  draft: T,
  scene: T["scene"],
): T {
  const saved = cloneSnapshot({
    geojson: draft.current.geojson,
    newFeatureIndexes: [],
  });
  return {
    ...draft,
    scene,
    baseline: cloneSnapshot(saved),
    current: cloneSnapshot(saved),
    history: [],
  };
}
