import { describe, expect, it } from "vitest";
import MultiPolygon from "ol/geom/MultiPolygon";
import Polygon from "ol/geom/Polygon";

import {
  acceptPublishedDraft,
  appendHistory,
  deleteEditableVertices,
  draftChanged,
  editableVertexCoordinates,
  editableVertices,
  editableVerticesInExtent,
  extendRasterResolutions,
  featureClassCounts,
  featureCounts,
  geometryInsideFootprint,
  preventMapMiddleButtonDefault,
  publishScenes,
  sceneClassCounts,
  snapshotsEqual,
  sortEditorScenes,
  undoDraft,
  type DraftSnapshot,
  type PublishableDraft,
} from "./datasetEditor";

const snapshot = (roles: string[], newFeatureIndexes: number[] = []): DraftSnapshot => ({
  geojson: {
    type: "FeatureCollection",
    features: roles.map((role) => ({
      type: "Feature",
      properties: { _mlsystem2_role: role },
      geometry: { type: "Polygon", coordinates: [] },
    })),
  },
  newFeatureIndexes,
});

const draft = (name: string, baseline: DraftSnapshot, current: DraftSnapshot): PublishableDraft => ({
  scene: {
    annotation_name: name,
    revision: `revision-${name}`,
    total_count: 0,
    positive_count: 0,
    hard_negative_count: 0,
  },
  baseline,
  current,
  history: [],
});

describe("черновики редактора датасетов", () => {
  it("расширяет нативную шкалу GeoTIFF до 1000%", () => {
    expect(extendRasterResolutions([8, 4, 2, 1], 1)).toEqual([
      8,
      4,
      2,
      1,
      0.5,
      0.25,
      0.125,
      0.1,
    ]);
  });

  it("сохраняет существующий overzoom и ограничивает его десятикратным масштабом", () => {
    expect(extendRasterResolutions([2, 1, 0.5, 0.0625], 1)).toEqual([
      2,
      1,
      0.5,
      0.25,
      0.125,
      0.1,
    ]);
  });

  it("отменяет нативное действие только для средней кнопки мыши", () => {
    let prevented = 0;
    const preventDefault = () => {
      prevented += 1;
    };

    expect(preventMapMiddleButtonDefault({ button: 0, preventDefault })).toBe(false);
    expect(prevented).toBe(0);
    expect(preventMapMiddleButtonDefault({ button: 1, preventDefault })).toBe(true);
    expect(prevented).toBe(1);
  });

  it("принимает численную погрешность на границе footprint, но отклоняет реальный выход", () => {
    const left = 11_478_026;
    const bottom = 6_962_204;
    const right = 11_522_108;
    const top = 7_011_923;
    const footprint = new Polygon([[
      [left, bottom],
      [right, bottom],
      [right, top],
      [left, top],
      [left, bottom],
    ]]);
    const boundaryGeometry = new Polygon([[
      [right - 10, bottom + 10],
      [right + 1e-9, bottom + 10],
      [right + 1e-9, bottom + 20],
      [right - 10, bottom + 20],
      [right - 10, bottom + 10],
    ]]);
    const outsideGeometry = new Polygon([[
      [right - 10, bottom + 10],
      [right + 0.001, bottom + 10],
      [right + 0.001, bottom + 20],
      [right - 10, bottom + 20],
      [right - 10, bottom + 10],
    ]]);

    expect(footprint.intersectsCoordinate([right, bottom + 10])).toBe(false);
    expect(geometryInsideFootprint(boundaryGeometry, footprint)).toBe(true);
    expect(geometryInsideFootprint(outsideGeometry, footprint)).toBe(false);
  });

  it("возвращает все редактируемые вершины без замыкающих дублей", () => {
    const polygon = new Polygon([
      [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
      [[1, 1], [2, 1], [1, 2], [1, 1]],
    ]);
    const multipolygon = new MultiPolygon([
      polygon.getCoordinates(),
      [[[10, 10], [12, 10], [11, 12], [10, 10]]],
    ]);

    expect(editableVertexCoordinates(polygon)).toEqual([
      [0, 0], [4, 0], [4, 4], [0, 4],
      [1, 1], [2, 1], [1, 2],
    ]);
    expect(editableVertexCoordinates(multipolygon)).toHaveLength(10);
  });

  it("выбирает рамкой все вершины внутри области", () => {
    const polygon = new Polygon([[
      [0, 0], [4, 0], [4, 4], [0, 4], [0, 0],
    ]]);

    expect(editableVerticesInExtent(polygon, [3.5, -0.5, 4.5, 4.5])).toEqual([
      { polygonIndex: 0, ringIndex: 0, vertexIndex: 1, coordinate: [4, 0] },
      { polygonIndex: 0, ringIndex: 0, vertexIndex: 2, coordinate: [4, 4] },
    ]);
  });

  it("удаляет выбранные вершины и снова замыкает кольцо", () => {
    const polygon = new Polygon([[
      [0, 0], [2, 0], [4, 0], [4, 4], [2, 4], [0, 4], [0, 0],
    ]]);
    const selected = editableVertices(polygon).filter((vertex) =>
      vertex.vertexIndex === 1 || vertex.vertexIndex === 3,
    );

    const result = deleteEditableVertices(polygon, selected);

    expect(result.removedCount).toBe(2);
    expect(result.blockedRingCount).toBe(0);
    expect((result.geometry as Polygon).getCoordinates()).toEqual([[
      [0, 0], [4, 0], [2, 4], [0, 4], [0, 0],
    ]]);
    expect(polygon.getCoordinates()[0]).toHaveLength(7);
  });

  it("не удаляет вершины кольца, если останется меньше трёх", () => {
    const polygon = new Polygon([[
      [0, 0], [4, 0], [0, 4], [0, 0],
    ]]);

    const result = deleteEditableVertices(polygon, [editableVertices(polygon)[0]]);

    expect(result.removedCount).toBe(0);
    expect(result.blockedRingCount).toBe(1);
    expect((result.geometry as Polygon).getCoordinates()).toEqual(polygon.getCoordinates());
  });

  it("удаляет вершины независимо в отверстиях и частях MultiPolygon", () => {
    const multipolygon = new MultiPolygon([
      [
        [[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]],
        [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]],
      ],
      [[[10, 10], [14, 10], [14, 14], [10, 14], [10, 10]]],
    ]);
    const selected = editableVertices(multipolygon).filter((vertex) =>
      (vertex.polygonIndex === 0 && vertex.ringIndex === 1 && vertex.vertexIndex === 1) ||
      (vertex.polygonIndex === 1 && vertex.ringIndex === 0 && vertex.vertexIndex === 2),
    );

    const result = deleteEditableVertices(multipolygon, selected);
    const coordinates = (result.geometry as MultiPolygon).getCoordinates();

    expect(result.removedCount).toBe(2);
    expect(result.blockedRingCount).toBe(0);
    expect(coordinates[0][1]).toEqual([[1, 1], [3, 3], [1, 3], [1, 1]]);
    expect(coordinates[1][0]).toEqual([[10, 10], [14, 10], [10, 14], [10, 10]]);
  });

  it("считает роли из текущей разметки", () => {
    expect(featureCounts(snapshot(["positive", "hard_negative", "positive"]).geojson)).toEqual({
      total: 3,
      positive: 2,
      hardNegative: 1,
    });
  });

  it("считает типы объектов снимка из сохранённого состояния и черновика", () => {
    const scene = {
      annotation_name: "scene.geojson",
      total_count: 3,
      positive_count: 2,
      hard_negative_count: 1,
      class_counts: { river: 2, lake: 0 },
    };
    expect(sceneClassCounts(scene, undefined)).toEqual({ river: 2, lake: 0 });

    const geojson = {
      type: "FeatureCollection",
      features: [
        { properties: { _mlsystem2_role: "positive", _mlsystem2_class: "river" } },
        { properties: { _mlsystem2_role: "positive", _mlsystem2_class: "lake" } },
        { properties: { _mlsystem2_role: "hard_negative" } },
      ],
    };
    const currentDraft = {
      baseline: { geojson, newFeatureIndexes: [] },
      current: { geojson, newFeatureIndexes: [] },
      history: [],
    };
    expect(featureClassCounts(geojson)).toEqual({ river: 1, lake: 1 });
    expect(sceneClassCounts(scene, currentDraft)).toEqual({ river: 1, lake: 1 });
  });

  it("не считает изменением только другой порядок полей GeoJSON", () => {
    const positive = {
      type: "Feature",
      id: "positive",
      properties: { _mlsystem2_role: "positive" },
      geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [0, 0]]] },
    };
    const negative = {
      type: "Feature",
      id: "negative",
      properties: { _mlsystem2_role: "hard_negative" },
      geometry: { type: "Polygon", coordinates: [[[2, 2], [3, 2], [2, 2]]] },
    };
    const baseline: DraftSnapshot = {
      geojson: {
        type: "FeatureCollection",
        crs: { type: "name", properties: { name: "EPSG:32637" } },
        features: [positive, negative],
      },
      newFeatureIndexes: [],
    };
    const current: DraftSnapshot = {
      geojson: {
        features: [negative, positive],
        crs: { properties: { name: "EPSG:32637" }, type: "name" },
        type: "FeatureCollection",
      },
      newFeatureIndexes: [],
    };
    expect(draftChanged({ baseline, current, history: [] })).toBe(false);
    expect(snapshotsEqual(baseline, current)).toBe(true);
    expect(snapshotsEqual(baseline, { ...current, newFeatureIndexes: [0] })).toBe(false);
  });

  it("сортирует по текущему количеству в обоих направлениях", () => {
    const scenes = [
      { annotation_name: "b.geojson", total_count: 5, positive_count: 5, hard_negative_count: 0 },
      { annotation_name: "a.geojson", total_count: 1, positive_count: 1, hard_negative_count: 0 },
    ];
    const drafts = {
      "a.geojson": draft("a.geojson", snapshot([]), snapshot(["positive", "positive"])),
    };
    expect(sortEditorScenes(scenes, drafts, "descending").map((item) => item.annotation_name)).toEqual([
      "b.geojson",
      "a.geojson",
    ]);
    expect(sortEditorScenes(scenes, drafts, "ascending").map((item) => item.annotation_name)).toEqual([
      "a.geojson",
      "b.geojson",
    ]);
  });

  it("ограничивает историю последними ста действиями", () => {
    let history: DraftSnapshot[] = [];
    for (let index = 0; index < 105; index += 1) {
      history = appendHistory(history, snapshot(Array(index).fill("positive")));
    }
    expect(history).toHaveLength(100);
    expect(featureCounts(history[0].geojson).total).toBe(5);
  });

  it("отменяет последний snapshot вместе с признаком нового объекта", () => {
    const baseline = snapshot([]);
    const previous = snapshot(["positive"], [0]);
    const current = snapshot(["positive", "hard_negative"], [0, 1]);
    const undone = undoDraft({ baseline, current, history: [previous] });
    expect(undone?.current).toEqual(previous);
    expect(undone?.history).toEqual([]);
  });

  it("формирует одну публикацию только из изменённых снимков", () => {
    const first = draft("b.geojson", snapshot([]), snapshot(["positive"], [0]));
    const second = draft("a.geojson", snapshot([]), snapshot(["hard_negative"], [0]));
    const clean = draft("c.geojson", snapshot(["positive"]), snapshot(["positive"]));
    expect(draftChanged(first)).toBe(true);
    expect(publishScenes({ first, second, clean })).toEqual([
      {
        annotation_name: "a.geojson",
        revision: "revision-a.geojson",
        geojson: second.current.geojson,
      },
      {
        annotation_name: "b.geojson",
        revision: "revision-b.geojson",
        geojson: first.current.geojson,
      },
    ]);

    const published = acceptPublishedDraft(first, {
      ...first.scene,
      revision: "new-revision",
      total_count: 1,
      positive_count: 1,
    });
    expect(draftChanged(published)).toBe(false);
    expect(published.current.newFeatureIndexes).toEqual([]);
    expect(published.history).toEqual([]);
  });
});
