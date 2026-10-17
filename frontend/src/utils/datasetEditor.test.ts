import { describe, expect, it } from "vitest";
import Collection from "ol/Collection";
import Feature from "ol/Feature";
import type OLMap from "ol/Map";
import View from "ol/View";
import GeoJSON from "ol/format/GeoJSON";
import MultiPolygon from "ol/geom/MultiPolygon";
import Polygon from "ol/geom/Polygon";
import Snap from "ol/interaction/Snap";

import {
  acceptPublishedDraft,
  appendHistory,
  deleteEditableVertices,
  datasetObjectTotals,
  draftChanged,
  editableVertexCoordinates,
  editableVertices,
  editableVerticesInExtent,
  editorSnapTargets,
  extendRasterResolutions,
  featureClassCounts,
  featureCounts,
  geometryInsideFootprint,
  isCurrentDatasetScene,
  isDeleteShortcut,
  isUndoShortcut,
  preventMapMiddleButtonDefault,
  polygonWithHole,
  publishScenes,
  sceneClassCounts,
  selectedDeleteAction,
  snapshotsEqual,
  sortEditorScenes,
  undoDraft,
  type CountedSceneWithDraft,
  type DraftSnapshot,
  type PublishableDraft,
} from "./datasetEditor";

const snapshot = (
  roles: string[],
  newFeatureIndexes: number[] = [],
  deleted = false,
): DraftSnapshot => ({
  geojson: {
    type: "FeatureCollection",
    features: roles.map((role) => ({
      type: "Feature",
      properties: { _mlsystem2_role: role },
      geometry: { type: "Polygon", coordinates: [] },
    })),
  },
  newFeatureIndexes,
  deleted,
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

describe("вырезы в полигонах общего редактора", () => {
  const rectangle = (left: number, bottom: number, right: number, top: number) => [
    [left, bottom], [right, bottom], [right, top], [left, top], [left, bottom],
  ];

  it.each([false, true])("добавляет настоящее отверстие независимо от направления контура: %s", (reverse) => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10)]);
    const ring = rectangle(2, 2, 4, 4);
    const result = polygonWithHole(polygon, new Polygon([reverse ? ring.reverse() : ring]));

    expect(result).toBeInstanceOf(Polygon);
    expect(result.getArea()).toBe(96);
    expect(result.intersectsCoordinate([3, 3])).toBe(false);
    expect(result.intersectsCoordinate([1, 1])).toBe(true);
    expect(editableVertices(result)).toHaveLength(8);
    expect((result as Polygon).getCoordinates()[0]).toEqual(polygon.getCoordinates()[0]);
    expect(polygon.getLinearRingCount()).toBe(1);
    expect(polygon.getArea()).toBe(100);
  });

  it("сохраняет прежние отверстия и все части составного объекта", () => {
    const polygon = new MultiPolygon([
      [rectangle(0, 0, 10, 10), rectangle(1, 1, 2, 2)],
      [rectangle(20, 0, 30, 10)],
    ]);
    const result = polygonWithHole(polygon, new Polygon([rectangle(22, 2, 24, 4)])) as MultiPolygon;

    expect(result.getCoordinates()[0]).toEqual(polygon.getCoordinates()[0]);
    expect(result.getCoordinates()[1]).toHaveLength(2);
    expect(result.getArea()).toBe(195);
    expect(result.intersectsCoordinate([23, 3])).toBe(false);
    expect(polygon.getCoordinates()[1]).toHaveLength(1);
  });

  it("находит отдельную часть, расположенную в отверстии другой части", () => {
    const polygon = new MultiPolygon([
      [rectangle(0, 0, 30, 30), rectangle(5, 5, 25, 25)],
      [rectangle(10, 10, 20, 20)],
    ]);
    const result = polygonWithHole(polygon, new Polygon([rectangle(12, 12, 14, 14)])) as MultiPolygon;
    expect(result.getCoordinates()[0]).toEqual(polygon.getCoordinates()[0]);
    expect(result.getCoordinates()[1]).toHaveLength(2);
    expect(result.getArea()).toBe(polygon.getArea() - 4);
  });

  it.each([
    rectangle(-1, 2, 2, 4),
    rectangle(0, 2, 2, 4),
    [[0, 0], [2, 1], [1, 2], [0, 0]],
    rectangle(11, 2, 13, 4),
  ].map((ring) => ({ ring })))("отклоняет выход за внешний контур и касание его границ: %j", ({ ring }) => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10)]);
    expect(() => polygonWithHole(polygon, new Polygon([ring]))).toThrow("целиком находиться внутри");
    expect(polygon.getLinearRingCount()).toBe(1);
  });

  it("проверяет рёбра выреза, а не только попадание вершин внутрь", () => {
    const concave = new Polygon([[
      [0, 0], [10, 0], [10, 10], [6, 10], [6, 4], [4, 4], [4, 10], [0, 10], [0, 0],
    ]]);
    const hole = new Polygon([[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]]);
    expect(hole.getCoordinates()[0].every((point) => concave.intersectsCoordinate(point))).toBe(true);
    expect(() => polygonWithHole(concave, hole)).toThrow("целиком находиться внутри");
  });

  it.each([
    rectangle(3, 3, 4, 4),
    rectangle(1, 1, 6, 6),
    rectangle(4, 4, 7, 7),
    rectangle(5, 3, 7, 4),
  ].map((ring) => ({ ring })))("не пересекает, не касается и не охватывает прежнее отверстие: %j", ({ ring }) => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10), rectangle(2, 2, 5, 5)]);
    expect(() => polygonWithHole(polygon, new Polygon([ring]))).toThrow("существующих отверстий");
    expect(polygon.getArea()).toBe(91);
  });

  it("не объединяет две части MultiPolygon одним вырезом", () => {
    const polygon = new MultiPolygon([[rectangle(0, 0, 10, 10)], [rectangle(11, 0, 20, 10)]]);
    expect(() => polygonWithHole(polygon, new Polygon([rectangle(2, 2, 18, 4)])))
      .toThrow("одной части");
  });

  it.each([
    [[1, 1], [5, 5], [1, 5], [5, 1], [1, 1]],
    [[1, 1], [5, 5], [1, 5], [4, 1], [1, 1]],
    [[1, 1], [5, 1], [3, 1], [3, 5], [1, 5], [1, 1]],
    [[1, 1], [2, 2], [3, 3], [1, 1]],
    [[1, 1], [2, 2], [1, 1]],
    [[1, 1], [NaN, 2], [3, 3], [1, 1]],
  ].map((ring) => ({ ring })))("не допускает самопересечение, возврат по ребру и вырожденный контур: %j", ({ ring }) => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10)]);
    expect(() => polygonWithHole(polygon, new Polygon([ring]))).toThrow();
    expect(polygon.getArea()).toBe(100);
  });

  it("не теряет точность на координатах EPSG:3857 и не меняет размерность", () => {
    const polygon = new Polygon([rectangle(11_478_026, 6_962_204, 11_478_036, 6_962_214)
      .map((point) => [...point, 10])]);
    const result = polygonWithHole(polygon, new Polygon([
      rectangle(11_478_028, 6_962_206, 11_478_030, 6_962_208),
    ])) as Polygon;
    expect(result.getLayout()).toBe("XYZ");
    expect(result.getCoordinates()[1].every((point) => point.length === 3 && point[2] === 10)).toBe(true);
    expect(result.getArea()).toBe(96);
  });

  it("отменяет вырез целиком через общую историю черновика", () => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10)]);
    const before: DraftSnapshot = {
      geojson: { type: "FeatureCollection", features: [{
        type: "Feature", id: "река-1", properties: { имя: "Река", _mlsystem2_role: "positive" },
        geometry: { type: "Polygon", coordinates: polygon.getCoordinates() },
      }] },
      newFeatureIndexes: [],
    };
    const after = JSON.parse(JSON.stringify(before)) as DraftSnapshot;
    const feature = (after.geojson.features as Record<string, unknown>[])[0];
    feature.geometry = { type: "Polygon", coordinates: (polygonWithHole(polygon,
      new Polygon([rectangle(2, 2, 4, 4)])) as Polygon).getCoordinates() };
    const edited = { baseline: before, current: after, history: appendHistory([], before) };
    expect(draftChanged(edited)).toBe(true);
    expect(featureCounts(after.geojson).total).toBe(1);
    expect(undoDraft(edited)?.current).toEqual(before);
    expect(feature.id).toBe("река-1");
    expect(feature.properties).toEqual((before.geojson.features as Record<string, unknown>[])[0].properties);
  });

  it.each([[0, 1], [0, 1, 2], [0, 1, 2, 3]].map((indexes) => ({ indexes })))(
    "удаляет отверстие целиком, если после Del осталось бы меньше трёх вершин: %j",
    ({ indexes }) => {
      const polygon = new Polygon([rectangle(0, 0, 10, 10), rectangle(2, 2, 4, 4)]);
      const selected = editableVertices(polygon).filter((vertex) =>
        vertex.ringIndex === 1 && indexes.includes(vertex.vertexIndex),
      );

      const result = deleteEditableVertices(polygon, selected);

      expect(result.removedCount).toBe(4);
      expect(result.blockedRingCount).toBe(0);
      expect((result.geometry as Polygon).getCoordinates()).toEqual([rectangle(0, 0, 10, 10)]);
      expect((result.geometry as Polygon).getArea()).toBe(100);
      expect((result.geometry as Polygon).intersectsCoordinate([3, 3])).toBe(true);
      expect(polygon.getLinearRingCount()).toBe(2);
    },
  );

  it("удаляет треугольное отверстие при удалении одной вершины", () => {
    const polygon = new Polygon([
      rectangle(0, 0, 10, 10),
      [[2, 2], [4, 2], [3, 4], [2, 2]],
    ]);
    const selected = editableVertices(polygon).filter((vertex) =>
      vertex.ringIndex === 1 && vertex.vertexIndex === 0,
    );

    const result = deleteEditableVertices(polygon, selected);

    expect(result.removedCount).toBe(3);
    expect(result.blockedRingCount).toBe(0);
    expect((result.geometry as Polygon).getCoordinates()).toEqual([rectangle(0, 0, 10, 10)]);
  });

  it("удаляет несколько отверстий без сдвига индексов остальных колец и частей", () => {
    const polygon = new MultiPolygon([
      [
        rectangle(0, 0, 10, 10),
        rectangle(1, 1, 2, 2),
        rectangle(3, 3, 4, 4),
        rectangle(5, 5, 6, 6),
      ],
      [rectangle(20, 0, 30, 10), rectangle(21, 1, 22, 2)],
    ]);
    const selected = editableVertices(polygon).filter((vertex) =>
      (vertex.polygonIndex === 0 && (vertex.ringIndex === 1 || vertex.ringIndex === 2)) ||
      (vertex.polygonIndex === 1 && vertex.ringIndex === 1 && vertex.vertexIndex === 0),
    );

    const result = deleteEditableVertices(polygon, selected);
    const coordinates = (result.geometry as MultiPolygon).getCoordinates();

    expect(result.removedCount).toBe(9);
    expect(result.blockedRingCount).toBe(0);
    expect(coordinates[0]).toEqual([rectangle(0, 0, 10, 10), rectangle(5, 5, 6, 6)]);
    expect(coordinates[1]).toEqual([
      rectangle(20, 0, 30, 10), [[22, 1], [22, 2], [21, 2], [22, 1]],
    ]);
    expect(polygon.getCoordinates()[0]).toHaveLength(4);
  });

  it.each([2, 4])("не удаляет внешний контур, но удаляет выбранное отверстие: %i", (outerCount) => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10), rectangle(2, 2, 4, 4)]);
    const selected = editableVertices(polygon).filter((vertex) =>
      vertex.ringIndex === 1 || vertex.vertexIndex < outerCount,
    );
    const result = deleteEditableVertices(polygon, selected);

    expect(result.removedCount).toBe(4);
    expect(result.blockedRingCount).toBe(1);
    expect((result.geometry as Polygon).getCoordinates()).toEqual([polygon.getCoordinates()[0]]);
  });

  it("сохраняет удаление отверстия в GeoJSON и восстанавливает его одной отменой", () => {
    const polygon = new Polygon([rectangle(0, 0, 10, 10), rectangle(2, 2, 4, 4)]);
    const feature = new Feature({ geometry: polygon, _mlsystem2_role: "positive", имя: "Река" });
    feature.setId("река-1");
    const format = new GeoJSON();
    const options = { dataProjection: "EPSG:3857", featureProjection: "EPSG:3857" };
    const before: DraftSnapshot = {
      geojson: { ...format.writeFeaturesObject([feature], options) }, newFeatureIndexes: [],
    };
    const result = deleteEditableVertices(polygon,
      editableVertices(polygon).filter((vertex) => vertex.ringIndex === 1));
    feature.setGeometry(result.geometry as Polygon);
    const after: DraftSnapshot = {
      geojson: { ...format.writeFeaturesObject([feature], options) }, newFeatureIndexes: [],
    };
    const saved = JSON.parse(JSON.stringify(after)) as DraftSnapshot;
    const [loaded] = format.readFeatures(saved.geojson, options);

    expect((loaded.getGeometry() as Polygon).getCoordinates()).toEqual([rectangle(0, 0, 10, 10)]);
    expect(loaded.getId()).toBe("река-1");
    expect(loaded.get("имя")).toBe("Река");
    expect(loaded.get("_mlsystem2_role")).toBe("positive");
    expect(featureCounts(saved.geojson).total).toBe(1);
    const edited = { baseline: before, current: saved, history: appendHistory([], before) };
    expect(draftChanged(edited)).toBe(true);
    expect(undoDraft(edited)?.current).toEqual(before);
    const [restored] = format.readFeatures(undoDraft(edited)!.current.geojson, options);
    expect((restored.getGeometry() as Polygon).getCoordinates()).toEqual(polygon.getCoordinates());
  });
});

describe("черновики редактора датасетов", () => {
  it("воспроизводит отскок настоящего Snap и устраняет его без отключения привязки к соседям", () => {
    const selected = new Feature(new Polygon([[
      [100, 100], [400, 100], [400, 400], [100, 400], [100, 100],
    ]]));
    const neighbor = new Feature(new Polygon([[
      [430, 100], [470, 100], [470, 200], [430, 200], [430, 100],
    ]]));
    const view = new View({ projection: "EPSG:3857", resolution: 1, center: [0, 0] });
    const map = {
      getView: () => view,
      getPixelFromCoordinate: (coordinate: number[]) => coordinate,
    } as unknown as OLMap;
    const beforeFeatures = new Collection([selected, neighbor]);
    const afterFeatures = new Collection(editorSnapTargets([selected, neighbor], [selected], true));
    const before = new Snap({ features: beforeFeatures });
    const after = new Snap({ features: afterFeatures });
    before.setMap(map);
    after.setMap(map);
    try {
      expect(before.snapTo([109, 107], [109, 107], map)?.vertex).toEqual([109, 100]);
      expect(after.snapTo([109, 107], [109, 107], map)).toBeNull();
      expect(after.snapTo([425, 100], [425, 100], map)?.vertex).toEqual([430, 100]);
    } finally {
      beforeFeatures.clear();
      afterFeatures.clear();
    }
  });

  it("при правке не привязывает вершину к своему старому контуру, сохраняя привязку к соседям", () => {
    const selected = new Polygon([[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]);
    const neighbor = new Polygon([[[5, 0], [8, 0], [8, 4], [5, 4], [5, 0]]]);
    const visible = [selected, neighbor];
    expect(editorSnapTargets(visible, [selected], true)).toEqual([neighbor]);
    expect(editorSnapTargets(visible, [selected, neighbor], true)).toEqual([]);
    expect(editorSnapTargets(visible, [], true)).toEqual(visible);
    expect(editorSnapTargets(visible, [selected], false)).toEqual(visible);
    expect(visible).toEqual([selected, neighbor]);
  });

  it("не запрашивает сцену предыдущего датасета после переключения", () => {
    expect(isCurrentDatasetScene("новый", "старый", "старая.geojson", "старая.geojson"))
      .toBe(false);
    expect(isCurrentDatasetScene("новый", "новый", "старая.geojson", ""))
      .toBe(false);
    expect(isCurrentDatasetScene("новый", "новый", "новая.geojson", "новая.geojson"))
      .toBe(true);
  });

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

  it("распознаёт отмену на физической клавише Z в латинской и русской раскладках", () => {
    expect(isUndoShortcut({
      altKey: false,
      code: "KeyZ",
      ctrlKey: true,
      key: "z",
      metaKey: false,
    })).toBe(true);
    expect(isUndoShortcut({
      altKey: false,
      code: "KeyZ",
      ctrlKey: true,
      key: "я",
      metaKey: false,
    })).toBe(true);
    expect(isUndoShortcut({
      altKey: false,
      code: "",
      ctrlKey: false,
      key: "Я",
      metaKey: true,
    })).toBe(true);
    expect(isUndoShortcut({
      altKey: true,
      code: "KeyZ",
      ctrlKey: true,
      key: "я",
      metaKey: false,
    })).toBe(false);
    expect(isUndoShortcut({
      altKey: false,
      code: "KeyY",
      ctrlKey: true,
      key: "н",
      metaKey: false,
    })).toBe(false);
  });

  it("удаляет клавишей Del сначала вершины, а без них — выделенный полигон", () => {
    expect(isDeleteShortcut({ code: "Delete", key: "Delete" })).toBe(true);
    expect(isDeleteShortcut({ code: "", key: "Del" })).toBe(true);
    expect(isDeleteShortcut({ code: "Backspace", key: "Backspace" })).toBe(true);
    expect(isDeleteShortcut({ code: "KeyD", key: "d" })).toBe(false);
    expect(selectedDeleteAction(2, 1)).toBe("vertices");
    expect(selectedDeleteAction(0, 1)).toBe("features");
    expect(selectedDeleteAction(0, 0)).toBeNull();
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

  it("суммирует объекты всего датасета по текущим черновикам и не считает удалённые снимки", () => {
    const localGeojson = {
      type: "FeatureCollection",
      features: [
        { properties: { _mlsystem2_role: "positive", _mlsystem2_class: "river" } },
        { properties: { _mlsystem2_role: "hard_negative" } },
      ],
    };
    const localDraft = {
      baseline: { geojson: localGeojson, newFeatureIndexes: [] },
      current: { geojson: localGeojson, newFeatureIndexes: [] },
      history: [],
    };
    const scenes: CountedSceneWithDraft[] = [
      {
        annotation_name: "server.geojson",
        total_count: 9,
        positive_count: 9,
        hard_negative_count: 0,
        class_counts: { river: 9 },
        draft: {
          positive_count: 2,
          hard_negative_count: 1,
          class_counts: { lake: 2 },
        },
      },
      {
        annotation_name: "local.geojson",
        total_count: 7,
        positive_count: 7,
        hard_negative_count: 0,
        class_counts: { lake: 7 },
      },
      {
        annotation_name: "deleted.geojson",
        total_count: 4,
        positive_count: 4,
        hard_negative_count: 0,
        class_counts: { river: 4 },
        draft: {
          deleted: true,
          positive_count: 4,
          hard_negative_count: 0,
          class_counts: { river: 4 },
        },
      },
    ];

    expect(datasetObjectTotals(scenes, { "local.geojson": localDraft })).toEqual({
      positive: 3,
      hardNegative: 2,
      classCounts: { lake: 2, river: 1 },
    });
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

  it("считает удаление снимка изменением и отменяет его через общую историю", () => {
    const baseline = snapshot(["positive"]);
    const deleted = snapshot(["positive"], [], true);
    const state = { baseline, current: deleted, history: [baseline] };

    expect(draftChanged(state)).toBe(true);
    expect(snapshotsEqual(baseline, deleted)).toBe(false);
    expect(undoDraft(state)?.current.deleted).toBe(false);
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
