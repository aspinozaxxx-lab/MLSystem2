import { describe, expect, it } from "vitest";
import Polygon from "ol/geom/Polygon";
import MultiPolygon from "ol/geom/MultiPolygon";
import Feature from "ol/Feature";
import GeoJSON from "ol/format/GeoJSON";
import type Geometry from "ol/geom/Geometry";
import VectorSource from "ol/source/Vector";
import type { AnnotationFeatureHistoryEntry } from "./api/types";
import { polygonHistoryPreview } from "./DatasetPolygonHistory";
import { clipEditorGeometry, editorGeometryNeedsUntangling, synchronizeEditorSource } from "./utils/datasetEditor";

describe("автоматическая обрезка контуров", () => {
  const footprint = new Polygon([[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], [[4, 4], [4, 6], [6, 6], [6, 4], [4, 4]]]);
  it("сохраняет внутренний пропуск данных", () => {
    const source = new Polygon([[[-2, -2], [12, -2], [12, 12], [-2, 12], [-2, -2]]]);
    const clipped = clipEditorGeometry(source, footprint) as Polygon;
    expect(clipped.getArea()).toBe(96);
    expect(clipped.getCoordinates()).toHaveLength(2);
    expect(source.getArea()).toBe(196);
  });
  it("проверяет не только вершины: полигон может перекрыть отверстие", () => {
    const source = new Polygon([[[1, 1], [9, 1], [9, 9], [1, 9], [1, 1]]]);
    expect((clipEditorGeometry(source, footprint) as Polygon).getArea()).toBe(60);
  });
  it("сохраняет раздельные части как один MultiPolygon", () => {
    const parts = new MultiPolygon([[[[0, 0], [4, 0], [4, 10], [0, 10], [0, 0]]], [[[6, 0], [10, 0], [10, 10], [6, 10], [6, 0]]]]);
    const source = new Polygon([[[-1, -1], [11, -1], [11, 11], [-1, 11], [-1, -1]]]);
    const clipped = clipEditorGeometry(source, parts) as MultiPolygon;
    expect(clipped.getType()).toBe("MultiPolygon");
    expect(clipped.getPolygons()).toHaveLength(2);
    expect(clipped.getArea()).toBe(80);
  });
  it("не меняет вершины уже допустимого контура", () => {
    const source = new Polygon([[[1, 1], [1, 2], [1, 3], [3, 3], [3, 1], [1, 1]]]);
    const clipped = clipEditorGeometry(source, footprint);
    expect(clipped).not.toBe(source);
    expect((clipped as Polygon).getCoordinates()).toEqual(source.getCoordinates());
  });
  it("возвращает пустое пересечение при касании и полностью внешнем контуре", () => {
    const source = new Polygon([[[10, 0], [12, 0], [12, 10], [10, 10], [10, 0]]]);
    expect(clipEditorGeometry(source, footprint)).toBeNull();
  });
  it.each([0, -3, 20])("не добавляет вершину в самопересечение даже за footprint: смещение %s", (offset) => {
    const coordinates = [[[offset, 1], [offset + 8, 9], [offset, 9], [offset + 8, 1], [offset, 1]]];
    const source = new Polygon(coordinates);
    expect(editorGeometryNeedsUntangling(source)).toBe(true);
    const saved = clipEditorGeometry(source, footprint) as Polygon;
    expect(saved.getType()).toBe("Polygon");
    expect(saved.getCoordinates()).toEqual(coordinates);
    expect(source.getCoordinates()).toEqual(coordinates);
    const feature = new Feature<Geometry>({ geometry: saved });
    const restored = new GeoJSON().readFeature(new GeoJSON().writeFeatureObject(feature)) as Feature<Geometry>;
    expect((restored.getGeometry() as Polygon).getCoordinates()).toEqual(coordinates);
  });
  it("снова обрезает контур после распутывания", () => {
    const source = new Polygon([[[-2, -2], [12, 12], [-2, 12], [12, -2], [-2, -2]]]);
    const editable = clipEditorGeometry(source, footprint) as Polygon;
    editable.setCoordinates([[[-2, -2], [12, -2], [12, 12], [-2, 12], [-2, -2]]]);
    expect(editorGeometryNeedsUntangling(editable)).toBe(false);
    const result = clipEditorGeometry(editable, footprint) as Polygon;
    expect(result.getArea()).toBe(96);
    expect(result.getCoordinates()).toHaveLength(2);
  });
  it("не перестраивает пересекающиеся кольца и наложенные части", () => {
    const crossedRings = new Polygon([[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]],
      [[6, 2], [10, 2], [10, 6], [6, 6], [6, 2]]]);
    const overlapped = new MultiPolygon([[[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]]],
      [[[4, 4], [12, 4], [12, 12], [4, 12], [4, 4]]]]);
    for (const geometry of [crossedRings, overlapped]) {
      expect(editorGeometryNeedsUntangling(geometry)).toBe(true);
      expect((clipEditorGeometry(geometry, footprint) as Polygon | MultiPolygon).getCoordinates()).toEqual(geometry.getCoordinates());
    }
  });
});

describe("просмотр опубликованных состояний", () => {
  const feature = (id: string) => ({ type: "Feature", id, properties: {}, geometry: { type: "Polygon", coordinates: [] } });
  const entries: AnnotationFeatureHistoryEntry[] = [
    { id: "исходный", action: "baseline", feature: feature("исходный"), crs: { properties: { name: "EPSG:3857" } } },
    { id: "правка", action: "updated", feature: feature("новый"), username: "Автор", published_at: "2026-10-19T10:00:00Z" },
    { id: "удаление", action: "deleted", feature: null },
  ];
  it("показывает отдельные контуры до и после, не меняя историю", () => {
    const original = JSON.stringify(entries);
    expect(polygonHistoryPreview(entries, 1).map((item) => [item.feature.id, item.before])).toEqual([["исходный", true], ["новый", false]]);
    expect(JSON.stringify(entries)).toBe(original);
  });
  it("не создаёт фиктивную геометрию до рождения или после удаления", () => {
    expect(polygonHistoryPreview(entries, 0)).toHaveLength(1);
    expect(polygonHistoryPreview(entries, 2).map((item) => item.before)).toEqual([true]);
  });
});

describe("согласование карты с автосохранением", () => {
  it("добавляет серверные свойства без потери выбранного объекта и перемещения вершин", () => {
    const feature = new Feature<Geometry>({ geometry: new Polygon([[[1, 1], [3, 1], [3, 3], [1, 1]]]), _mlsystem2_role: "positive" });
    feature.setId("полигон");
    const geometry = feature.getGeometry();
    const source = new VectorSource<Feature<Geometry>>({ features: [feature] });
    const payload = new GeoJSON().writeFeaturesObject([feature]);
    payload.features[0].properties!._mlsystem2_origin_key = "manual:полигон";
    const before = JSON.stringify(payload);
    expect(synchronizeEditorSource(source, payload, "EPSG:3857")).toBe(false);
    expect(source.getFeatureById("полигон")).toBe(feature);
    expect(feature.getGeometry()).toBe(geometry);
    expect(feature.get("_mlsystem2_origin_key")).toBe("manual:полигон");
    expect(JSON.stringify(payload)).toBe(before);
  });

  it("показывает серверную обрезку, сохраняя ID выбранного полигона", () => {
    const feature = new Feature<Geometry>({ geometry: new Polygon([[[-1, 0], [3, 0], [3, 3], [-1, 0]]]) });
    feature.setId("полигон");
    const source = new VectorSource<Feature<Geometry>>({ features: [feature] });
    const saved = feature.clone(); saved.setId("полигон");
    saved.setGeometry(new Polygon([[[0, 0], [3, 0], [3, 3], [0, 0]]]));
    expect(synchronizeEditorSource(source, new GeoJSON().writeFeaturesObject([saved]), "EPSG:3857")).toBe(true);
    expect(source.getFeatureById("полигон")).toBe(feature);
    expect(feature.getGeometry()?.getExtent()).toEqual([0, 0, 3, 3]);
  });
});
