import { describe, expect, it, vi } from "vitest";
import Feature from "ol/Feature";
import type Geometry from "ol/geom/Geometry";
import VectorSource from "ol/source/Vector";
import Collection from "ol/Collection";
import type OLMap from "ol/Map";
import MapBrowserEvent from "ol/MapBrowserEvent";
import Observable from "ol/Observable";
import Translate from "ol/interaction/Translate";
import View from "ol/View";
import Polygon from "ol/geom/Polygon";
import MultiPolygon from "ol/geom/MultiPolygon";
import { getCenter } from "ol/extent";
import { createTestRegionObjectCounter, replaceTestRegionLabels, resizeTestRegion, squareTestRegion, testRegionObjectCountStyle, TEST_REGION_SIZES, testRegionSize, testRegions, translateTestRegion } from "./testRegions";

describe("Тестовые области", () => {
  it("привязывает квадрат к сетке исходного растра, включая поворот и выход за край", () => {
    const grid = [0.25, 0.03, 4871900, 0.02, -0.25, 7711230];
    const geometry = squareTestRegion([grid[2] + 1, grid[5] - 2], 512, grid);
    const [a, b, c, d, e, f] = grid, determinant = a * e - b * d;
    const pixels = geometry.getCoordinates()[0].map(([x, y]) => [
      (e * (x - c) - b * (y - f)) / determinant,
      (-d * (x - c) + a * (y - f)) / determinant,
    ]);
    expect(pixels[1][0] - pixels[0][0]).toBeCloseTo(512, 6);
    expect(pixels[2][1] - pixels[1][1]).toBeCloseTo(512, 6);
    expect(pixels[0][0]).toBeLessThan(0);
    expect(pixels[0][0]).toBeCloseTo(Math.round(pixels[0][0]), 6);
    expect(pixels[0]).toEqual(pixels[4]);
    expect(TEST_REGION_SIZES).toEqual(Array.from({ length: 16 }, (_, i) => (i + 1) * 512));
  });

  it("при замене версии меняет эталонные объекты и сохраняет области с отверстиями и ID", () => {
    const region = { id: "область", properties: { _mlsystem2_role: "annotation_zone" },
      geometry: { type: "Polygon", coordinates: [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], [[.5, .5], [1, .5], [1, 1], [.5, 1], [.5, .5]]] } };
    const previous = { features: [{ id: "старый" }, region] };
    const labels = { features: [{ id: "новый" }, { id: "зона обучения", properties: { _mlsystem2_role: "annotation_zone" } }] };
    const next = replaceTestRegionLabels(previous, labels);
    expect(next.features).toEqual([{ id: "новый" }, region]);
    expect(testRegions(next)).toEqual([region]);
    expect(previous.features).toHaveLength(2);
  });

  it("меняет размер рамки вокруг того же центра без накопления сдвига на повёрнутой сетке", () => {
    const grid = [0.25, 0.03, 4871900, 0.02, -0.25, 7711230];
    const initial = squareTestRegion([4872123.12, 7711093.46], 512, grid);
    let current = initial;
    for (const size of [1024, 8192, 1536, 512, 512]) {
      current = resizeTestRegion(current, size, grid);
      expect(testRegionSize(current, grid)).toBe(size);
      getCenter(current.getExtent()).forEach((value, i) => expect(value).toBeCloseTo(getCenter(initial.getExtent())[i], 7));
    }
    expect(current.getCoordinates()).toEqual(initial.getCoordinates());
  });

  it("свободными остаются отверстия, несколько частей, неверный размер и повреждённый контур", () => {
    const grid = [1, 0, 0, 0, -1, 1024];
    const square = squareTestRegion([512, 512], 512, grid);
    expect(testRegionSize(new MultiPolygon([square.getCoordinates()]), grid)).toBe(512);
    const withHole = square.clone();
    withHole.setCoordinates([...withHole.getCoordinates(), [[400, 400], [410, 400], [410, 410], [400, 410], [400, 400]]]);
    expect(testRegionSize(withHole, grid)).toBe(0);
    expect(testRegionSize(new MultiPolygon([square.getCoordinates(), square.getCoordinates()]), grid)).toBe(0);
    expect(testRegionSize(squareTestRegion([512, 512], 513, grid), grid)).toBe(0);
    expect(testRegionSize(new Polygon([[[0, 0], [512, 0], [0, 0], [0, 512], [0, 0]]]), grid)).toBe(0);
  });

  it("переносит все части и отверстия одинаково, не меняя исходный объект и размеры", () => {
    const grid = [0.25, 0.03, 4871900, 0.02, -0.25, 7711230];
    const source = new MultiPolygon([
      [[[1, 2], [11, 3], [9, 12], [1, 2]], [[4, 5], [6, 5], [5, 7], [4, 5]]],
      [[[20, 20], [30, 20], [30, 30], [20, 20]]],
    ]);
    const before = source.getCoordinates();
    const dx = grid[0] * 17.2 + grid[1] * -23.4, dy = grid[3] * 17.2 + grid[4] * -23.4;
    const moved = translateTestRegion(source, [dx, dy], grid) as MultiPolygon;
    expect(source.getCoordinates()).toEqual(before);
    expect(moved.getArea()).toBeCloseTo(source.getArea(), 9);
    moved.getCoordinates().forEach((polygon, p) => polygon.forEach((ring, r) => ring.forEach((point, i) => {
      expect(point[0] - before[p][r][i][0]).toBeCloseTo(grid[0] * 17 - grid[1] * 23, 9);
      expect(point[1] - before[p][r][i][1]).toBeCloseTo(grid[3] * 17 - grid[4] * 23, 9);
    })));
    const square = squareTestRegion([4872123, 7711093], 512, grid);
    expect(testRegionSize(translateTestRegion(square, [dx, dy], grid), grid)).toBe(512);
  });
});

function rectangle(left: number, bottom: number, right: number, top: number) {
  return new Polygon([[[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]]);
}

describe("число полигонов внутри тестовой области", () => {
  it("считает раздельные части обрезанного объекта прямо во время переноса, без изменения исходника", () => {
    const geometry = new Polygon([[[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10], [0, 0]]]);
    const before = geometry.getCoordinates(), object = new Feature<Geometry>(geometry);
    const region = rectangle(-1, 4, 11, 11), zone = new Feature<Geometry>({ geometry: region, _mlsystem2_role: "annotation_zone" });
    const source = new VectorSource({ features: [object, zone] });
    const counter = createTestRegionObjectCounter(source, null);
    try {
      expect(counter.count(region)).toBe(2);
      region.translate(0, -8);
      expect(counter.count(region)).toBe(1);
      region.translate(30, 0);
      expect(counter.count(region)).toBe(0);
      expect(geometry.getCoordinates()).toEqual(before);
      expect(source.getFeatures()).toHaveLength(2);
      expect(object.getGeometry()).toBe(geometry);
    } finally { counter.dispose(); }
  });

  it("отличает части MultiPolygon от отверстий и исключает касания без площади", () => {
    const outer = rectangle(0, 0, 10, 10);
    const withHole = [...outer.getCoordinates(), rectangle(3, 3, 7, 7).getCoordinates()[0]];
    const geometry = new MultiPolygon([withHole, rectangle(20, 0, 25, 5).getCoordinates()]);
    const source = new VectorSource<Feature<Geometry>>({ features: [new Feature(geometry)] });
    const counter = createTestRegionObjectCounter(source, null);
    try {
      expect(counter.count(rectangle(-1, -1, 30, 11))).toBe(2);
      expect(counter.count(rectangle(4, 4, 6, 6))).toBe(0);
      expect(counter.count(rectangle(10, 0, 12, 8))).toBe(0);
      expect(counter.count(rectangle(25, 5, 27, 7))).toBe(0);
      expect(counter.count(new Polygon([rectangle(-1, -1, 30, 11).getCoordinates()[0], rectangle(-.5, -.5, 10.5, 10.5).getCoordinates()[0]]))).toBe(1);
    } finally { counter.dispose(); }
  });

  it("учитывает nodata и не считает зоны, отрицательные примеры и скрытие слоя класса", () => {
    const footprint = new MultiPolygon([rectangle(0, 0, 3, 10).getCoordinates(), rectangle(7, 0, 10, 10).getCoordinates()]);
    const source = new VectorSource<Feature<Geometry>>({ features: [
      new Feature({ geometry: rectangle(1, 1, 9, 9), _mlsystem2_class: "участки" }),
      new Feature({ geometry: rectangle(1, 1, 2, 2), _mlsystem2_role: "hard_negative" }),
      new Feature({ geometry: rectangle(1, 1, 2, 2), _mlsystem2_role: "annotation_zone" }),
    ] });
    const counter = createTestRegionObjectCounter(source, footprint);
    try {
      expect(counter.count(rectangle(-5, -5, 15, 15))).toBe(2);
      expect(counter.count(rectangle(4, 1, 6, 9))).toBe(0);
      expect(counter.count(rectangle(-10, -10, -1, -1))).toBe(0);
    } finally { counter.dispose(); }
  });

  it("проверяет внутренние отверстия и вогнутый полезный контур, а не только его охватывающую рамку", () => {
    const footprint = new Polygon([
      [[0, 0], [10, 0], [10, 6], [6, 6], [6, 10], [0, 10], [0, 0]],
      rectangle(2, 2, 4, 4).getCoordinates()[0],
    ]);
    const source = new VectorSource<Feature<Geometry>>({ features: [
      new Feature(rectangle(8, 8, 9, 9)), new Feature(rectangle(2.5, 2.5, 3.5, 3.5)),
      new Feature(rectangle(1, 2.5, 5, 3.5)), new Feature(rectangle(.2, .2, .8, .8)),
    ] });
    const counter = createTestRegionObjectCounter(source, footprint);
    try { expect(counter.count(rectangle(-1, -1, 11, 11))).toBe(3); } finally { counter.dispose(); }
  });

  it("обновляет число настоящим взаимодействием Translate до отпускания кнопки мыши", () => {
    const object = new Feature<Geometry>(new Polygon([[[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10], [0, 0]]]));
    const region = rectangle(-1, 4, 11, 11), zone = new Feature<Geometry>({ geometry: region, _mlsystem2_role: "annotation_zone" });
    const source = new VectorSource({ features: [object, zone] });
    const counter = createTestRegionObjectCounter(source, null), ended = vi.fn();
    const map = Object.assign(new Observable(), {
      getView: () => new View({ projection: "EPSG:3857" }), isRendered: () => true,
      getViewport: () => ({ classList: { add() {}, remove() {} } }),
      forEachFeatureAtPixel: (_: unknown, callback: (feature: Feature<Geometry>, layer: unknown) => unknown) => callback(zone, null),
    }) as unknown as OLMap;
    const interaction = new Translate({ features: new Collection([zone]), condition: () => true });
    interaction.setMap(map); interaction.on("translateend", ended);
    const send = (type: string, coordinate: number[]) => {
      const pointer = Object.assign(new Event(type), { button: 0, pointerId: 1 }) as PointerEvent;
      const event = new MapBrowserEvent(type, map, pointer, type === "pointerdrag", undefined, type === "pointerup" ? [] : [pointer]);
      event.coordinate = coordinate; event.pixel = coordinate;
      interaction.handleEvent(event);
    };
    try {
      expect(counter.count(region)).toBe(2);
      send("pointerdown", [1, 5]); send("pointerdrag", [1, -3]);
      expect(ended).not.toHaveBeenCalled();
      expect(counter.count(region)).toBe(1);
      send("pointerup", [1, -3]); expect(ended).toHaveBeenCalledOnce();
    } finally { interaction.dispose(); counter.dispose(); }
  });

  it("обновляет кэш при смене версии объектов, но не пересчитывает неподвижные области во время переноса соседней", () => {
    const source = new VectorSource<Feature<Geometry>>();
    const object = new Feature<Geometry>(rectangle(1, 1, 2, 2));
    source.addFeature(object);
    const first = rectangle(0, 0, 5, 5), second = rectangle(0, 0, 10, 10);
    const zone = new Feature<Geometry>({ geometry: first, _mlsystem2_role: "annotation_zone" });
    source.addFeature(zone);
    const query = vi.spyOn(source, "getFeaturesInExtent"), counter = createTestRegionObjectCounter(source, null);
    try {
      expect(counter.count(first)).toBe(1);
      expect(counter.count(second)).toBe(1);
      for (let i = 0; i < 10; i++) expect(counter.count(second)).toBe(1);
      expect(query).toHaveBeenCalledTimes(2);
      first.translate(20, 0);
      expect(counter.count(first)).toBe(0);
      expect(counter.count(second)).toBe(1);
      expect(query).toHaveBeenCalledTimes(3);
      object.getGeometry()!.translate(20, 0);
      expect(counter.count(second)).toBe(0);
      object.setGeometry(rectangle(1, 1, 2, 2));
      expect(counter.count(second)).toBe(1);
      object.set("_mlsystem2_role", "annotation_zone");
      expect(counter.count(second)).toBe(0);
      object.set("_mlsystem2_role", "positive");
      expect(counter.count(second)).toBe(1);
      source.removeFeature(object);
      expect(counter.count(second)).toBe(0);
      source.addFeature(new Feature(rectangle(1, 1, 2, 2)));
      expect(counter.count(second)).toBe(1);
      source.clear();
      expect(counter.count(second)).toBe(0);
    } finally { counter.dispose(); query.mockRestore(); }
  });

  it("не теряет части при больших мировых координатах и на повёрнутой рамке", () => {
    const geometry = new Polygon([[[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10], [0, 0]]]);
    const region = new Polygon([[[-1, 4], [11, 5], [11, 11], [-1, 11], [-1, 4]]]);
    geometry.translate(4871900, 7711230); region.translate(4871900, 7711230);
    const source = new VectorSource<Feature<Geometry>>({ features: [new Feature(geometry)] });
    const counter = createTestRegionObjectCounter(source, rectangle(4871800, 7711130, 4872000, 7711330));
    try { expect(counter.count(region)).toBe(2); } finally { counter.dispose(); }
  });

  it("ставит только число внутри правого верхнего угла цветом рамки, включая ноль и выделение", () => {
    for (const [count, color] of [[0, "#A855F7"], [17, "#38BDF8"]] as const) {
      const style = testRegionObjectCountStyle(rectangle(10, 20, 40, 50), count, color);
      expect(style.getText()?.getText()).toBe(String(count));
      expect(style.getText()?.getFill()?.getColor()).toBe(color);
      expect(style.getText()?.getTextAlign()).toBe("right");
      expect(style.getText()?.getTextBaseline()).toBe("top");
      expect(style.getText()?.getOffsetX()).toBeLessThan(0);
      expect(style.getText()?.getOffsetY()).toBeGreaterThan(0);
      expect(style.getGeometryFunction()(new Feature())?.getExtent()).toEqual([40, 50, 40, 50]);
    }
  });
});
