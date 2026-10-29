import { describe, expect, it } from "vitest";
import Polygon from "ol/geom/Polygon";
import MultiPolygon from "ol/geom/MultiPolygon";
import { getCenter } from "ol/extent";
import { replaceTestRegionLabels, resizeTestRegion, squareTestRegion, TEST_REGION_SIZES, testRegionSize, testRegions, translateTestRegion } from "./testRegions";

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
