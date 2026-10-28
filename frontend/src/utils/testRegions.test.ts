import { describe, expect, it } from "vitest";
import { replaceTestRegionLabels, squareTestRegion, TEST_REGION_SIZES, testRegions } from "./testRegions";

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
});
