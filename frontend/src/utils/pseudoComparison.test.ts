import { describe, expect, it } from "vitest";
import type { PseudoMarkupViewInfo } from "../api/types";
import { comparisonAppearance, comparisonScenes } from "./pseudoComparison";

describe("сравнение псевдоразметок", () => {
  it("объединяет один TIFF из разных датасетов, сохраняя разные снимки с одинаковыми именами", () => {
    const view = (id: string, scenes: { id: string; name: string }[]) => ({ id, scenes } as PseudoMarkupViewInfo);
    const scenes = comparisonScenes([view("реки", [{ id: "общий", name: "снимок.tif" }, { id: "копия", name: "снимок.tif" }]),
      view("здания", [{ id: "общий", name: "снимок.tif" }])]);
    expect(scenes).toHaveLength(2);
    expect(scenes.find((scene) => scene.id === "общий")?.resultIds).toEqual(["реки", "здания"]);
    expect(scenes.find((scene) => scene.id === "копия")?.resultIds).toEqual(["реки"]);
  });
  it("показывает независимые яркие пересечения и различия, а один оставшийся слой — собственным цветом", () => {
    const colors = { first: "#ff1744", second: "#ffea00" };
    const style = (kind: string, count: number, overlap = true, difference = true) =>
      comparisonAppearance({ comparison_kind: kind, comparison_result_id: "first" }, ["first", "second"], colors, count, overlap, difference);
    expect(style("intersection", 2)?.fill).toBe("#00ff66cc");
    expect(style("intersection", 2, false)).toBeNull();
    expect(style("difference", 2)?.fill).toBe("#ff1744cc");
    expect(style("difference", 2, true, false)).toBeNull();
    expect(style("layer", 2)?.fill).toBe("transparent");
    expect(style("layer", 1)?.fill).toBe("#ff1744b3");
    expect(style("layer", 2, false, false)?.fill).toBe("#ff1744b3");
    expect(comparisonAppearance({ comparison_kind: "layer", comparison_result_id: "hidden" }, ["first"], colors, 1, true, true)).toBeNull();
  });
});
