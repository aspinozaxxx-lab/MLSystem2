import { describe, expect, it } from "vitest";
import type { PseudoMarkupComparisonCounts, PseudoMarkupPixelComparison, PseudoMarkupViewInfo } from "../api/types";
import { comparisonAppearance, comparisonScenes, comparisonTotal, loadComparisonSceneCounts } from "./pseudoComparison";

describe("сравнение псевдоразметок", () => {
  it("считает выбранный снимок первым, отдаёт готовые значения сразу и сохраняет их при переключении", async () => {
    const completed = new Map<string, PseudoMarkupPixelComparison>();
    const calls: string[] = []; const delivered: number[] = [];
    const value = { intersection: 2, only_first: 3, only_second: 4 };
    const load = async (id: string) => {
      calls.push(id);
      return { result_ids: ["1", "2"], scenes: { [id]: value }, total: value, warnings: [] } as PseudoMarkupComparisonCounts;
    };
    await loadComparisonSceneCounts(["первый", "второй", "третий"], "третий", completed, new AbortController().signal, load,
      () => delivered.push(completed.size));
    expect(calls).toEqual(["третий", "первый", "второй"]);
    expect(delivered).toEqual([1, 2, 3]);
    await loadComparisonSceneCounts(["первый", "второй", "третий"], "второй", completed, new AbortController().signal, load, () => {});
    expect(calls).toHaveLength(3);
    expect(comparisonTotal(Object.fromEntries(completed))).toEqual({ intersection: 6, only_first: 9, only_second: 12 });
  });
  it("не продолжает старую очередь и не принимает её ответ после отмены", async () => {
    const controller = new AbortController(); const completed = new Map();
    const calls: string[] = []; let delivered = false;
    const load = async (id: string, signal: AbortSignal) => {
      calls.push(id); expect(signal).toBe(controller.signal); controller.abort();
      return { scenes: { [id]: { intersection: 1, only_first: 0, only_second: 0 } } } as PseudoMarkupComparisonCounts;
    };
    await expect(loadComparisonSceneCounts(["первый", "второй"], "второй", completed, controller.signal, load,
      () => { delivered = true; })).rejects.toMatchObject({ name: "AbortError" });
    expect(calls).toEqual(["второй"]);
    expect(completed.size).toBe(0); expect(delivered).toBe(false);
  });
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
