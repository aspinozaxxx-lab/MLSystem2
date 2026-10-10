import { describe, expect, it } from "vitest";
import { comparisonLayerLabel, comparisonLayerStyle, sceneF1Score } from "./testF1Viewer";
import type { TestF1SceneInfo, TestF1ScoreInfo } from "../api/types";

describe("Слои сравнения тестовой разметки", () => {
  it("после отключения эталона оставляет только прогноз, после отключения прогноза — только эталон", () => {
    for (const name of ["tp", "fp", "fn"]) {
      expect(comparisonLayerStyle(name, false, true)).toBeNull();
      expect(comparisonLayerStyle(name, true, false)).toBeNull();
      expect(comparisonLayerStyle(name, true, true)?.fill).not.toBe("transparent");
    }
    expect(comparisonLayerStyle("reference", false, true)).toBeNull();
    expect(comparisonLayerStyle("predicted", false, true)?.fill).not.toBe("transparent");
    expect(comparisonLayerStyle("predicted", true, false)).toBeNull();
    expect(comparisonLayerStyle("reference", true, false)?.fill).not.toBe("transparent");
    for (const name of ["tp", "fp", "fn", "reference", "predicted"]) expect(comparisonLayerStyle(name, false, false)).toBeNull();
  });
  it("сохраняет разные цвета совпадения, лишнего и пропуска", () => {
    expect(new Set(["tp", "fp", "fn"].map((name) => comparisonLayerStyle(name, true, true)?.color)).size).toBe(3);
  });
  it("объектовый режим подписывает найденные и пропущенные объекты", () => {
    expect(comparisonLayerLabel("tp", "objects")).toBe("TP · Найденные объекты");
    expect(comparisonLayerLabel("fn", "objects")).toBe("FN · Пропущенные объекты");
    expect(comparisonLayerLabel("tp", "pixel")).toBe("TP · Совпадение");
    expect(comparisonLayerLabel("reference", "objects")).toBe("Эталон");
  });
});

it("выбор типа показывает его F1, а managed-тайл сохраняет собственную оценку", () => {
  const score = (f1: number): TestF1ScoreInfo => ({ f1, precision: f1, recall: f1, true_positive: 1, false_positive: 1, false_negative: 1 });
  const scene: TestF1SceneInfo = { id: "scene", name: "Тайл", raster_url: "/raster", footprint_url: "/footprint", layers_url: "/layers",
    bounds: [0, 0, 1, 1], has_alpha: false, has_nir: false, raster_available: true, object_layers_available: true, sample_name: "Выборка", sample_revision: 1,
    pixel: score(0.9), objects: score(0.8), class_schema: [{ id: 2, slug: "second" }],
    metrics: { pixel: { per_class: { second: score(0.4) } }, objects: { per_class: { second: score(0.3) } } } };
  expect(sceneF1Score(scene, "pixel", 2).f1).toBe(0.4);
  expect(sceneF1Score(scene, "objects", 2).f1).toBe(0.3);
  expect(sceneF1Score(scene, "pixel", null).f1).toBe(0.9);
  expect(sceneF1Score({ ...scene, target_class_id: 2 }, "pixel", 2).f1).toBe(0.9);
});
