import Feature from "ol/Feature";
import Polygon from "ol/geom/Polygon";
import { describe, expect, it } from "vitest";

import { clonePseudoFeatureForDraft, pseudoMarkupStyle } from "./DatasetEditorPage";


describe("стиль псевдоразметки редактора датасета", () => {
  it("использует отдельный цвет каждого класса сети", () => {
    const geometry = new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]);
    const first = new Feature({
      geometry,
      object_type_slug: "water",
      object_type_color: "#3366CC",
    });
    const second = new Feature({
      geometry,
      object_type_id: 2,
    });
    const objectTypes = [
      { id: 1, slug: "water", name: "Вода", color: "#3366CC", priority: 0 },
      { id: 2, slug: "swamp", name: "Заболачивание", color: "#22AA55", priority: 1 },
    ];

    expect(pseudoMarkupStyle(first, objectTypes).getStroke()?.getColor()).toBe("#3366CC");
    expect(pseudoMarkupStyle(second, objectTypes).getStroke()?.getColor()).toBe("#22AA55");
  });

  it("копирует выбранный объект сети в указанный класс без служебных полей прогноза", () => {
    const geometry = new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]);
    const pseudo = new Feature({
      geometry,
      object_type_slug: "water",
      confidence: 0.91,
    });
    const dataset = {
      task: "multiclass" as const,
      object_types: [
        { id: 1, slug: "water", name: "Вода", color: "#3366CC", priority: 0 },
        { id: 2, slug: "swamp", name: "Заболачивание", color: "#22AA55", priority: 1 },
      ],
    };

    const accepted = clonePseudoFeatureForDraft(pseudo, "swamp", dataset);

    expect(accepted).not.toBeNull();
    expect(accepted?.getGeometry()).not.toBe(geometry);
    expect(accepted?.get("_mlsystem2_role")).toBe("positive");
    expect(accepted?.get("_mlsystem2_class")).toBe("swamp");
    expect(accepted?.get("object_type_slug")).toBeUndefined();
    expect(accepted?.get("confidence")).toBeUndefined();
    expect(accepted?.getId()).toBeTruthy();
  });

  it("добавляет выбранный объект сети как hard negative", () => {
    const pseudo = new Feature({
      geometry: new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]),
      object_type_slug: "water",
    });
    const dataset = {
      task: "multiclass" as const,
      object_types: [
        { id: 1, slug: "water", name: "Вода", color: "#3366CC", priority: 0 },
      ],
    };

    const accepted = clonePseudoFeatureForDraft(pseudo, "hard_negative", dataset);

    expect(accepted?.get("_mlsystem2_role")).toBe("hard_negative");
    expect(accepted?.get("_mlsystem2_class")).toBeUndefined();
  });

  it("добавляет объект binary-сети как обычную положительную разметку", () => {
    const pseudo = new Feature({
      geometry: new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]),
    });
    const accepted = clonePseudoFeatureForDraft(pseudo, "positive", {
      task: "binary",
      object_types: [],
    });

    expect(accepted?.get("_mlsystem2_role")).toBe("positive");
    expect(accepted?.get("_mlsystem2_class")).toBeUndefined();
  });
});
