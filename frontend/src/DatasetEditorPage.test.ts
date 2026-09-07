import Feature from "ol/Feature";
import Polygon from "ol/geom/Polygon";
import { Stroke, Style } from "ol/style";
import { describe, expect, it } from "vitest";

import {
  clonePseudoFeatureForDraft,
  displayedClassStyles,
  isEditorFeatureVisible,
  pseudoMarkupStyle,
} from "./DatasetEditorPage";


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

  it("привязывает hard negative управляемого датасета к выбранному типу", () => {
    const pseudo = new Feature({
      geometry: new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]),
    });
    const accepted = clonePseudoFeatureForDraft(pseudo, "hard_negative:water", {
      task: "multiclass",
      managed: true,
      object_types: [
        { id: 1, slug: "water", name: "Вода", color: "#3366CC", priority: 0 },
        { id: 2, slug: "swamp", name: "Заболачивание", color: "#22AA55", priority: 1 },
      ],
    });

    expect(accepted?.get("_mlsystem2_role")).toBe("hard_negative");
    expect(accepted?.get("_mlsystem2_class")).toBe("water");
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

describe("видимость и подсветка классов редактора", () => {
  const objectTypes = [
    { id: 1, slug: "water", name: "Вода", color: "#3366CC", priority: 0 },
    { id: 2, slug: "swamp", name: "Заболачивание", color: "#22AA55", priority: 1 },
  ];

  it("скрывает один класс в разметке и предсказаниях по slug или ID, сохраняя остальные", () => {
    const display = { hiddenClasses: new Set(["water"]), highlightedClass: "water" };
    const markup = new Feature({ _mlsystem2_class: "water" });
    const predictionBySlug = new Feature({ object_type_slug: "water" });
    const predictionById = new Feature({ object_type_id: "1" });
    const otherMarkup = new Feature({ _mlsystem2_class: "swamp" });
    const otherPrediction = new Feature({ object_type_id: 2 });
    const base = new Style({ stroke: new Stroke({ color: "#3366CC", width: 3 }) });

    expect(isEditorFeatureVisible(markup, objectTypes, display)).toBe(false);
    expect(displayedClassStyles(markup, objectTypes, display, base)).toBeUndefined();
    for (const prediction of [predictionBySlug, predictionById]) {
      expect(isEditorFeatureVisible(prediction, objectTypes, display, true)).toBe(false);
      expect(displayedClassStyles(prediction, objectTypes, display, base, true)).toBeUndefined();
    }
    expect(displayedClassStyles(otherMarkup, objectTypes, display, base)).toBe(base);
    expect(displayedClassStyles(otherPrediction, objectTypes, display, base, true)).toBe(base);
  });

  it("отдельно переключает общую плашку отрицательных примеров", () => {
    const negative = new Feature({ _mlsystem2_role: "hard_negative", _mlsystem2_class: "water" });
    const positive = new Feature({ _mlsystem2_role: "positive", _mlsystem2_class: "water" });
    const hideWater = { hiddenClasses: new Set(["water"]), highlightedClass: null };
    const hideNegatives = { hiddenClasses: new Set(["hard_negative"]), highlightedClass: null };

    expect(isEditorFeatureVisible(negative, objectTypes, hideWater)).toBe(true);
    expect(isEditorFeatureVisible(negative, objectTypes, hideNegatives)).toBe(false);
    expect(isEditorFeatureVisible(positive, objectTypes, hideNegatives)).toBe(true);
  });

  it.each([false, true])("подсвечивает класс белой обводкой, сохраняя цвет и пунктир, псевдоразметка: %s", (pseudo) => {
    const feature = new Feature({ _mlsystem2_class: "water", object_type_id: 1 });
    const base = pseudoMarkupStyle(feature, objectTypes);
    const display = { hiddenClasses: new Set<string>(), highlightedClass: "water" };
    const styles = displayedClassStyles(feature, objectTypes, display, base, pseudo) as Style[];

    expect(styles).toHaveLength(2);
    expect(styles[0].getStroke()?.getColor()).toBe("#FFFFFF");
    expect(styles[1].getStroke()?.getColor()).toBe("#3366CC");
    expect(styles[1].getStroke()?.getWidth()).toBeGreaterThan(base.getStroke()!.getWidth()!);
    expect(styles[1].getStroke()?.getLineDash()).toEqual(base.getStroke()?.getLineDash());
    expect(styles[1].getZIndex()).toBeGreaterThan(styles[0].getZIndex()!);
    expect(displayedClassStyles(feature, objectTypes, { ...display, highlightedClass: null }, base, pseudo)).toBe(base);
  });

  it("не меняет геометрию, свойства и исходный стиль и сохраняет маркеры выбранных вершин", () => {
    const geometry = new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]);
    const feature = new Feature({ geometry, _mlsystem2_class: "water" });
    const properties = feature.getProperties();
    const coordinates = geometry.getCoordinates();
    const revision = feature.getRevision();
    const base = new Style({ stroke: new Stroke({ color: "#3366CC", width: 3 }) });
    const vertexStyle = new Style({ zIndex: 20 });
    const display = { hiddenClasses: new Set<string>(), highlightedClass: "water" };

    const styles = displayedClassStyles(feature, objectTypes, display, [base, vertexStyle]) as Style[];

    expect(styles.at(-1)).toBe(vertexStyle);
    expect(styles[1].getFill()).toBeNull();
    expect(base.getStroke()?.getWidth()).toBe(3);
    expect(base.getZIndex()).toBeUndefined();
    expect(feature.getProperties()).toEqual(properties);
    expect(geometry.getCoordinates()).toEqual(coordinates);
    expect(feature.getRevision()).toBe(revision);
  });

  it("оставляет бинарную разметку и предсказания с неизвестным типом видимыми", () => {
    const feature = new Feature({ _mlsystem2_role: "positive" });
    const display = { hiddenClasses: new Set<string>(), highlightedClass: null };
    const base = new Style();

    expect(displayedClassStyles(feature, [], display, base)).toBe(base);
    expect(displayedClassStyles(feature, [], display, base, true)).toBe(base);
  });
});
