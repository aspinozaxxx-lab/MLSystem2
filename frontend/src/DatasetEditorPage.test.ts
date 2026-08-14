import Feature from "ol/Feature";
import Polygon from "ol/geom/Polygon";
import { describe, expect, it } from "vitest";

import { pseudoMarkupStyle } from "./DatasetEditorPage";


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
});
