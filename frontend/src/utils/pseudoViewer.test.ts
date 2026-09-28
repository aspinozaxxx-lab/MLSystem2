import { describe, expect, it } from "vitest";
import { pseudoClass, pseudoClasses, pseudoRasterStyle } from "./pseudoViewer";

describe("слои просмотра псевдоразметки", () => {
  it("сохраняет классы, цвета и отдельные объекты в общей мозаике", () => {
    const groups = pseudoClasses([
      { properties: { object_type_slug: "river", object_type_name: "Реки", object_type_color: "#22aaff" } },
      { properties: { object_type_slug: "river", object_type_name: "Реки", object_type_color: "#22aaff" } },
      { properties: { object_type_slug: "lake", object_type_name: "Озёра", object_type_color: "#5566FF" } },
    ]);
    expect(groups.map((item) => [item.name, item.count])).toEqual([["Озёра", 1], ["Реки", 2]]);
    expect(groups[1].color).toBe("#22AAFF");
  });
  it("поддерживает старую бинарную и пустую псевдоразметку", () => {
    expect(pseudoClasses([])).toEqual([]);
    expect(pseudoClasses([{ properties: null }, {}])[0].count).toBe(2);
    expect(pseudoClass({ object_type_color: "не цвет" })).toEqual({ key: "objects", name: "Объекты", color: "#22D3EE" });
  });
  it("подставляет NIR в NRG и NGB, сохраняя отдельную маску прозрачности", () => {
    expect(pseudoRasterStyle("NRG", false, true).color).toEqual([
      "color", ["*", ["band", 4], 255], ["*", ["band", 1], 255], ["*", ["band", 2], 255], ["band", 5],
    ]);
    expect(pseudoRasterStyle("NGB", false, true).color).toEqual([
      "color", ["*", ["band", 4], 255], ["*", ["band", 2], 255], ["*", ["band", 3], 255], ["band", 5],
    ]);
  });
  it("оставляет RGB и RGBA в естественных цветах при выборе NIR-сочетаний", () => {
    for (const alpha of [false, true]) {
      expect(pseudoRasterStyle("NRG", alpha, false)).toEqual(pseudoRasterStyle("RGB", alpha, false));
      expect(pseudoRasterStyle("NGB", alpha, false)).toEqual(pseudoRasterStyle("RGB", alpha, false));
    }
    expect((pseudoRasterStyle("NRG", true, false).color as unknown[]).at(-1)).toEqual(["*", ["band", 4], ["band", 5]]);
  });
});
