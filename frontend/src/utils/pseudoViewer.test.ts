import { describe, expect, it } from "vitest";
import { pseudoClass, pseudoClasses } from "./pseudoViewer";

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
});
