import { describe, expect, it } from "vitest";
import WebGLTileLayer from "ol/layer/WebGLTile";
import DataTile from "ol/source/DataTile";
import { pseudoClass, pseudoClasses, pseudoRasterCacheSizes, pseudoRasterScenes, pseudoRasterStyle, pseudoViewportScenes } from "./pseudoViewer";

describe("слои просмотра псевдоразметки", () => {
  it("отмечает частично видимые и перекрывающиеся снимки при перемещении и приближении карты", () => {
    const scenes = [
      { id: "слева", bounds: [0, 0, 10, 10] },
      { id: "перекрытие", bounds: [8, 0, 18, 10] },
      { id: "справа", bounds: [30, 0, 40, 10] },
    ];
    const visible = (center: number[], resolution: number, size = [20, 20]) =>
      pseudoViewportScenes(scenes, { center, resolution, rotation: 0 }, size).map((scene) => scene.id);
    expect(visible([20, 5], 3)).toEqual(["слева", "перекрытие", "справа"]);
    expect(visible([9, 5], 0.1)).toEqual(["слева", "перекрытие"]);
    expect(visible([35, 5], 0.1)).toEqual(["справа"]);
    expect(visible([25, 5], 0.1)).toEqual([]);
    expect(visible([25, 5], 0.1, [160, 20])).toEqual(["перекрытие", "справа"]);
  });
  it("не отмечает снимок за углом повёрнутой карты, даже если он входит в её bounding box", () => {
    const scenes = [
      { id: "в центре", bounds: [-1, -1, 1, 1] },
      { id: "за углом", bounds: [11, 11, 13, 13] },
      { id: "частично", bounds: [12, -1, 16, 1] },
    ];
    expect(pseudoViewportScenes(scenes, { center: [0, 0], resolution: 1, rotation: Math.PI / 4 }, [20, 20])
      .map((scene) => scene.id)).toEqual(["в центре", "частично"]);
    expect(pseudoViewportScenes([], { center: [0, 0], resolution: 1, rotation: 0 }, [20, 20])).toEqual([]);
  });
  it.each([[4, 25, 0], [4, 2, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]])(
    "сохраняет ограниченный кэш и не роняет OpenLayers при смене снимков (%i/%i/%i)", (...counts) => {
      const sizes = pseudoRasterCacheSizes(counts);
      expect(sizes.reduce((sum, size) => sum + size, 0)).toBeLessThanOrEqual(256);
      for (const cacheSize of sizes) {
        const layer = new WebGLTileLayer({ cacheSize });
        const renderer = layer.getRenderer()!;
        try {
          // Реальный путь OpenLayers, на котором падала мозаика ОКС500 после накопления ключей.
          expect(() => {
            for (let index = 0; index < 300; index += 1) renderer.prependStaleKey(`снимок-${index % 29}`);
          }).not.toThrow();
          expect(renderer.getStaleKeys().length).toBeLessThanOrEqual(cacheSize / 2);
        } finally { layer.dispose(); }
      }
    },
  );
  it("читает схему каналов по одному снимку, а для карты выбирает только видимые", () => {
    const scenes = [
      { id: "первый", bounds: [0, 0, 10, 10], source: new DataTile({ bandCount: 4 }) },
      { id: "второй", bounds: [20, 0, 30, 10], source: new DataTile({ bandCount: 4 }) },
    ];
    const selected: string[][] = [];
    const layer = new WebGLTileLayer({ sources: (extent, resolution) => {
      const visible = pseudoRasterScenes(scenes, extent, resolution);
      selected.push(visible.map((scene) => scene.id));
      return visible.map((scene) => scene.source);
    } });
    try {
      layer.getRenderer();
      expect(selected.length).toBeGreaterThan(0);
      expect(selected.every((ids) => ids.length === 1 && ids[0] === "первый")).toBe(true);
      expect(layer.getSources([21, 1, 29, 9], 1)).toEqual([scenes[1].source]);
      expect(layer.getSources([40, 0, 50, 10], 1)).toEqual([]);
    } finally {
      layer.dispose();
      scenes.forEach((scene) => scene.source.dispose());
    }
  });
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
