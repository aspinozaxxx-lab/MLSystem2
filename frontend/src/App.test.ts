import { describe, expect, it } from "vitest";

import { perClassF1Values } from "./App";

describe("поклассовые метрики результатов", () => {
  it("сортирует классы по русскому алфавиту независимо от порядка JSON", () => {
    const values = perClassF1Values({
      pixel: {
        per_class: {
          wind: { slug: "wind", name: "Ветровая эрозия", f1: 0.6 },
          salinity: { slug: "salinity", name: "Засоление", f1: 0.7 },
          desert: { slug: "desert", name: "Опустынивание", f1: 0.8 },
          abrasion: { slug: "abrasion", name: "Абразия", f1: 0.9 },
        },
      },
    }, "pixel");

    expect(values.map((item) => item.name)).toEqual([
      "Абразия",
      "Ветровая эрозия",
      "Засоление",
      "Опустынивание",
    ]);
  });

  it("сортирует сохранённые метрики тестовой разметки тем же способом", () => {
    const metrics = {
      objects: {
        per_class: {
          floodings: { slug: "floodings", name: "Переувлажнения", f1: 0.62 },
          swampings: { slug: "swampings", name: "Заболачивание", f1: 0.71 },
        },
      },
    };

    expect(perClassF1Values(metrics, "objects").map((item) => item.name)).toEqual([
      "Заболачивание",
      "Переувлажнения",
    ]);
  });
});
