import { describe, expect, it } from "vitest";
import { selectAvailableTestMarkupClass } from "./useTestMarkupClasses";

describe("сохранённый класс тестовых разметок", () => {
  it("сохраняет выбор после изменения порядка и обновления счётчиков", () => {
    expect(selectAvailableTestMarkupClass(["озёра", "реки"], "реки")).toBe("реки");
    expect(selectAvailableTestMarkupClass(["реки", "озёра"], "реки")).toBe("реки");
  });
  it("выбирает доступный класс после удаления прежнего и допускает пустой каталог", () => {
    expect(selectAvailableTestMarkupClass(["озёра"], "реки")).toBe("озёра");
    expect(selectAvailableTestMarkupClass([], "реки")).toBe("");
  });
});
