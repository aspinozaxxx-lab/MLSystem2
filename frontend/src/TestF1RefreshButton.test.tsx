import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { newsArticles } from "./News";
import { TestF1RefreshButton } from "./TestF1RefreshButton";

describe("ручной пересчёт тестового F1", () => {
  it("не показывает кнопку у актуального значения", () => {
    expect(renderToStaticMarkup(<TestF1RefreshButton status="current" endpoint="/проверка" label="Сеть" />)).toBe("");
  });
  it("показывает маленькую подписанную иконку у устаревшей оценки", () => {
    const html = renderToStaticMarkup(<TestF1RefreshButton status="stale" endpoint="/проверка" label="Сеть" />);
    expect(html).toContain('aria-label="Пересчитать F1: Сеть"');
    expect(html).toContain("test-f1-refresh");
    expect(html).not.toContain("disabled");
  });
  it.each(["queued", "running"])("не допускает повторное нажатие при %s", (status) => {
    const html = renderToStaticMarkup(<TestF1RefreshButton status={status} endpoint="/проверка" label="Сеть" />);
    expect(html).toContain("disabled");
    expect(html).toContain("Расчёт F1 уже в очереди");
  });
  it("публикует большую новость с инструкцией и явно обозначенной схемой", () => {
    const article = newsArticles.find((item) => item.slug === "manual-test-markup-editor");
    expect(article?.date).toBe("2026-10-14");
    if (!article || !("sections" in article)) throw new Error("Большая статья не найдена");
    expect(article.sections.length).toBeGreaterThanOrEqual(5);
    expect(article.image?.caption).toContain("Схема");
    expect(article.image?.alt).toContain("Схема");
    expect(article.sections.some((section) => section.steps?.some((step) => step.includes("Создать и открыть редактор")))).toBe(true);
    expect(article.image?.src).toBe("/news/manual-test-markup-editor.svg");
  });
});
