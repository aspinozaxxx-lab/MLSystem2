import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ResultChangesTable, ResultClassCard } from "./App";
import type { ResultChangeInfo, ResultClassInfo } from "./api/types";
import { newsArticles } from "./News";

describe("результаты класса на одной странице", () => {
  it("выделяет выбранный датасет, сохраняя основную отметку и действующие ссылки", () => {
    const item: ResultClassInfo = { key: "лес", name: "Границы леса", quality_metric: "pixel", is_custom: false, datasets: [
      { key: "основной", name: "main", quality_metric: "pixel", is_primary: true },
      { key: "второй", name: "Копия для сравнения", quality_metric: "pixel", is_primary: false },
    ] };
    const html = renderToStaticMarkup(<ResultClassCard item={item} selectedDatasetKey="второй" />);
    expect(html).toContain('class="dataset-link primary-dataset"');
    expect(html).toContain('class="dataset-link selected-dataset"');
    expect(html).toContain(`href="#/results/${encodeURIComponent("второй")}" aria-current="true"`);
    expect(html).toContain(`href="#/dataset-editor/${encodeURIComponent("второй")}"`);
    expect(html).toContain("Копия для сравнения");
  });

  it("показывает всю переданную историю, ошибки, активное задание и метки мобильных карточек", () => {
    const changes: ResultChangeInfo[] = Array.from({ length: 35 }, (_, index) => ({
      id: String(index), dataset_key: `датасет-${index % 2}`, dataset_name: `Датасет ${index % 2}`,
      class_key: "лес", model_name: `SegFormer B${index % 3}`, changed_at: "2026-10-13T12:00:00Z",
      source: "manual", status: index === 0 ? "running" : index === 1 ? "error" : "ok",
      type: "training", item_type: index === 0 ? "job" : "training_result", action: "обучение",
    }));
    const html = renderToStaticMarkup(<ResultChangesTable changes={changes} showJobLog={async () => {}} />);
    expect(html.match(/result-change-row/g)).toHaveLength(35);
    expect(html).toContain("ошибка");
    expect(html).toContain("Выполняется");
    expect(html).toContain('data-label="Датасет"');
    expect(html).toContain('data-label="Модель"');
    expect(html).toContain('data-label="Время"');
    expect(html).not.toContain("<th>Класс</th>");
    expect(html).toContain(`href="#/results/${encodeURIComponent("датасет-1")}"`);
  });

  it("публикует описание единой страницы с подходящей обозначенной схемой", () => {
    const article = newsArticles.find(item => item.slug === "results-class-dashboard");
    expect(article?.date).toBe("2026-10-13");
    expect(article && "image" in article && article.image?.src).toBe("/news/results-class-dashboard.svg");
    expect(article && "image" in article && article.image?.alt).toContain("Схема");
    expect(article && "image" in article && article.image?.caption).toContain("не скриншот");
  });
});
