import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { FeedbackInfo } from "./api/types";
import { FeedbackCard, feedbackPageContext } from "./Feedback";

const item: FeedbackInfo = {
  id: 17, kind: "improvement", title: "Поиск снимков", message: "Добавить поиск по имени", author: "author",
  page_path: "#/dataset-editor/forest", page_title: "Редактор датасетов", app_version: null, credit_name: null,
  status: "preparing", revision: 3, preparation: "Поиск без учёта регистра", progress: "Решение подготовлено",
  approved_at: null, approved_by: null, approval_note: null, news_slug: null, commit_sha: null,
  created_at: "2026-10-01T10:00:00Z", updated_at: "2026-10-01T10:01:00Z",
};

describe("обращения пользователей", () => {
  it("сохраняет исходную страницу и исключает параметры запроса", () => {
    expect(feedbackPageContext("#/dataset-editor/forest?token=secret")).toEqual({ page_path: "#/dataset-editor/forest", page_title: "Редактор датасетов" });
    expect(feedbackPageContext("").page_path).toBe("#/");
  });
  it("показывает готовое решение и ожидание подтверждения", () => {
    const html = renderToStaticMarkup(<FeedbackCard item={item} />);
    expect(html).toContain("Началась подготовка");
    expect(html).toContain("Ожидается подтверждение владельца");
    expect(html).toContain('href="#/feedback/17"');
    expect(html).toContain("Поиск без учёта регистра");
  });
  it("экранирует пользовательский текст и ведёт из результата в новость", () => {
    const html = renderToStaticMarkup(<FeedbackCard item={{ ...item, status: "implemented", message: "<script>danger</script>", news_slug: "search-images" }} />);
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain('href="#/news/search-images"');
    expect(html).not.toContain("Ожидается подтверждение владельца");
  });
});
