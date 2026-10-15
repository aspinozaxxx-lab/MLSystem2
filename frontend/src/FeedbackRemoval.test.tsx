import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { NewsPage, newsArticles } from "./News";
import { usageActions, usagePages } from "./utils/usage";

describe("удаление механизма предложений", () => {
  it("не регистрирует страницу и действие отправки обращений в аналитике", () => {
    expect(usagePages).not.toHaveProperty("feedback");
    expect(usageActions).not.toHaveProperty("feedback_send");
  });

  it("публикует удаление и сохраняет прежние адреса новостей как архив", () => {
    const update = newsArticles.find(item => item.slug === "remove-user-proposals");
    expect(update && "updates" in update && update.updates.length).toBeGreaterThan(0);
    for (const slug of ["grovika-user-proposals", "grovika-feedback-simple-form"]) {
      const html = renderToStaticMarkup(<NewsPage slug={slug} />);
      expect(html).toContain("Архив:");
      expect(html).toContain("удалён");
      expect(html).not.toContain('href="#/feedback');
    }
  });
});
