import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { TestSampleDownloadOptionsForm } from "./TestSampleDownloadOptions";

describe("скачивание тестовой разметки", () => {
  it("по умолчанию выбирает общий GeoJSON без снимков, оставляя оба формата и добавление TIFF", () => {
    const submit = vi.fn();
    const html = renderToStaticMarkup(<TestSampleDownloadOptionsForm onCancel={vi.fn()} onSubmit={submit} />);
    expect(html.match(/type="radio"/g)).toHaveLength(2);
    expect(html).toContain("GeoJSON на каждый снимок");
    expect(html).toContain("Один GeoJSON");
    expect(html).toMatch(/checked=""[^>]*value="single"/);
    expect(html).toMatch(/<input type="checkbox"\s*\/>/);
    expect(html).toContain("Добавить снимки TIFF");
    expect(html).toContain("По умолчанию архив содержит только GeoJSON");
    expect(html).not.toContain("превью");
    expect(submit).not.toHaveBeenCalled();
  });
});
