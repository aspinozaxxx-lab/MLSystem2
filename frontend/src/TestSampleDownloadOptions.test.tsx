import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { TestSampleDownloadOptionsForm } from "./TestSampleDownloadOptions";

describe("скачивание тестовой разметки", () => {
  it("предлагает два состава и выбирает общий GeoJSON по умолчанию", () => {
    const submit = vi.fn();
    const html = renderToStaticMarkup(<TestSampleDownloadOptionsForm onCancel={vi.fn()} onSubmit={submit} />);
    expect(html.match(/type="radio"/g)).toHaveLength(2);
    expect(html).toContain("GeoJSON на каждый снимок + TIFF");
    expect(html).toContain("Один GeoJSON + TIFF");
    expect(html).toMatch(/checked=""[^>]*value="single"/);
    expect(html).not.toContain("превью");
    expect(submit).not.toHaveBeenCalled();
  });
});
