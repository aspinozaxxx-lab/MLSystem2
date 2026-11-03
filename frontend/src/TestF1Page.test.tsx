import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { TestF1DownloadOptions } from "./TestF1Page";

describe("выгрузка сохранённого прогноза тестового F1", () => {
  it("предлагает отдельные GeoJSON в ZIP и один общий файл, без TIFF и эталона", () => {
    const submit = vi.fn();
    const html = renderToStaticMarkup(<TestF1DownloadOptions busy={false} onCancel={vi.fn()} onSubmit={submit} />);
    expect(html.match(/type="radio"/g)).toHaveLength(2);
    expect(html).toContain("По снимкам"); expect(html).toContain("Один общий GeoJSON");
    expect(html).toMatch(/checked=""[^>]*value="per_image"/);
    expect(html).toContain("Скачать ZIP");
    expect(html).not.toContain("TIFF"); expect(html).not.toContain("Эталон");
    expect(submit).not.toHaveBeenCalled();
  });
  it("блокирует повторные нажатия, пока сервер формирует выгрузку", () => {
    const html = renderToStaticMarkup(<TestF1DownloadOptions busy onCancel={vi.fn()} onSubmit={vi.fn()} />);
    expect(html).toContain("Формирование…");
    expect(html).toContain('<fieldset class="download-mode-fieldset" disabled="">');
    expect(html).toContain('type="submit" disabled=""');
  });
});
