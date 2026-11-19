import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { PersonalNoteEditor } from "./PersonalNotePage";

const props = {
  content: "", dirty: false, busy: "", attachments: [], uploadProgress: "",
  onChange: vi.fn(), onSave: vi.fn(), onAttach: vi.fn(), onRemove: vi.fn(),
};

describe("личная заметка", () => {
  it("принимает пустой текст без названия и показывает состояние рядом с сохранением", () => {
    const html = renderToStaticMarkup(<PersonalNoteEditor {...props} />);
    expect(html).toContain('aria-label="Текст заметки"');
    expect(html).toContain("Все изменения сохранены");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>.*?Сохранить/);
    expect(html).not.toContain("required=");
    expect(html).not.toContain('type="text"');
  });

  it("разрешает сохранить очищенную заметку и явно помечает несохранённые изменения", () => {
    const html = renderToStaticMarkup(<PersonalNoteEditor {...props} dirty />);
    expect(html).toContain("Есть несохранённые изменения");
    expect(html).toMatch(/<button type="button" class="primary">/);
  });

  it("показывает сохранение и защищённые ссылки файлов; текст выводит без HTML", () => {
    const html = renderToStaticMarkup(<PersonalNoteEditor {...props} content="<script>текст</script>" busy="save" attachments={[{
      id: "файл", original_name: "<b>план.txt</b>", size_bytes: 123,
      download_url: "/api/v1/personal-note/attachments/файл/download",
    }]} />);
    expect(html).toContain("Сохранение…");
    expect(html).toContain("&lt;script&gt;текст&lt;/script&gt;");
    expect(html).toContain("&lt;b&gt;план.txt&lt;/b&gt;");
    expect(html).toContain('download=""');
    expect(html).toContain('aria-label="Удалить файл &lt;b&gt;план.txt&lt;/b&gt;"');
    expect(html).toContain("Вложения сохраняются сразу");
  });
});
