import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ImageryType, TestSampleBatchOptionsResponse, TestSampleCreationSettings } from "./api/types";
import { TEST_SAMPLE_TILE_SIZES, TestMarkupCreatePage } from "./TestMarkupCreatePage";

const state = vi.hoisted(() => ({
  index: 0,
  options: null as TestSampleBatchOptionsResponse | null,
  drafts: {} as Record<string, TestSampleCreationSettings>,
}));

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  return {
    ...actual,
    useState(initialValue: unknown) {
      const index = state.index++;
      return actual.useState(index === 0 ? state.options : index === 1 ? "класс\\main" : index === 2 ? state.drafts : initialValue);
    },
  };
});
vi.mock("./useTestMarkupClasses", () => ({
  useTestMarkupClasses: () => ({
    index: { classes: [{ key: "класс", name: "Класс", sample_count: 0, has_primary: false }] },
    classKey: "класс", setClassKey: vi.fn(), loadClasses: vi.fn(),
  }),
}));

function render(imageryType: ImageryType | null = "kanopus", changes: Partial<TestSampleCreationSettings> = {}) {
  state.index = 0;
  state.drafts = { "класс\\main": {
    tile_size: 8192, min_image_count: 5, image_count: 10, min_object_count: 150,
    min_object_area_m2: 0, exclude_boundary_objects: false, use_optimization: true, ...changes,
  } };
  state.options = { classes: [{ class_key: "класс", class_name: "Класс", datasets: [{
    dataset_key: "класс\\main", dataset_name: "Основной", class_key: "класс", class_name: "Класс",
    image_count: 12, imagery_type: imageryType, pseudo_status: "unavailable",
    quality_metric: "pixel", task: "binary", training_is_primary: false,
  }] }] };
  const requests = vi.fn();
  const run = async <T,>(operation: () => Promise<T>): Promise<T | undefined> => { requests(); return operation(); };
  const html = renderToStaticMarkup(<TestMarkupCreatePage run={run} />);
  return { html, requests };
}

describe("размеры и объём в мастере тестовых разметок", () => {
  it("сохраняет прежние размеры и добавляет все размеры до 8192 с шагом 512", () => {
    expect(TEST_SAMPLE_TILE_SIZES).toEqual([512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]);
    const { html } = render();
    for (const size of TEST_SAMPLE_TILE_SIZES) expect(html).toContain(`${size} × ${size}</option>`);
    expect(html).toContain('<option value="8192" selected="">');
  });

  it("показывает обе границы объёма до кнопки создания, без дополнительных запросов", () => {
    const { html, requests } = render();
    expect(html).toContain("≈ 960 МБ (5 шт.)");
    expect(html).toContain("— 1,9 ГБ (10 шт.)");
    expect(html).toContain("Канопус: ориентир с учётом сжатия и пирамид");
    expect(html).toContain("Без превью, масок и запасных тайлов");
    expect(html).toContain('aria-live="polite"');
    expect(html.indexOf("Примерный объём снимков (TIFF)")).toBeLessThan(html.indexOf("Создать разметку"));
    expect(requests).not.toHaveBeenCalled();
  });

  it("пересчитывает диапазон по типу, размеру и количеству снимков", () => {
    expect(render("ortho").html).toContain("≈ 1,3 ГБ (5 шт.)");
    expect(render("ortho").html).toContain("— 2,5 ГБ (10 шт.)");
    const { html } = render("ortho", { tile_size: 4096, min_image_count: 2, image_count: 3 });
    expect(html).toContain("≈ 128 МБ (2 шт.)");
    expect(html).toContain("— 192 МБ (3 шт.)");
    expect(html).toContain("Ортофото: ориентир");
  });

  it("не показывает ложный диапазон при ошибке ввода или неизвестном типе", () => {
    const { html } = render("kanopus", { min_image_count: 11 });
    expect(html).toContain("Укажите корректный размер и диапазон количества снимков для оценки");
    expect(html).not.toContain("creation-size-range");
    expect(html.match(/<button[^>]+type="submit"[^>]*>/)?.[0]).toContain("disabled");
    expect(render(null).html).toContain("Тип снимков не указан — оценка объёма недоступна");
  });
});
