import { describe, expect, it } from "vitest";

import type {
  DatasetInfo,
  TestSampleCatalogResponse,
  TestSampleDetail,
  TestSampleDraftPreview,
} from "../api/types";
import {
  applyTestMarkupPreview,
  containedImageOneToOneScale,
  estimateTestMarkupImageVolume,
  formatTestMarkupImageVolume,
  isDatasetReadyForTestMarkup,
  sortTestMarkupDatasets,
  testMarkupDraft,
  testMarkupDraftChanged,
  testMarkupStats,
} from "./testMarkups";

function catalog(): TestSampleCatalogResponse {
  return {
    classes: [
      {
        key: "class",
        name: "Класс",
        samples: [
          { id: "one", class_key: "class", dataset_key: "ready", is_primary: true } as never,
          { id: "two", class_key: "class", dataset_key: "strict", is_primary: false } as never,
        ],
      },
    ],
  };
}

describe("тестовые разметки", () => {
  it("оценивает диапазон TIFF по типу снимков и обеим границам количества", () => {
    expect(estimateTestMarkupImageVolume("kanopus", 1024, 5, 10)).toEqual({
      minBytes: 15 * 1024 ** 2, maxBytes: 30 * 1024 ** 2,
    });
    expect(estimateTestMarkupImageVolume("ortho", 1024, 5, 10)).toEqual({
      minBytes: 20 * 1024 ** 2, maxBytes: 40 * 1024 ** 2,
    });
  });

  it("учитывает площадь тайла, а не только сторону, и не включает запасные тайлы", () => {
    const small = estimateTestMarkupImageVolume("kanopus", 4096, 2, 2)!;
    const large = estimateTestMarkupImageVolume("kanopus", 8192, 2, 2)!;
    expect(small.minBytes).toBe(small.maxBytes);
    expect(large.minBytes).toBe(small.minBytes * 4);
    expect(large.maxBytes).toBe(384 * 1024 ** 2);
    expect(estimateTestMarkupImageVolume("ortho", 8192, 5, 10)).toEqual({
      minBytes: 1280 * 1024 ** 2, maxBytes: 2560 * 1024 ** 2,
    });
  });

  it.each([
    [0, 1, 2], [NaN, 1, 2], [1024.5, 1, 2], [1024, 0, 2],
    [1024, -1, 2], [1024, 2, 1], [1024, 1, Infinity], [1024, 1, 2.5],
    [8192, 1, Number.MAX_SAFE_INTEGER],
  ])("не показывает оценку при некорректных параметрах %s, %s, %s", (size, minCount, maxCount) => {
    expect(estimateTestMarkupImageVolume("kanopus", size, minCount, maxCount)).toBeNull();
  });

  it("не подменяет неизвестный тип снимков Канопусом", () => {
    expect(estimateTestMarkupImageVolume(null, 1024, 5, 10)).toBeNull();
    expect(estimateTestMarkupImageVolume(undefined, 1024, 5, 10)).toBeNull();
    expect(estimateTestMarkupImageVolume("неизвестный" as never, 1024, 5, 10)).toBeNull();
  });

  it("показывает объём в МБ и ГБ с русским десятичным разделителем", () => {
    expect(formatTestMarkupImageVolume(768 * 1024)).toBe("0,8 МБ");
    expect(formatTestMarkupImageVolume(30 * 1024 ** 2)).toBe("30 МБ");
    expect(formatTestMarkupImageVolume(1024 ** 3)).toBe("1 ГБ");
    expect(formatTestMarkupImageVolume(2560 * 1024 ** 2)).toBe("2,5 ГБ");
  });

  it("показывает готовые датасеты старого и поснимочного формата при создании", () => {
    const dataset = (update: Partial<DatasetInfo>): DatasetInfo => ({
      key: "dataset",
      name: "Класс\\датасет",
      is_custom: false,
      quality_metric: "pixel",
      source_available: true,
      is_primary: false,
      task: "binary",
      combined: false,
      managed: false,
      source_status: "current",
      hard_negative_count: 0,
      materialization_status: "not_applicable",
      ...update,
    });
    const legacy = dataset({
      key: "legacy",
      scenes_file: "/data/scenes.txt",
      annotation_file: "/data/markup.geojson",
      diagnostics: [],
    });
    const perImage = dataset({
      key: "per-image",
      format: "per_image",
      annotations_dir: "/data/markup",
      image_count: 12,
      diagnostics: [],
    });
    const empty = {
      ...perImage,
      key: "empty",
      image_count: 0,
    };
    const invalid = {
      ...perImage,
      key: "invalid",
      diagnostics: ["Для GeoJSON не найден TIFF"],
    };

    expect(isDatasetReadyForTestMarkup(legacy)).toBe(true);
    expect(isDatasetReadyForTestMarkup(perImage)).toBe(true);
    expect(isDatasetReadyForTestMarkup(empty)).toBe(false);
    expect(isDatasetReadyForTestMarkup(invalid)).toBe(false);
  });

  it("считает разметки и наличие основной", () => {
    expect(testMarkupStats(catalog(), "class")).toEqual({ count: 2, hasPrimary: true });
    expect(testMarkupStats(catalog(), "missing")).toEqual({ count: 0, hasPrimary: false });
  });

  it("показывает датасеты без основной разметки первыми", () => {
    const ready = { key: "ready", class_key: "class", class_name: "Класс", dataset_name: "Готовый" } as DatasetInfo;
    const missing = { key: "missing", class_key: "missing", class_name: "Новый класс", dataset_name: "Новый" } as DatasetInfo;
    expect(sortTestMarkupDatasets([ready, missing], catalog()).map((item) => item.key)).toEqual([
      "missing",
      "ready",
    ]);
  });

  it("отличает черновик от сохранённого состояния и применяет preview", () => {
    const sample = {
      name: "Разметка",
      is_primary: false,
      tiles: [
        { index: 1, enabled: true },
        { index: 2, enabled: false },
      ],
    } as TestSampleDetail;
    const draft = testMarkupDraft(sample);
    expect(testMarkupDraftChanged(sample, draft)).toBe(false);

    const preview = { enabled_tile_indices: [2] } as TestSampleDraftPreview;
    const updated = applyTestMarkupPreview(draft, preview);
    expect(updated.enabledTileIndices).toEqual([2]);
    expect(testMarkupDraftChanged(sample, updated)).toBe(true);
  });

  it("считает масштаб 1:1 от вписывания по обеим сторонам области просмотра", () => {
    expect(containedImageOneToOneScale(1400, 420, 3584, 3584, 12)).toBeCloseTo(3584 / 420);
    expect(containedImageOneToOneScale(1000, 800, 2000, 1000, 12)).toBe(2);
    expect(containedImageOneToOneScale(1000, 500, 1000, 2000, 12)).toBe(4);
    expect(containedImageOneToOneScale(1000, 800, 400, 300, 12)).toBe(1);
    expect(containedImageOneToOneScale(0, 800, 3584, 3584, 12)).toBe(1);
  });
});
