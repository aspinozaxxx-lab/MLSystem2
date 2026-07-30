import { describe, expect, it } from "vitest";

import type {
  DatasetInfo,
  TestSampleCatalogResponse,
  TestSampleDetail,
  TestSampleDraftPreview,
  TestSampleSummary,
} from "../api/types";
import {
  applyTestMarkupPreview,
  changeTestMarkupDownloadSelection,
  initialTestMarkupDownloadSelection,
  sortTestMarkupDatasets,
  testMarkupDownloadOptions,
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
        datasets: [
          {
            key: "ready",
            name: "Готовый",
            samples: [
              { id: "one", dataset_key: "ready", is_primary: true } as never,
              { id: "two", dataset_key: "ready", is_primary: false } as never,
            ],
          },
        ],
      },
    ],
  };
}

function downloadCatalog(): TestSampleCatalogResponse {
  const sample = (
    id: string,
    datasetKey: string,
    name: string,
    createdAt: string,
    isPrimary: boolean,
    enabledImageCount: number,
  ) => ({
    id,
    name,
    dataset_key: datasetKey,
    dataset_name: `Вырубки\\${datasetKey}`,
    class_name: "Вырубки",
    is_primary: isPrimary,
    enabled_image_count: enabledImageCount,
    created_at: createdAt,
  }) as TestSampleSummary;

  return {
    classes: [
      {
        key: "deforestation",
        name: "Вырубки",
        datasets: [
          {
            key: "main",
            name: "main",
            samples: [
              sample("new", "main", "Новая", "2026-07-30T10:00:00Z", false, 2),
              sample("primary", "main", "Основная", "2026-07-29T10:00:00Z", true, 2),
              sample("old", "main", "Старая", "2026-07-28T10:00:00Z", false, 2),
            ],
          },
          {
            key: "empty",
            name: "empty",
            samples: [
              sample("empty-primary", "empty", "Пустая", "2026-07-30T10:00:00Z", true, 0),
            ],
          },
        ],
      },
    ],
  };
}

describe("тестовые разметки", () => {
  it("считает разметки и наличие основной", () => {
    expect(testMarkupStats(catalog(), "ready")).toEqual({ count: 2, hasPrimary: true });
    expect(testMarkupStats(catalog(), "missing")).toEqual({ count: 0, hasPrimary: false });
  });

  it("показывает датасеты без основной разметки первыми", () => {
    const ready = { key: "ready", class_name: "Класс", dataset_name: "Готовый" } as DatasetInfo;
    const missing = { key: "missing", class_name: "Класс", dataset_name: "Новый" } as DatasetInfo;
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

  it("сортирует варианты и отмечает доступные основные разметки", () => {
    const options = testMarkupDownloadOptions(downloadCatalog());

    expect(options.map(({ sample }) => sample.id)).toEqual([
      "empty-primary",
      "primary",
      "new",
      "old",
    ]);
    expect(options[1].datasetName).toBe("main");
    expect(initialTestMarkupDownloadSelection(options)).toEqual(new Set(["primary"]));
  });

  it("оставляет одну разметку датасета и снимает все отметки", () => {
    const options = testMarkupDownloadOptions(downloadCatalog());
    const initial = initialTestMarkupDownloadSelection(options);
    const replaced = changeTestMarkupDownloadSelection(
      options,
      initial,
      { type: "toggle", sampleId: "new", checked: true },
    );

    expect(replaced).toEqual(new Set(["new"]));
    expect(changeTestMarkupDownloadSelection(
      options,
      replaced,
      { type: "clear" },
    )).toEqual(new Set());
  });
});
