import { describe, expect, it } from "vitest";

import type {
  DatasetInfo,
  TestSampleCatalogResponse,
  TestSampleDetail,
  TestSampleDraftPreview,
} from "../api/types";
import {
  applyTestMarkupPreview,
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
});
