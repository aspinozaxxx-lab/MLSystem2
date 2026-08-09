import { describe, expect, it } from "vitest";

import {
  acceptPublishedDraft,
  appendHistory,
  draftChanged,
  featureCounts,
  publishScenes,
  sortEditorScenes,
  undoDraft,
  type DraftSnapshot,
  type PublishableDraft,
} from "./datasetEditor";

const snapshot = (roles: string[], newFeatureIndexes: number[] = []): DraftSnapshot => ({
  geojson: {
    type: "FeatureCollection",
    features: roles.map((role) => ({
      type: "Feature",
      properties: { _mlsystem2_role: role },
      geometry: { type: "Polygon", coordinates: [] },
    })),
  },
  newFeatureIndexes,
});

const draft = (name: string, baseline: DraftSnapshot, current: DraftSnapshot): PublishableDraft => ({
  scene: {
    annotation_name: name,
    revision: `revision-${name}`,
    total_count: 0,
    positive_count: 0,
    hard_negative_count: 0,
  },
  baseline,
  current,
  history: [],
});

describe("черновики редактора датасетов", () => {
  it("считает роли из текущей разметки", () => {
    expect(featureCounts(snapshot(["positive", "hard_negative", "positive"]).geojson)).toEqual({
      total: 3,
      positive: 2,
      hardNegative: 1,
    });
  });

  it("сортирует по текущему количеству в обоих направлениях", () => {
    const scenes = [
      { annotation_name: "b.geojson", total_count: 5, positive_count: 5, hard_negative_count: 0 },
      { annotation_name: "a.geojson", total_count: 1, positive_count: 1, hard_negative_count: 0 },
    ];
    const drafts = {
      "a.geojson": draft("a.geojson", snapshot([]), snapshot(["positive", "positive"])),
    };
    expect(sortEditorScenes(scenes, drafts, "descending").map((item) => item.annotation_name)).toEqual([
      "b.geojson",
      "a.geojson",
    ]);
    expect(sortEditorScenes(scenes, drafts, "ascending").map((item) => item.annotation_name)).toEqual([
      "a.geojson",
      "b.geojson",
    ]);
  });

  it("ограничивает историю последними ста действиями", () => {
    let history: DraftSnapshot[] = [];
    for (let index = 0; index < 105; index += 1) {
      history = appendHistory(history, snapshot(Array(index).fill("positive")));
    }
    expect(history).toHaveLength(100);
    expect(featureCounts(history[0].geojson).total).toBe(5);
  });

  it("отменяет последний snapshot вместе с признаком нового объекта", () => {
    const baseline = snapshot([]);
    const previous = snapshot(["positive"], [0]);
    const current = snapshot(["positive", "hard_negative"], [0, 1]);
    const undone = undoDraft({ baseline, current, history: [previous] });
    expect(undone?.current).toEqual(previous);
    expect(undone?.history).toEqual([]);
  });

  it("формирует одну публикацию только из изменённых снимков", () => {
    const first = draft("b.geojson", snapshot([]), snapshot(["positive"], [0]));
    const second = draft("a.geojson", snapshot([]), snapshot(["hard_negative"], [0]));
    const clean = draft("c.geojson", snapshot(["positive"]), snapshot(["positive"]));
    expect(draftChanged(first)).toBe(true);
    expect(publishScenes({ first, second, clean })).toEqual([
      {
        annotation_name: "a.geojson",
        revision: "revision-a.geojson",
        geojson: second.current.geojson,
      },
      {
        annotation_name: "b.geojson",
        revision: "revision-b.geojson",
        geojson: first.current.geojson,
      },
    ]);

    const published = acceptPublishedDraft(first, {
      ...first.scene,
      revision: "new-revision",
      total_count: 1,
      positive_count: 1,
    });
    expect(draftChanged(published)).toBe(false);
    expect(published.current.newFeatureIndexes).toEqual([]);
    expect(published.history).toEqual([]);
  });
});
