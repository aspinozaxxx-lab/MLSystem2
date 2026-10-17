import { describe, expect, it } from "vitest";
import type { AnnotationVersionCatalog } from "./api/types";
import { retainVersionEdit } from "./DatasetAnnotationVersions";
import { appendHistory, cloneSnapshot, draftChanged, snapshotsEqual, undoDraft, type DraftSnapshot } from "./utils/datasetEditor";

const catalog: AnnotationVersionCatalog = {
  image_id: "снимок", revision: "ревизия", selected_id: "первая",
  versions: [{ id: "первая", name: "Основная", revision: 1, usages: [] }, { id: "вторая", name: "Уточнённая", revision: 1, usages: [] }],
};
const baseline: DraftSnapshot = {
  geojson: { type: "FeatureCollection", features: [] }, newFeatureIndexes: [],
  annotationVersions: { catalog_revision: "ревизия", selected_id: "первая", edits: [], replacements: {} },
};

describe("выбор версии входит в черновик и историю", () => {
  it("считает переключение изменением даже при одинаковых полигонах и отменяет его одним действием", () => {
    const current = cloneSnapshot(baseline);
    current.annotationVersions!.selected_id = "вторая";
    const draft = { baseline, current, history: appendHistory([], baseline) };
    expect(draftChanged(draft)).toBe(true);
    expect(snapshotsEqual(baseline, current)).toBe(false);
    const undone = undoDraft(draft)!;
    expect(undone.current.annotationVersions?.selected_id).toBe("первая");
    expect(draftChanged(undone)).toBe(false);
  });

  it("сохраняет правки предыдущей версии при переходе к другой", () => {
    const current = cloneSnapshot(baseline);
    current.geojson = { ...current.geojson, features: [{ id: "новый объект" }] };
    const state = retainVersionEdit(current, catalog, baseline.geojson);
    state.selected_id = "вторая";
    expect(state.edits?.[0].id).toBe("первая");
    expect(state.edits?.[0].geojson.features).toEqual([{ id: "новый объект" }]);
    expect(baseline.annotationVersions?.edits).toEqual([]);
  });

  it("не создаёт правку общей версии при копировании без изменения полигонов", () => {
    expect(retainVersionEdit(baseline, catalog, baseline.geojson).edits).toEqual([]);
  });

  it("отмена удаления возвращает замену и созданную версию вместе с геометрией", () => {
    const current = cloneSnapshot(baseline);
    current.annotationVersions!.selected_id = "вторая";
    current.annotationVersions!.replacements = { первая: "вторая" };
    const restored = undoDraft({ baseline, current, history: [cloneSnapshot(baseline)] })!;
    expect(restored.current.annotationVersions?.replacements).toEqual({});
    expect(current.annotationVersions?.replacements).toEqual({ первая: "вторая" });
  });
});
