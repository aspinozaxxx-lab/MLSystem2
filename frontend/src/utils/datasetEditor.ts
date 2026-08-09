export type JsonObject = Record<string, unknown>;
export type SortDirection = "ascending" | "descending";

export type DraftSnapshot = {
  geojson: JsonObject;
  newFeatureIndexes: number[];
};

export type DraftState = {
  baseline: DraftSnapshot;
  current: DraftSnapshot;
  history: DraftSnapshot[];
};

export type CountedScene = {
  annotation_name: string;
  total_count: number;
  positive_count: number;
  hard_negative_count: number;
};

export type PublishableDraft = DraftState & {
  scene: CountedScene & { revision: string };
};

export function cloneSnapshot(snapshot: DraftSnapshot): DraftSnapshot {
  return {
    geojson: JSON.parse(JSON.stringify(snapshot.geojson)) as JsonObject,
    newFeatureIndexes: [...snapshot.newFeatureIndexes],
  };
}

export function featureCounts(geojson: JsonObject): {
  total: number;
  positive: number;
  hardNegative: number;
} {
  const features = Array.isArray(geojson.features) ? geojson.features : [];
  let positive = 0;
  let hardNegative = 0;
  for (const feature of features) {
    if (!feature || typeof feature !== "object") continue;
    const properties = (feature as JsonObject).properties;
    const role =
      properties && typeof properties === "object"
        ? (properties as JsonObject)._mlsystem2_role
        : undefined;
    if (role === "hard_negative") hardNegative += 1;
    else positive += 1;
  }
  return { total: positive + hardNegative, positive, hardNegative };
}

export function draftChanged(draft: DraftState): boolean {
  return canonicalJson(draft.baseline.geojson) !== canonicalJson(draft.current.geojson);
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortObjectKeys(value));
}

function sortObjectKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortObjectKeys);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as JsonObject)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortObjectKeys(item)]),
  );
}

export function appendHistory(
  history: DraftSnapshot[],
  snapshot: DraftSnapshot,
  limit = 100,
): DraftSnapshot[] {
  return [...history, cloneSnapshot(snapshot)].slice(-limit);
}

export function undoDraft(draft: DraftState): DraftState | null {
  const previous = draft.history.at(-1);
  if (!previous) return null;
  return {
    ...draft,
    current: cloneSnapshot(previous),
    history: draft.history.slice(0, -1),
  };
}

export function sceneCounts(
  scene: CountedScene,
  draft: DraftState | undefined,
): { total: number; positive: number; hardNegative: number } {
  return draft
    ? featureCounts(draft.current.geojson)
    : {
        total: scene.total_count,
        positive: scene.positive_count,
        hardNegative: scene.hard_negative_count,
      };
}

export function sortEditorScenes<T extends CountedScene>(
  scenes: T[],
  drafts: Record<string, DraftState>,
  direction: SortDirection,
): T[] {
  const factor = direction === "ascending" ? 1 : -1;
  return [...scenes].sort((left, right) => {
    const difference =
      sceneCounts(left, drafts[left.annotation_name]).total -
      sceneCounts(right, drafts[right.annotation_name]).total;
    return (
      difference * factor ||
      left.annotation_name.localeCompare(right.annotation_name, "ru", { sensitivity: "base" })
    );
  });
}

export function publishScenes(
  drafts: Record<string, PublishableDraft>,
): Array<{ annotation_name: string; revision: string; geojson: JsonObject }> {
  return Object.values(drafts)
    .filter(draftChanged)
    .sort((left, right) =>
      left.scene.annotation_name.localeCompare(right.scene.annotation_name, "ru", {
        sensitivity: "base",
      }),
    )
    .map((draft) => ({
      annotation_name: draft.scene.annotation_name,
      revision: draft.scene.revision,
      geojson: draft.current.geojson,
    }));
}

export function acceptPublishedDraft<T extends PublishableDraft>(
  draft: T,
  scene: T["scene"],
): T {
  const saved = cloneSnapshot({
    geojson: draft.current.geojson,
    newFeatureIndexes: [],
  });
  return {
    ...draft,
    scene,
    baseline: cloneSnapshot(saved),
    current: cloneSnapshot(saved),
    history: [],
  };
}
