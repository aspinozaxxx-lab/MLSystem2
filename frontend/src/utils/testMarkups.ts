import type {
  DatasetInfo,
  TestSampleCatalogResponse,
  TestSampleDetail,
  TestSampleDraftPreview,
  TestSampleSummary,
} from "../api/types";

export type TestMarkupDraft = {
  name: string;
  isPrimary: boolean;
  enabledTileIndices: number[];
};

export type TestMarkupStats = {
  count: number;
  hasPrimary: boolean;
};

export type TestMarkupDownloadOption = {
  datasetName: string;
  sample: TestSampleSummary;
};

export type TestMarkupDownloadSelectionChange =
  | { type: "clear" }
  | { type: "toggle"; sampleId: string; checked: boolean };

export function isDatasetReadyForTestMarkup(dataset: DatasetInfo): boolean {
  if (dataset.is_custom || (dataset.diagnostics || []).length) return false;
  const legacyReady = Boolean(dataset.scenes_file && dataset.annotation_file);
  const perImageReady = Boolean(
    dataset.annotations_dir
    && (dataset.image_count || 0) > 0,
  );
  return legacyReady || perImageReady;
}

export function flattenTestMarkups(catalog: TestSampleCatalogResponse | null): TestSampleSummary[] {
  return (catalog?.classes || []).flatMap((classGroup) =>
    (classGroup.samples || []).length
      ? classGroup.samples || []
      : (classGroup.datasets || []).flatMap((dataset) => dataset.samples || []),
  );
}

export function testMarkupDownloadOptions(
  catalog: TestSampleCatalogResponse | null,
): TestMarkupDownloadOption[] {
  return (catalog?.classes || [])
    .flatMap((classGroup) => {
      const samples = (classGroup.samples || []).length
        ? classGroup.samples || []
        : (classGroup.datasets || []).flatMap((dataset) => dataset.samples || []);
      return samples.map((sample) => ({
        datasetName: sample.source_dataset_name || sample.dataset_name,
        sample,
      }));
    })
    .sort((left, right) => {
      const classOrder = left.sample.class_name.localeCompare(right.sample.class_name, "ru");
      if (classOrder) return classOrder;
      const datasetOrder = left.datasetName.localeCompare(right.datasetName, "ru");
      if (datasetOrder) return datasetOrder;
      const primaryOrder = Number(right.sample.is_primary) - Number(left.sample.is_primary);
      if (primaryOrder) return primaryOrder;
      const createdOrder = right.sample.created_at.localeCompare(left.sample.created_at);
      return createdOrder || left.sample.id.localeCompare(right.sample.id);
    });
}

export function initialTestMarkupDownloadSelection(
  options: TestMarkupDownloadOption[],
): Set<string> {
  const selected = new Set<string>();
  const selectedClasses = new Set<string>();
  for (const { sample } of options) {
    if (
      sample.is_primary
      && sample.enabled_image_count > 0
      && !selectedClasses.has(sample.class_key)
    ) {
      selected.add(sample.id);
      selectedClasses.add(sample.class_key);
    }
  }
  return selected;
}

export function changeTestMarkupDownloadSelection(
  options: TestMarkupDownloadOption[],
  current: ReadonlySet<string>,
  change: TestMarkupDownloadSelectionChange,
): Set<string> {
  if (change.type === "clear") return new Set();

  const option = options.find(({ sample }) => sample.id === change.sampleId);
  const next = new Set(current);
  if (!option || option.sample.enabled_image_count <= 0) return next;
  if (!change.checked) {
    next.delete(change.sampleId);
    return next;
  }

  for (const candidate of options) {
    if (candidate.sample.class_key === option.sample.class_key) {
      next.delete(candidate.sample.id);
    }
  }
  next.add(change.sampleId);
  return next;
}

export function testMarkupStats(
  catalog: TestSampleCatalogResponse | null,
  classKey: string,
): TestMarkupStats {
  const samples = flattenTestMarkups(catalog).filter((sample) => sample.class_key === classKey);
  return {
    count: samples.length,
    hasPrimary: samples.some((sample) => sample.is_primary),
  };
}

export function sortTestMarkupDatasets(
  datasets: DatasetInfo[],
  catalog: TestSampleCatalogResponse | null,
): DatasetInfo[] {
  return [...datasets].sort((left, right) => {
    const primaryDifference = Number(testMarkupStats(catalog, left.class_key || left.key).hasPrimary)
      - Number(testMarkupStats(catalog, right.class_key || right.key).hasPrimary);
    if (primaryDifference) return primaryDifference;
    const leftLabel = `${left.class_name || left.name}\u0000${left.dataset_name || left.name}`;
    const rightLabel = `${right.class_name || right.name}\u0000${right.dataset_name || right.name}`;
    return leftLabel.localeCompare(rightLabel, "ru");
  });
}

export function testMarkupDraft(sample: TestSampleDetail): TestMarkupDraft {
  return {
    name: sample.name,
    isPrimary: sample.is_primary,
    enabledTileIndices: sortedIndices(
      (sample.tiles || []).filter((tile) => tile.enabled).map((tile) => tile.index),
    ),
  };
}

export function testMarkupDraftChanged(sample: TestSampleDetail, draft: TestMarkupDraft): boolean {
  const saved = testMarkupDraft(sample);
  return saved.name !== draft.name.trim()
    || saved.isPrimary !== draft.isPrimary
    || saved.enabledTileIndices.join(",") !== sortedIndices(draft.enabledTileIndices).join(",");
}

export function applyTestMarkupPreview(
  draft: TestMarkupDraft,
  preview: TestSampleDraftPreview,
): TestMarkupDraft {
  return {
    ...draft,
    enabledTileIndices: sortedIndices(preview.enabled_tile_indices || []),
  };
}

export function sortedIndices(values: number[]): number[] {
  return [...values].sort((left, right) => left - right);
}
