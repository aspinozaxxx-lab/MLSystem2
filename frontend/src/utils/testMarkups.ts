import type {
  DatasetInfo,
  ImageryType,
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

// Округлённые медианы объёма сохранённых тестовых TIFF, включая пирамиды.
// Форма использует постоянные коэффициенты и не читает исходные растры.
const testMarkupBytesPerPixel: Record<ImageryType, number> = { kanopus: 3, ortho: 4 };

export function estimateTestMarkupImageVolume(
  imageryType: ImageryType | null | undefined,
  tileSize: number,
  minImageCount: number,
  maxImageCount: number,
): { minBytes: number; maxBytes: number } | null {
  if (
    !imageryType || !Object.hasOwn(testMarkupBytesPerPixel, imageryType)
    || [tileSize, minImageCount, maxImageCount].some((value) => !Number.isSafeInteger(value) || value <= 0)
    || minImageCount > maxImageCount
  ) return null;

  const bytesPerImage = tileSize * tileSize * testMarkupBytesPerPixel[imageryType];
  const minBytes = bytesPerImage * minImageCount;
  const maxBytes = bytesPerImage * maxImageCount;
  return Number.isSafeInteger(maxBytes) ? { minBytes, maxBytes } : null;
}

export function formatTestMarkupImageVolume(bytes: number): string {
  const gigabyte = 1024 ** 3;
  const divisor = bytes >= gigabyte ? gigabyte : 1024 ** 2;
  const unit = bytes >= gigabyte ? "ГБ" : "МБ";
  return `${(bytes / divisor).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} ${unit}`;
}

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

export function containedImageOneToOneScale(
  viewportWidth: number,
  viewportHeight: number,
  imageWidth: number,
  imageHeight: number,
  maxScale: number,
): number {
  const dimensions = [viewportWidth, viewportHeight, imageWidth, imageHeight];
  if (
    dimensions.some((value) => !Number.isFinite(value) || value <= 0)
    || !Number.isFinite(maxScale)
    || maxScale < 1
  ) return 1;

  const fittedPixelRatio = Math.min(
    viewportWidth / imageWidth,
    viewportHeight / imageHeight,
  );
  return Math.min(maxScale, Math.max(1, 1 / fittedPixelRatio));
}
