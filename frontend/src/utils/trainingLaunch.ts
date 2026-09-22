import type { JsonRecord } from "../api/types";

export const SAMPLE_FACTOR_KEYS = [
  "tile_preparation.positive_factor",
  "tile_preparation.hard_negative_factor",
  "tile_preparation.background_factor",
] as const;

export type SamplePercentages = [number, number, number];
export type TrainingSection = "model" | "data" | "training" | "stopping";

const STOPPING_FIELDS = new Set([
  "train.epochs", "train.early_stopping_patience", "train.max_training_time_sec",
  "train.threshold", "train.max_val_batches_per_epoch", "next_gen.validation_fold",
  "next_gen.validation_interval_epochs", "next_gen.threshold_mode", "next_gen.evaluate_gaussian_blend",
]);

export function trainingSectionForField(key: string): TrainingSection {
  if (key === "train.pipeline_variant" || key === "train.pretrained") return "model";
  if (key.startsWith("tile_preparation.") || key === "dataset.val_fraction" || key === "next_gen.normalization") return "data";
  if (STOPPING_FIELDS.has(key)) return "stopping";
  return "training";
}

const roundPercent = (value: number) => Number(value.toFixed(6));
const clampPercent = (value: number, min = 0, max = 100) => roundPercent(Math.min(max, Math.max(min, value)));

export function samplePercentages(value: JsonRecord): SamplePercentages {
  return SAMPLE_FACTOR_KEYS.map((key) => roundPercent(Number(value[key] ?? 0) * 100)) as SamplePercentages;
}

function percentagesFromBoundaries(first: number, second: number): SamplePercentages {
  return [roundPercent(first), roundPercent(second - first), roundPercent(100 - second)];
}

export function moveSampleBoundary(values: SamplePercentages, boundary: 0 | 1, percent: number): SamplePercentages {
  if (!Number.isFinite(percent)) return values;
  const first = values[0];
  const second = roundPercent(values[0] + values[1]);
  return boundary === 0
    ? percentagesFromBoundaries(clampPercent(percent, 0, second), second)
    : percentagesFromBoundaries(first, clampPercent(percent, first, 100));
}

export function editSamplePercentage(values: SamplePercentages, index: number, percent: number): SamplePercentages {
  if (!Number.isFinite(percent)) return values;
  const target = clampPercent(percent);
  if (index === 0) return percentagesFromBoundaries(target, Math.max(values[0] + values[1], target));
  const first = Math.min(values[0], 100 - target);
  return percentagesFromBoundaries(first, index === 1 ? first + target : 100 - target);
}

export function configWithSamplePercentages(value: JsonRecord, percentages: SamplePercentages): JsonRecord {
  return { ...value, ...Object.fromEntries(SAMPLE_FACTOR_KEYS.map((key, index) => [key, percentages[index] / 100])) };
}

export function trainingNumber(value: unknown): string {
  return value == null || value === "" || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toLocaleString("ru-RU", { maximumFractionDigits: 6 });
}

export function trainingTile(value: JsonRecord) {
  const size = Number(value["tile_preparation.tile_size"]) || 0;
  const context = Number(value["tile_preparation.context"]) || 0;
  return { size, context, core: Math.max(0, size - 2 * context) };
}

export function trainingSummary({
  value, modelName, datasetName, secondaryPriority, runInferenceAfterTraining,
}: {
  value: JsonRecord;
  modelName: string;
  datasetName: string;
  secondaryPriority: boolean;
  runInferenceAfterTraining: boolean;
}): string {
  const tile = trainingTile(value);
  const variant = String(value["train.pipeline_variant"] || "legacy").replace("_", "-");
  const time = value["train.max_training_time_sec"];
  const patience = Number(value["train.early_stopping_patience"]);
  const checks = variant === "next-gen";
  const ending = (one: string, few: string, many: string) =>
    patience % 100 >= 11 && patience % 100 <= 14 ? many : patience % 10 === 1 ? one : patience % 10 >= 2 && patience % 10 <= 4 ? few : many;
  const patienceUnit = checks ? ending("проверки", "проверок", "проверок") : ending("эпохи", "эпох", "эпох");
  return `${modelName.replace(" (next-gen)", "")} на «${datasetName}», ${variant}. `
    + `Тайлы ${trainingNumber(tile.size)} × ${trainingNumber(tile.size)} px, центр ${trainingNumber(tile.core)} × ${trainingNumber(tile.core)} px; `
    + `валидация ${trainingNumber(value["dataset.val_fraction"] == null ? null : Number(value["dataset.val_fraction"]) * 100)}%. `
    + `Batch size ${trainingNumber(value["train.batch_size"])}, LR ${value["train.learning_rate"] ?? "—"}. `
    + `Максимум ${trainingNumber(value["train.epochs"])} эпох${time == null || time === "" ? "" : `, лимит ${trainingNumber(Number(time) / 60)} мин`}; `
    + `ранняя остановка после ${trainingNumber(value["train.early_stopping_patience"])} ${patienceUnit} без улучшения. `
    + `${secondaryPriority ? "Второстепенный" : "Обычный"} приоритет.`
    + (runInferenceAfterTraining ? " После успешного обучения — псевдоразметка всех снимков датасета." : "");
}
