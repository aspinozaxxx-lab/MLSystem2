import type { ConfigField, ConfigSchema, DatasetInfo, JsonRecord } from "../api/types";

export function trainingConfigSchema(
  schema: ConfigSchema | undefined,
  task: DatasetInfo["task"],
  pipelineVariant = "legacy",
): ConfigSchema | undefined {
  if (!schema) return undefined;
  const allowedLosses =
    pipelineVariant === "next_gen2"
      ? ["cross_entropy_tversky"]
      : task === "multiclass"
      ? ["cross_entropy", "cross_entropy_dice"]
      : ["bce_dice", "focal_dice", "focal_tversky"];
  return {
    ...schema,
    fields: schema.fields.map((field) =>
      field.key === "train.loss"
        ? {
            ...field,
            options: allowedLosses,
            tooltip:
              task === "multiclass"
                ? "Multiclass loss: cross entropy, отдельно или вместе с Dice."
                : field.tooltip,
          }
        : field,
    ),
  };
}


export function coerceConfigValue(field: ConfigField, raw: unknown): unknown {
  if (field.value_type === "boolean") return Boolean(raw);
  if (field.value_type.startsWith("integer")) {
    const number = Number.parseInt(String(raw), 10);
    return Number.isFinite(number) ? number : null;
  }
  if (field.value_type.startsWith("number")) {
    const number = Number.parseFloat(String(raw));
    return Number.isFinite(number) ? number : null;
  }
  return String(raw);
}

export function configFieldTooltip(field: ConfigField): string {
  const parts = [field.tooltip, configAllowedRange(field), field.recommended_range ? `Рекомендуется: ${field.recommended_range}` : ""];
  return parts.filter(Boolean).join(" · ");
}


function configAllowedRange(field: ConfigField): string {
  const min = field.min_value;
  const max = field.max_value;
  if (min !== null && min !== undefined && max !== null && max !== undefined) return `${min}..${max}`;
  if (min !== null && min !== undefined) return `>= ${min}`;
  if (max !== null && max !== undefined) return `<= ${max}`;
  return "";
}


export function configWithField(
  value: JsonRecord,
  key: string,
  nextValue: unknown,
  pipelineDefaults?: Record<string, JsonRecord>,
): JsonRecord {
  if (value["train.pipeline_variant"] === "next_gen2" && ![
    "train.pipeline_variant", "train.epochs", "train.early_stopping_patience", "train.max_training_time_sec",
  ].includes(key)) return value;
  const preset = key === "train.pipeline_variant" ? pipelineDefaults?.[String(nextValue)] : undefined;
  const next = { ...value, ...preset, [key]: nextValue };
  if (key === "train.pipeline_variant" && nextValue === "next_gen") {
    next["train.max_val_batches_per_epoch"] = null;
    if (value["train.pipeline_variant"] === "next_gen2") next["train.loss"] = "bce_dice";
  }
  return next;
}

export function trainingConfigFieldVisible(
  key: string,
  pipelineVariant: string,
  architecture?: string,
): boolean {
  if (key.startsWith("next_gen.")) return pipelineVariant === "next_gen";
  if (pipelineVariant === "next_gen2") return [
    "train.pipeline_variant", "train.epochs", "train.early_stopping_patience", "train.max_training_time_sec",
  ].includes(key);
  if (key === "train.pretrained") {
    return pipelineVariant === "next_gen" && architecture === "segformer_b0";
  }
  return true;
}

