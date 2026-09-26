import type { ConfigField, ConfigSchema, DatasetInfo, JsonRecord, TrainingTemplate } from "../api/types";

export function trainingConfigForTemplate(
  template: Pick<TrainingTemplate, "architecture" | "default_config" | "config_schema"> | undefined,
  task: DatasetInfo["task"],
  pipelineChoice: string | null,
  pretrainedChoice: boolean | null = null,
): JsonRecord {
  const next = { ...(template?.default_config || {}) };
  const variant = task === "multiclass" ? "legacy" : pipelineChoice;
  const preset = variant ? template?.config_schema.pipeline_defaults?.[variant] : undefined;
  if (preset && next["train.pipeline_variant"] !== variant) Object.assign(next, preset);
  if (task === "multiclass") {
    next["train.pipeline_variant"] = "legacy";
    next["train.loss"] = "cross_entropy_dice";
  }
  if (pretrainedChoice !== null && template?.architecture.startsWith("smp_segformer_")) {
    next["train.pretrained"] = pretrainedChoice;
  }
  return next;
}

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
      field.key === "train.pipeline_variant" && task === "multiclass"
        ? { ...field, options: ["legacy"] }
        : field.key === "train.loss"
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
    "train.pipeline_variant", "train.pretrained", "tile_preparation.tile_size", "train.epochs", "train.early_stopping_patience", "train.max_training_time_sec",
  ].includes(key)) return value;
  const preset = key === "train.pipeline_variant" ? pipelineDefaults?.[String(nextValue)] : undefined;
  const next = { ...value, ...preset, [key]: nextValue };
  if (key === "train.pipeline_variant" && typeof value["train.pretrained"] === "boolean") {
    next["train.pretrained"] = value["train.pretrained"];
  }
  if (next["train.pipeline_variant"] === "next_gen2") {
    const tileSize = Number(next["tile_preparation.tile_size"]);
    next["tile_preparation.stride"] = tileSize / 2;
    next["tile_preparation.context"] = 0;
    const baseBatch = Number(pipelineDefaults?.next_gen2?.["train.batch_size"] ?? 16);
    next["train.batch_size"] = Math.max(1, Math.ceil(baseBatch *
      ({512: 16, 768: 8, 1024: 4, 1536: 2} as Record<number, number>)[tileSize] / 16));
  }
  if (key === "train.pipeline_variant" && nextValue === "legacy" && next["train.loss"] === "cross_entropy_tversky") {
    next["train.loss"] = "bce_dice";
  }
  return next;
}

export function trainingConfigFieldVisible(
  key: string,
  pipelineVariant: string,
  architecture?: string,
): boolean {
  if (key.startsWith("next_gen.")) return false;
  if (key === "train.pretrained") return Boolean(architecture?.startsWith("smp_segformer_"));
  if (pipelineVariant === "next_gen2") return [
    "train.pipeline_variant", "tile_preparation.tile_size", "train.epochs", "train.early_stopping_patience", "train.max_training_time_sec",
  ].includes(key);
  return true;
}

