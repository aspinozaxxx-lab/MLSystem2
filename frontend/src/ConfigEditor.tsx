import type { ConfigField, ConfigSchema, JsonRecord } from "./api/types";
import { coerceConfigValue, configFieldTooltip, configWithField, trainingConfigFieldVisible } from "./utils/trainingConfig";

export type FieldPresentation = { label?: string; unit?: string; scale?: number };

export function ConfigEditor({
  schema,
  value,
  onChange,
  architecture,
  readonly = false,
  onApplyField,
  presentation = {},
  showPipelineNote = true,
}: {
  schema: ConfigSchema;
  value: JsonRecord;
  onChange: (next: JsonRecord) => void;
  architecture?: string;
  readonly?: boolean;
  onApplyField?: (key: string, value: unknown) => void;
  presentation?: Record<string, FieldPresentation>;
  showPipelineNote?: boolean;
}) {
  const setField = (field: ConfigField, raw: unknown) => {
    const scale = presentation[field.key]?.scale || 1;
    const scaled = raw === "" || scale === 1 ? raw : Number(raw) / scale;
    const nextValue = coerceConfigValue(field,
      typeof scaled === "number" && field.value_type.startsWith("integer") ? Math.round(scaled) : scaled,
    );
    onChange(configWithField(value, field.key, nextValue, schema.pipeline_defaults));
  };
  const pipelineVariant = String(value["train.pipeline_variant"] || "legacy");
  return (
    <div className="config-grid">
      {(schema.fields || []).filter((field) =>
        trainingConfigFieldVisible(field.key, pipelineVariant, architecture),
      ).map((field) => {
        const current = value[field.key] ?? "";
        const nextGen2Tile = pipelineVariant === "next_gen2" && field.key === "tile_preparation.tile_size";
        const options = nextGen2Tile ? ["512", "768", "1024", "1536"] : field.options;
        const display = presentation[field.key];
        const labelText = display?.label || field.label;
        const scale = display?.scale || 1;
        const displayValue = current === "" || scale === 1 ? String(current) : String(Number((Number(current) * scale).toFixed(8)));
        const fieldTooltip = scale === 1 ? configFieldTooltip(field) : configFieldTooltip({
          ...field,
          min_value: field.min_value == null ? null : Number((field.min_value * scale).toFixed(6)),
          max_value: field.max_value == null ? null : Number((field.max_value * scale).toFixed(6)),
          recommended_range: null,
        }) + (display?.unit ? ` ${display.unit}` : "");
        const fixedPipelineVariant = field.key === "train.pipeline_variant" && options?.length === 1;
        const tooltip = nextGen2Tile
          ? "Размер квадратного тайла в пикселях. По умолчанию 512. Шаг равен половине размера; контекст отсутствует. Batch size автоматически учитывает архитектуру сети и размер тайла."
          : pipelineVariant === "next_gen2" && field.key === "train.epochs"
          ? "Максимальное число эпох. По умолчанию 20; обучение может завершиться раньше по early stopping или лимиту времени."
          : pipelineVariant === "next_gen2" && field.key === "train.early_stopping_patience"
          ? "Число полных эпох без уменьшения validation loss. По умолчанию 10."
          : pipelineVariant === "next_gen2" && field.key === "train.weight_decay"
            ? "Регуляризация AdamW. Исходный ноутбук использует значение по умолчанию 0.01."
            : fieldTooltip;
        const label = (
          <span title={tooltip || labelText}>
            <span>{labelText}</span>
          </span>
        );
        if (field.value_type === "boolean") {
          return (
            <label className="field checkbox-field" key={field.key} title={tooltip || field.label}>
              <input
                type="checkbox"
                name={field.key}
                checked={Boolean(current)}
                disabled={readonly || fixedPipelineVariant}
                onChange={(event) => setField(field, event.target.checked)}
              />
              <span>{labelText}</span>
              {onApplyField && !readonly ? (
                <button className="secondary compact-action" type="button" onClick={() => onApplyField(field.key, Boolean(current))}>
                  ко всем
                </button>
              ) : null}
            </label>
          );
        }
        return (
          <label className="field" key={field.key}>
            {label}
            <div className={`inline-row${display?.unit ? " config-unit-input" : ""}`}>
              {options?.length ? (
                <select
                  value={String(current)}
                  name={field.key}
                  disabled={readonly || fixedPipelineVariant}
                  title={tooltip || field.label}
                  onChange={(event) => setField(field, event.target.value)}
                >
                  {options.map((option) => (
                    <option value={option} key={option}>
                      {field.key === "train.pipeline_variant" ? option.replace("_", "-") : option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={field.value_type.startsWith("integer") || field.value_type.startsWith("number") ? "number" : "text"}
                  step={field.value_type.startsWith("number") || scale !== 1 ? "any" : "1"}
                  name={field.key}
                  value={displayValue}
                  min={field.min_value == null ? undefined : field.min_value * scale}
                  max={field.max_value == null ? undefined : field.max_value * scale}
                  disabled={readonly || fixedPipelineVariant}
                  required={field.required && !field.value_type.endsWith("-null")}
                  placeholder={field.value_type.endsWith("-null") ? "Без лимита" : undefined}
                  onChange={(event) => setField(field, event.target.value)}
                  title={tooltip || field.label}
                />
              )}
              {display?.unit ? <span className="config-unit" aria-hidden="true">{display.unit}</span> : null}
              {onApplyField && !readonly ? (
                <button className="secondary compact-action" type="button" onClick={() => onApplyField(field.key, value[field.key])}>
                  ко всем
                </button>
              ) : null}
            </div>
          </label>
        );
      })}
      {showPipelineNote && pipelineVariant === "next_gen2" ? (
        <div className="training-pipeline-description">
          {(schema.pipeline_descriptions?.next_gen2 || "").split("\n\n").map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
        </div>
      ) : null}
    </div>
  );
}

