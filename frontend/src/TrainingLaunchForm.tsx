import { ChevronDown, Play } from "lucide-react";
import { type CSSProperties, type FormEvent, type ReactNode, useState } from "react";
import type { ConfigSchema, DatasetInfo, JsonRecord, ModelInfo, TrainingTemplate } from "./api/types";
import { ConfigEditor, type FieldPresentation } from "./ConfigEditor";
import { SampleBalance } from "./SampleBalance";
import { trainingConfigFieldVisible } from "./utils/trainingConfig";
import {
  configWithSamplePercentages, SAMPLE_FACTOR_KEYS, samplePercentages,
  trainingNumber, trainingSectionForField, trainingSummary, trainingTile, type TrainingSection,
} from "./utils/trainingLaunch";

const TILE_FIELDS = ["tile_preparation.tile_size", "tile_preparation.stride", "tile_preparation.context"];
const TRAIN_FIELDS = ["train.batch_size", "train.learning_rate", "train.weight_decay", "train.max_train_batches_per_epoch"];
const STOP_FIELDS = ["train.epochs", "train.early_stopping_patience", "train.max_training_time_sec"];
const FIELD_PRESENTATION: Record<string, FieldPresentation> = {
  "train.pipeline_variant": { label: "Конвейер" },
  "train.pretrained": { label: "Предобученные веса" },
  "dataset.val_fraction": { label: "Доля валидации", unit: "%", scale: 100 },
  "tile_preparation.tile_size": { unit: "px" },
  "tile_preparation.stride": { label: "Шаг", unit: "px" },
  "tile_preparation.context": { label: "Контекст", unit: "px" },
  "tile_preparation.augmentation_level": { label: "Уровень аугментаций" },
  "train.background_weight": { label: "Background weight" },
  "train.max_train_batches_per_epoch": { label: "Train batches / эпоха" },
  "train.max_val_batches_per_epoch": { label: "Validation batches / эпоха" },
  "train.epochs": { label: "Максимум эпох" },
  "train.max_training_time_sec": { label: "Лимит времени", unit: "мин", scale: 1 / 60 },
  "train.threshold": { label: "Threshold" },
};

type TrainingLaunchFormProps = {
  models: ModelInfo[];
  datasets: DatasetInfo[];
  architecture: string;
  onArchitectureChange: (value: string) => void;
  datasetKey: string;
  onDatasetChange: (value: string) => void;
  template?: TrainingTemplate;
  schema?: ConfigSchema;
  value: JsonRecord;
  onChange: (value: JsonRecord) => void;
  runInferenceAfterTraining: boolean;
  onRunInferenceChange: (value: boolean) => void;
  secondaryPriority: boolean;
  onSecondaryPriorityChange: (value: boolean) => void;
  busy: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void | Promise<void>;
};

export function TrainingLaunchForm({
  models, datasets, architecture, onArchitectureChange, datasetKey, onDatasetChange,
  template, schema, value, onChange, runInferenceAfterTraining, onRunInferenceChange,
  secondaryPriority, onSecondaryPriorityChange, busy, onSubmit,
}: TrainingLaunchFormProps) {
  const [openSection, setOpenSection] = useState<TrainingSection | null>("model");
  const variant = String(value["train.pipeline_variant"] || "legacy");
  const variantLabel = variant.replace("_", "-");
  const nextGen2 = variant === "next_gen2";
  const modelName = models.find((model) => model.architecture === architecture)?.display_name.replace(" (next-gen)", "") || "Модель не выбрана";
  const dataset = datasets.find((item) => item.key === datasetKey);
  const datasetName = dataset?.name || "Датасет не выбран";
  const tile = trainingTile(value);
  const tileError = tile.size > 0 && tile.core <= 0
    ? "Контекст должен оставлять ненулевой центр: уменьшите контекст или увеличьте размер тайла."
    : Number(value["tile_preparation.stride"]) > tile.size
    ? "Шаг не должен превышать размер тайла."
    : "";
  const presentation: Record<string, FieldPresentation> = {
    ...FIELD_PRESENTATION,
    "train.early_stopping_patience": { unit: "эпох" },
  };
  const fields = (schema?.fields || []).filter((field) => {
    if (!trainingConfigFieldVisible(field.key, variant, architecture)) return false;
    if (field.key === "train.focal_alpha") return ["focal_dice", "focal_tversky"].includes(String(value["train.loss"]));
    if (["train.tversky_alpha", "train.tversky_beta"].includes(field.key)) return value["train.loss"] === "focal_tversky";
    return true;
  });
  const groupKeys = (section: TrainingSection) => fields.filter((field) => trainingSectionForField(field.key) === section).map((field) => field.key);
  const renderFields = (keys: readonly string[]) => schema ? (
    <ConfigEditor schema={{ ...schema, fields: keys.flatMap((key) => fields.filter((field) => field.key === key)) }}
      value={value} onChange={onChange} architecture={architecture} presentation={presentation} showPipelineNote={false} />
  ) : null;

  const revealField = (form: HTMLFormElement, section: TrainingSection, field?: HTMLElement) => {
    setOpenSection(section);
    requestAnimationFrame(() => {
      field?.focus();
      if (field instanceof HTMLInputElement || field instanceof HTMLSelectElement) field.reportValidity();
      else form.querySelector(`#training-${section}-heading`)?.scrollIntoView({ block: "nearest" });
    });
  };
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy || !template || !schema) return;
    const form = event.currentTarget;
    const invalid = form.querySelector<HTMLInputElement | HTMLSelectElement>("input:invalid, select:invalid");
    if (invalid) {
      revealField(form, (invalid.closest<HTMLElement>("[data-training-section]")?.dataset.trainingSection as TrainingSection) || "model", invalid);
      return;
    }
    if (tileError) {
      revealField(form, "data", form.elements.namedItem(tile.core <= 0 ? "tile_preparation.context" : "tile_preparation.stride") as HTMLElement);
      return;
    }
    void onSubmit(event);
  };

  const section = (id: TrainingSection, number: string, title: string, summary: string, children: ReactNode) => (
    <section className="training-section" data-training-section={id}>
      <h2>
        <button className="training-section-toggle" id={`training-${id}-heading`} type="button"
          aria-expanded={openSection === id} aria-controls={`training-${id}-body`}
          onClick={() => setOpenSection(openSection === id ? null : id)}>
          <span className="training-section-number" aria-hidden="true">{number}</span>
          <span className="training-section-title">{title}</span>
          <span className="training-section-summary">{summary}</span>
          <ChevronDown size={17} aria-hidden="true" />
        </button>
      </h2>
      <div className="training-section-body" id={`training-${id}-body`} role="region"
        aria-labelledby={`training-${id}-heading`} hidden={openSection !== id}>
        {children}
      </div>
    </section>
  );

  const tilePreview = (<figure className="training-tile-preview">
                    <div>Вход {trainingNumber(tile.size)} × {trainingNumber(tile.size)} px</div>
                    <div className="training-tile-image" role="img" aria-label={`Вход ${tile.size} на ${tile.size} пикселей, контекст ${tile.context}, полезный центр ${tile.core} на ${tile.core} пикселей`}>
                      <div className="training-tile-core" style={{ "--training-core": `${tile.size ? Math.min(100, tile.core / tile.size * 100) : 0}%` } as CSSProperties}>
                        {tile.core > 0 ? <span>Полезный<br />центр</span> : null}
                      </div>
                    </div>
                    <figcaption>Центр {trainingNumber(tile.core)} × {trainingNumber(tile.core)} px</figcaption>
                  </figure>);

  return (
    <form className="training-launch" onSubmit={submit} noValidate aria-busy={busy}>
      <fieldset className="training-controls" disabled={busy}>
        <div className="training-accordion">
          {section("model", "01", "Модель и датасет", `${modelName} · ${datasetName} · ${variantLabel}`, <>
            <div className="training-source-fields">
              <label className="field"><span>Архитектура модели</span>
                <select name="architecture" value={architecture} onChange={(event) => onArchitectureChange(event.target.value)} required>
                  {models.map((model) => <option value={model.architecture} key={model.architecture}>{model.display_name}</option>)}
                </select>
              </label>
              {renderFields(["train.pipeline_variant"])}
              <label className="field"><span>Датасет</span>
                <select name="dataset_key" value={datasetKey} onChange={(event) => onDatasetChange(event.target.value)} required>
                  {datasets.map((item) => <option value={item.key} key={item.key}>{item.name}{item.image_count == null ? "" : ` (${item.image_count} снимков)`}</option>)}
                </select>
              </label>
              {nextGen2 ? renderFields(["tile_preparation.tile_size"]) : null}
            </div>
            {datasetKey === "custom" ? <div className="training-uploads">
              <label className="field"><span>Разметка GeoJSON</span><input name="annotation_geojson" type="file" accept=".geojson,application/geo+json" required /></label>
              <label className="field"><span>Список снимков TXT</span><input name="scenes_txt" type="file" accept=".txt,text/plain" required /></label>
            </div> : null}
            <div className="training-source-meta">
              {renderFields(["train.pretrained"])}
              {dataset?.imagery_type ? <span>{dataset.imagery_type === "ortho" ? "Ортофото" : "Канопус"}{dataset.input_channels ? ` · ${dataset.input_channels} канала` : ""} · {dataset.task}</span> : null}
              {template ? <span>Шаблон: {template.display_name} · версия {template.version}</span> : <span className="training-error">Нет шаблона для выбранной модели.</span>}
            </div>
          </>)}

          {!nextGen2 && section("data", "02", "Подготовка данных", `Тайл ${trainingNumber(tile.size)} px · центр ${trainingNumber(tile.core)} px · валидация ${trainingNumber(Number(value["dataset.val_fraction"]) * 100)}%`,
            <div className="training-data-workspace">
              <div className="training-tiling">
                <h3>Нарезка снимков</h3>
                <div className="training-tile-editor">
                  <div className="training-tile-fields">
                    {renderFields(TILE_FIELDS)}
                    <p className="training-help">Контекст по краям не участвует в loss и оценке качества.</p>
                  </div>
                  {tilePreview}
                </div>
                {tileError ? <p className="training-error" role="alert">{tileError}</p> : null}
              </div>
              <div className="training-sampling">
                <h3>Выборка и преобразования</h3>
                {renderFields(groupKeys("data").filter((key) => !TILE_FIELDS.includes(key) && !SAMPLE_FACTOR_KEYS.some((factor) => factor === key)))}
                {SAMPLE_FACTOR_KEYS.every((key) => fields.some((field) => field.key === key))
                  ? <SampleBalance value={samplePercentages(value)} onChange={(percentages) => onChange(configWithSamplePercentages(value, percentages))} />
                  : renderFields(SAMPLE_FACTOR_KEYS)}
              </div>
            </div>)}

          {!nextGen2 && section("training", "03", "Обучение", `Batch size ${trainingNumber(value["train.batch_size"])} · LR ${value["train.learning_rate"] ?? "—"} · ${value["train.loss"] ?? "—"}`,
            <div className="training-parameter-groups">
              <div><h3>Параметры обучения</h3>{renderFields(TRAIN_FIELDS)}</div>
              <div><h3>Loss и веса</h3>
                {renderFields(groupKeys("training").filter((key) => !TRAIN_FIELDS.includes(key)))}
              </div>
            </div>)}

          {section("stopping", nextGen2 ? "02" : "04", "Условия остановки", `Максимум ${trainingNumber(value["train.epochs"])} эпох · patience ${trainingNumber(value["train.early_stopping_patience"])}${value["train.max_training_time_sec"] == null ? "" : ` · ${trainingNumber(Number(value["train.max_training_time_sec"]) / 60)} мин`}`,
            <div className="training-parameter-groups training-stop-groups">
              <div><h3>Когда завершить обучение</h3>{renderFields(STOP_FIELDS)}
                <p className="training-help">Patience — эпохи без улучшения. Лимит времени проверяется после завершения эпохи.</p>
              </div>
              <div><h3>Валидация и оценка качества</h3>{renderFields(groupKeys("stopping").filter((key) => !STOP_FIELDS.includes(key)))}
                {nextGen2 ? <p className="training-note">Валидация каждую эпоху. Лучшие веса и ранняя остановка определяются по минимуму validation loss.</p> : null}
              </div>
            </div>)}
        </div>

        <div className="training-launch-options" role="group" aria-label="Очередь и результат">
          <label className="training-launch-check"><input type="checkbox" checked={runInferenceAfterTraining} onChange={(event) => onRunInferenceChange(event.target.checked)} />
            <span>Псевдоразметка после обучения<small>Все снимки датасета, после успешного завершения.</small></span>
          </label>
          <label className="training-launch-check"><input type="checkbox" checked={secondaryPriority} onChange={(event) => onSecondaryPriorityChange(event.target.checked)} />
            <span>Второстепенный приоритет<small>Обучение и псевдоразметка уступают ресурсы обычным заданиям.</small></span>
          </label>
        </div>
      </fieldset>

      {nextGen2 ? <section className="training-pipeline-description" aria-label="Описание next-gen2">
        <h2>Особенности next-gen2 <span>Профиль для выбранной SegFormer и размера тайла</span></h2>
        <div className="training-pipeline-details">
          <div>{(schema?.pipeline_descriptions?.next_gen2 || "").split("\n\n").map((paragraph) => <p key={paragraph}>{paragraph}</p>)}</div>
          {tilePreview}
        </div>
      </section> : null}

      <aside className="training-launch-summary" aria-label="Краткое описание обучения">
        <div><h2>Кратко об обучении <span>Обновляется при изменении параметров</span></h2>
          <p aria-live="polite" aria-atomic="true">{template ? trainingSummary({ value, modelName, datasetName, secondaryPriority, runInferenceAfterTraining }) : "Выберите модель с настроенным шаблоном обучения."}</p>
        </div>
        <button className="primary" type="submit" disabled={busy || !template || !schema}><Play size={16} />{busy ? "Запуск…" : "Запустить обучение"}</button>
      </aside>
    </form>
  );
}
