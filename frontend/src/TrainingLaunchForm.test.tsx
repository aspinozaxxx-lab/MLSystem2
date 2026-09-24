import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ConfigSchema, DatasetInfo, JsonRecord, TrainingTemplate } from "./api/types";
import { TrainingLaunchForm } from "./TrainingLaunchForm";

const schema: ConfigSchema = {
  pipeline_descriptions: { next_gen2: "Фиксированный профиль. Train/validation/test 60/20/20.\n\nВыбор весов по validation loss." },
  fields: [
    ...["train.pipeline_variant", "train.loss"].map((key) => ({ key, label: key === "train.loss" ? "Loss" : "Конвейер", value_type: "select", required: true, tooltip: "", options: key === "train.loss" ? ["bce_dice", "focal_tversky"] : ["legacy", "next_gen", "next_gen2"] })),
    ...["tile_preparation.tile_size", "tile_preparation.stride", "tile_preparation.context", "train.batch_size", "train.epochs", "train.early_stopping_patience"].map((key) => ({ key, label: key, value_type: "integer", required: true, tooltip: "" })),
    ...["tile_preparation.positive_factor", "tile_preparation.hard_negative_factor", "tile_preparation.background_factor", "dataset.val_fraction", "train.learning_rate", "train.focal_alpha", "train.tversky_alpha"].map((key) => ({ key, label: key, value_type: "number", required: true, tooltip: "" })),
    { key: "train.max_training_time_sec", label: "Лимит времени", value_type: "integer-null", required: true, tooltip: "" },
  ],
};
const value: JsonRecord = {
  "train.pipeline_variant": "next_gen", "train.loss": "bce_dice", "tile_preparation.tile_size": 512,
  "tile_preparation.stride": 256, "tile_preparation.context": 128, "dataset.val_fraction": 0.2,
  "tile_preparation.positive_factor": 0.6, "tile_preparation.hard_negative_factor": 0.2,
  "tile_preparation.background_factor": 0.2, "train.batch_size": 8, "train.learning_rate": 0.00006,
  "train.epochs": 60, "train.early_stopping_patience": 4, "train.max_training_time_sec": 3300,
};
const dataset = { key: "forest", name: "Вырубки / Основной", task: "binary", imagery_type: "kanopus", input_channels: 4 } as DatasetInfo;
const template = { id: "base", display_name: "SegFormer B0 HF", version: 1, config_schema: schema, default_config: value } as TrainingTemplate;
const ignore = () => {};
function render(config = value) {
  return renderToStaticMarkup(<TrainingLaunchForm
    models={[{ architecture: "segformer_b0", display_name: "SegFormer B0 HF", input_channels: 4, output_channels: 1, pretrained: true }]}
    datasets={[dataset]} architecture="segformer_b0" datasetKey="forest" template={template} schema={schema} value={config}
    onArchitectureChange={ignore} onDatasetChange={ignore} onChange={ignore}
    runInferenceAfterTraining={false} onRunInferenceChange={ignore} secondaryPriority={false} onSecondaryPriorityChange={ignore}
    busy={false} onSubmit={ignore}
  />);
}

describe("форма запуска обучения", () => {
  it("открывает выбор модели первым и оставляет две опции вне сворачиваемых разделов", () => {
    const html = render();
    expect(html.match(/aria-expanded="true"/g)).toHaveLength(1);
    expect(html.match(/id="training-model-heading"[^>]+/g)?.[0]).toContain('aria-expanded="true"');
    expect(html.match(/<div[^>]+id="training-model-body"[^>]*>/)?.[0]).not.toContain("hidden");
    expect(html.match(/<section class="training-section"/g)).toHaveLength(4);
    const options = html.match(/<div class="training-launch-options"[\s\S]*?<\/fieldset>/)?.[0] || "";
    expect(options.match(/type="checkbox"/g)).toHaveLength(2);
    expect(options).not.toContain("hidden");
    expect(html).not.toContain("MLflow");
  });

  it("относит максимум эпох к остановке и сохраняет три редактируемые доли", () => {
    const html = render();
    const stopping = html.split('data-training-section="stopping"')[1];
    const training = html.split('data-training-section="training"')[1].split('data-training-section="stopping"')[0];
    expect(stopping).toContain('name="train.epochs"');
    expect(training).not.toContain('name="train.epochs"');
    for (const label of ["С объектами", "Hard negative", "Фон"]) expect(html).toContain(`aria-label="${label}, процент"`);
    expect(html.match(/role="slider"/g)).toHaveLength(2);
    expect(html).toContain("с объектами — hard negative — фон");
  });

  it("показывает проценты и минуты, сохраняя исходные доли и секунды в конфигурации", () => {
    const html = render();
    expect(html.match(/<input[^>]+name="dataset.val_fraction"[^>]*>/)?.[0]).toContain('value="20"');
    expect(html.match(/<input[^>]+name="train.max_training_time_sec"[^>]*>/)?.[0]).toContain('value="55"');
    expect(value["dataset.val_fraction"]).toBe(0.2);
    expect(value["train.max_training_time_sec"]).toBe(3300);
  });

  it("сохраняет Focal alpha для Focal Tversky", () => {
    expect(render({ ...value, "train.loss": "focal_tversky" })).toContain('name="train.focal_alpha"');
    expect(render({ ...value, "train.loss": "bce_dice" })).not.toContain('name="train.focal_alpha"');
  });

  it("убирает несовместимые поля next-gen2, оставляя схему тайла и условия остановки", () => {
    const html = render({ ...value, "train.pipeline_variant": "next_gen2", "tile_preparation.context": 0, "tile_preparation.stride": 512 });
    expect(html).not.toContain('name="tile_preparation.stride"');
    expect(html).not.toContain('name="tile_preparation.context"');
    expect(html).not.toContain('role="slider"');
    expect(html).toContain("полезный центр 512 на 512 пикселей");
    expect(html).toContain('name="train.epochs"');
    expect(html).toContain('name="train.early_stopping_patience"');
    expect(html).toContain('name="train.max_training_time_sec"');
    expect(html.match(/<input[^>]+type="number"/g)).toHaveLength(3);
    const tileSelect = html.match(/<select[^>]+name="tile_preparation.tile_size"[\s\S]*?<\/select>/)?.[0] || "";
    for (const size of [512, 768, 1024, 1536]) expect(tileSelect).toContain(`value="${size}"`);
    expect(tileSelect.match(/<option/g)).toHaveLength(4);
    expect(html).not.toContain('name="train.batch_size"');
    expect(html).not.toContain('name="train.learning_rate"');
    expect(html).toContain("Фиксированный профиль. Train/validation/test 60/20/20.");
    expect(html).toContain("Выбор весов по validation loss.");
  });
  it("обновляет схему тайла и краткое описание для большого размера next-gen2", () => {
    const html = render({ ...value, "train.pipeline_variant": "next_gen2", "tile_preparation.tile_size": 1536, "tile_preparation.context": 0, "tile_preparation.stride": 768 });
    expect(html).toContain("полезный центр 1536 на 1536 пикселей");
    expect(html).toContain("шаг 768 px, train/validation/test 60/20/20");
    expect(html).toContain('<option value="1536" selected="">1536</option>');
  });
});
