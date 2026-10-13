import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ConfigSchema, TrainingTemplate } from "./api/types";
import { ConfigEditor } from "./ConfigEditor";
import { newsArticles } from "./News";
import { configWithField, trainingConfigForTemplate, trainingConfigSchema, trainingPipelineLabel } from "./utils/trainingConfig";
import { trainingSummary } from "./utils/trainingLaunch";

const schema: ConfigSchema = {
  fields: [
    { key: "train.pipeline_variant", label: "Конвейер", value_type: "select", tooltip: "", required: true, options: ["legacy", "next_gen2", "object_f1", "object_f1_v2"] },
    { key: "tile_preparation.tile_size", label: "Размер тайла", value_type: "integer", tooltip: "", required: true },
    { key: "train.learning_rate", label: "Скорость обучения", value_type: "number", tooltip: "", required: true },
    { key: "train.pretrained", label: "Предобученные веса", value_type: "boolean", tooltip: "", required: true },
  ],
  pipeline_defaults: {
    legacy: { "train.pipeline_variant": "legacy", "train.loss": "bce_dice" },
    object_f1_v2: { "train.pipeline_variant": "object_f1_v2", "train.pretrained": true, "train.loss": "cross_entropy_tversky", "train.batch_size": 4, "tile_preparation.tile_size": 512 },
  },
  pipeline_descriptions: { object_f1_v2: "Фон, внутренность и широкая граница.\n\nВосстановление полных полигонов." },
};
const template: Pick<TrainingTemplate, "architecture" | "default_config" | "config_schema"> = {
  architecture: "smp_segformer_b2", default_config: { "train.pipeline_variant": "legacy" }, config_schema: schema,
};

describe("отдельный профиль object f1 v2", () => {
  it("сохраняет явный выбор и предобучение, но не допускает многоклассовый датасет", () => {
    const selected = trainingConfigForTemplate(template, "binary", "object_f1_v2", false);
    expect(selected["train.pipeline_variant"]).toBe("object_f1_v2");
    expect(selected["train.pretrained"]).toBe(false);
    expect(trainingConfigForTemplate({ ...template, default_config: selected }, "multiclass", "object_f1_v2")["train.pipeline_variant"]).toBe("legacy");
    expect(trainingConfigSchema(schema, "multiclass")?.fields[0].options).toEqual(["legacy", "next_gen2"]);
    const resized = configWithField(selected, "tile_preparation.tile_size", 768, schema.pipeline_defaults);
    expect(resized["train.batch_size"]).toBe(2);
    expect(resized["tile_preparation.stride"]).toBe(384);
    expect(configWithField(resized, "train.learning_rate", 0.1, schema.pipeline_defaults)).toEqual(resized);
    expect(configWithField(resized, "train.pipeline_variant", "legacy", schema.pipeline_defaults)["train.loss"]).toBe("bce_dice");
  });

  it("показывает понятное имя, доступные размеры и описание вместо редактирования фиксированных параметров", () => {
    const value = trainingConfigForTemplate(template, "binary", "object_f1_v2", true);
    const html = renderToStaticMarkup(<ConfigEditor schema={schema} value={value} architecture={template.architecture} onChange={() => {}} />);
    expect(html).toContain('value="object_f1_v2" selected="">object f1 v2');
    expect(html).toContain('name="train.pretrained"');
    expect(html).not.toContain('name="train.learning_rate"');
    for (const size of [512, 768, 1024, 1536]) expect(html).toContain(`value="${size}"`);
    expect(html).toContain("Восстановление полных полигонов");
    expect(trainingPipelineLabel("object_f1_v2")).toBe("object f1 v2");
    const summary = trainingSummary({ value, modelName: "SegFormer B2", datasetName: "ЗУ500", secondaryPriority: false, runInferenceAfterTraining: false });
    expect(summary).toContain("object f1 v2");
    expect(summary).toContain("фон, внутренность и широкая граница");
    expect(summary).toContain("object F1 после выделения объектов");
  });

  it("публикует статью с постоянным адресом и явно обозначенной схемой", () => {
    const article = newsArticles.find((item) => item.slug === "object-f1-v2-boundaries");
    expect(article?.date).toBe("2026-10-13");
    expect(article && "image" in article && article.image?.src).toBe("/news/object-f1-v2-boundaries.svg");
    expect(article && "image" in article && article.image?.alt).toContain("Схема");
    expect(article && "image" in article && article.image?.caption).toContain("не результат реального распознавания");
  });
});
