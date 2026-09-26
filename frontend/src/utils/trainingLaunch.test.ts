import { describe, expect, it } from "vitest";
import {
  configWithSamplePercentages, editSamplePercentage, moveSampleBoundary,
  samplePercentages, trainingSummary, type SamplePercentages,
} from "./trainingLaunch";

describe("баланс выборки", () => {
  it("сохраняет соседнюю границу при перетаскивании", () => {
    const first = moveSampleBoundary([60, 20, 20], 0, 45);
    expect(first).toEqual([45, 35, 20]);
    expect(moveSampleBoundary(first, 1, 65)).toEqual([45, 20, 35]);
    expect(moveSampleBoundary(first, 0, 100)).toEqual([80, 0, 20]);
    expect(moveSampleBoundary(first, 1, -10)).toEqual([45, 0, 55]);
  });

  it("позволяет последовательно ввести 60/20/20 и отдать одной зоне всю выборку", () => {
    const first = editSamplePercentage([50, 0, 50], 0, 60);
    expect(editSamplePercentage(first, 1, 20)).toEqual([60, 20, 20]);
    for (const index of [0, 1, 2]) {
      const values = editSamplePercentage([60, 20, 20], index, 100);
      expect(values[index]).toBe(100);
      expect(values.filter((_, i) => i !== index)).toEqual([0, 0]);
    }
  });

  it("не даёт отрицательных долей и сохраняет сумму при дробных значениях и нулевых зонах", () => {
    for (const start of [[0, 0, 100], [0, 100, 0], [100, 0, 0], [60, 20, 20], [33.333333, 33.333333, 33.333334]] as SamplePercentages[]) {
      for (const percent of [-20, 0, 0.01, 12.345678, 33.333333, 50, 99.99, 100, 120]) {
        for (const result of [
          ...[0, 1, 2].map((index) => editSamplePercentage(start, index, percent)),
          moveSampleBoundary(start, 0, percent), moveSampleBoundary(start, 1, percent),
        ]) {
          expect(result.every((value) => value >= 0 && value <= 100)).toBe(true);
          expect(result.reduce((sum, value) => sum + value, 0)).toBeCloseTo(100, 6);
          const config = configWithSamplePercentages({ "train.batch_size": 8 }, result);
          expect(config["train.batch_size"]).toBe(8);
          expect(samplePercentages(config)).toEqual(result);
        }
      }
    }
  });

  it("не переносит нечисловой ввод в конфигурацию", () => {
    const value: SamplePercentages = [60, 20, 20];
    expect(editSamplePercentage(value, 1, NaN)).toEqual(value);
    expect(moveSampleBoundary(value, 0, Infinity)).toEqual(value);
  });
});

describe("описание запуска", () => {
  const options = {
    modelName: "SegFormer B0 HF (next-gen)", datasetName: "Вырубки / Основной",
    secondaryPriority: false, runInferenceAfterTraining: false,
    value: {
      "train.pipeline_variant": "next_gen", "tile_preparation.tile_size": 768,
      "tile_preparation.context": 128, "dataset.val_fraction": 0.25,
      "train.batch_size": 8, "train.learning_rate": 1e-8, "train.epochs": 60,
      "train.early_stopping_patience": 4, "train.max_training_time_sec": 3300,
    },
  };
  it("отражает полезный центр, валидацию, ограничения и точный Learning rate", () => {
    const summary = trainingSummary(options);
    expect(summary).toContain("центр 512 × 512 px; валидация 25%");
    expect(summary).toContain("LR 1e-8");
    expect(summary).toContain("Максимум 60 эпох, лимит 55 мин");
    expect(summary).toContain("после 4 проверок без улучшения");
  });
  it("обновляет описание после изменения режима и флажков, учитывает пустой лимит", () => {
    const summary = trainingSummary({ ...options, secondaryPriority: true, runInferenceAfterTraining: true,
      value: { ...options.value, "train.pipeline_variant": "legacy", "train.max_training_time_sec": null },
    });
    expect(summary).toContain("после 4 эпох без улучшения");
    expect(summary).toContain("Второстепенный приоритет");
    expect(summary).toContain("псевдоразметка всех снимков датасета");
    expect(summary).not.toContain("лимит");
  });
});


it("объясняет разделение и выбор checkpoint в описании object f1", () => {
  const result = trainingSummary({modelName: "SegFormer B0", datasetName: "ОКС500", secondaryPriority: false,
    runInferenceAfterTraining: false, value: {"train.pipeline_variant": "object_f1", "train.pretrained": true,
      "tile_preparation.tile_size": 768, "train.batch_size": 8, "train.epochs": 30, "train.early_stopping_patience": 10}});
  expect(result).toContain("object f1");
  expect(result).toContain("по независимым снимкам; область и границы");
  expect(result).toContain("object F1 после выделения объектов");
});
