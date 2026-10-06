import { describe, expect, it } from "vitest";
import type { TrainingResultInfo } from "../api/types";
import { trainingResultFamilies } from "./trainingResults";

const result = (id: string, day: number, parent?: string): TrainingResultInfo => ({
  id, created_at: `2026-10-${String(day).padStart(2, "0")}T12:00:00Z`, model_name: "Сеть",
  architecture: "smp_unet_resnet34", status: "ok", continued_from_result_id: parent,
  source: "manual", pipeline_variant: "legacy", validation_fold: 0, is_primary: false,
  can_continue_training: true, input_channels: 4, quality_metric: "pixel", task: "binary",
});

describe("связанные этапы обучения", () => {
  it("держит цепочку и ответвления рядом, от исходной сети к новым этапам", () => {
    const family = trainingResultFamilies([
      result("третий", 4, "второй"), result("ветка", 3, "исходный"),
      result("второй", 2, "исходный"), result("исходный", 1),
    ]);
    expect(family.map(group => group.stages.map(stage => stage.id)))
      .toEqual([["исходный", "второй", "ветка", "третий"]]);
  });

  it("не объединяет независимые сети с одинаковым названием и сохраняет порядок групп", () => {
    const families = trainingResultFamilies([result("новая", 3), result("продолжение", 2, "старая"), result("старая", 1)]);
    expect(families.map(group => group.id)).toEqual(["новая", "старая"]);
  });

  it("оставляет каждую псевдоразметку и оценку у её собственных весов", () => {
    const source = { ...result("исходный", 1), epoch: 12, f1_score: 0.81,
      pseudo_markup_results: [{ id: "ранняя-псевдоразметка", status: "ok", source_dataset_name: "Снимки", created_at: "2026-10-01" }] } as TrainingResultInfo;
    const child = { ...result("новый", 2, source.id), epoch: 4, f1_score: 0.86,
      pseudo_markup_results: [{ id: "новая-псевдоразметка", status: "ok", source_dataset_name: "Снимки", created_at: "2026-10-02" }] } as TrainingResultInfo;
    const input = [child, source];
    const family = trainingResultFamilies(input)[0];
    expect(family.stages).toEqual([source, child]);
    expect(family.stages[0].pseudo_markup_results?.[0].id).toBe("ранняя-псевдоразметка");
    expect(family.stages[1].pseudo_markup_results?.[0].id).toBe("новая-псевдоразметка");
    expect(input).toEqual([child, source]);
  });

  it("сохраняет связь вариантов, когда исходная сеть отсутствует в выборке", () => {
    expect(trainingResultFamilies([result("второй", 2, "нет-в-выборке"), result("третий", 3, "нет-в-выборке")])[0].stages).toHaveLength(2);
  });
});
