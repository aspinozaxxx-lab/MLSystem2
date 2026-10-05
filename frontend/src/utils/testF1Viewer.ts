import type { TestF1SceneInfo, TestF1ScoreInfo } from "../api/types";

export const TEST_F1_LAYERS = {
  tp: { color: "#22c55e", label: "TP · Совпадение" },
  fp: { color: "#f43f5e", label: "FP · Лишнее" },
  fn: { color: "#fbbf24", label: "FN · Пропуск" },
  reference: { color: "#a78bfa", label: "Эталон" },
  predicted: { color: "#38bdf8", label: "Прогноз" },
} as const;

export function comparisonLayerStyle(layer: string, reference: boolean, predicted: boolean) {
  if (!(layer in TEST_F1_LAYERS)) return null;
  const name = layer as keyof typeof TEST_F1_LAYERS;
  const color = TEST_F1_LAYERS[name].color;
  if (name === "reference" || name === "predicted") {
    if (!(name === "reference" ? reference : predicted)) return null;
    return { color, fill: reference && predicted ? "transparent" : `${color}50`, width: 1.5 };
  }
  return reference && predicted ? { color, fill: `${color}80`, width: 0.5 } : null;
}

export function sceneF1Score(scene: TestF1SceneInfo, metric: "pixel" | "objects", classId: number | null): TestF1ScoreInfo {
  if (classId !== null && scene.target_class_id == null) {
    const type = scene.class_schema?.find((item) => Number(item.id) === classId);
    const classes = (scene.metrics?.[metric] as { per_class?: Record<string, TestF1ScoreInfo> } | undefined)?.per_class;
    const score = type ? classes?.[String(type.slug)] : undefined;
    if (score) return score;
  }
  return scene[metric];
}
