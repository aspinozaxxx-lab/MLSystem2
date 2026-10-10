import type { TestF1SceneInfo, TestF1ScoreInfo } from "../api/types";

export const TEST_F1_LAYERS = {
  tp: { color: "#00ff66", label: "TP · Совпадение" },
  fp: { color: "#ff1744", label: "FP · Лишнее" },
  fn: { color: "#ffea00", label: "FN · Пропуск" },
  reference: { color: "#c084fc", label: "Эталон" },
  predicted: { color: "#00d9ff", label: "Прогноз" },
} as const;

export function comparisonLayerLabel(layer: string, metric: "pixel" | "objects" = "pixel"): string {
  if (metric === "objects") {
    if (layer === "tp") return "TP · Найденные объекты";
    if (layer === "fp") return "FP · Лишние объекты";
    if (layer === "fn") return "FN · Пропущенные объекты";
  }
  return TEST_F1_LAYERS[layer as keyof typeof TEST_F1_LAYERS]?.label ?? layer;
}

export function comparisonLayerStyle(layer: string, reference: boolean, predicted: boolean) {
  if (!(layer in TEST_F1_LAYERS)) return null;
  const name = layer as keyof typeof TEST_F1_LAYERS;
  const color = TEST_F1_LAYERS[name].color;
  if (name === "reference" || name === "predicted") {
    if (!(name === "reference" ? reference : predicted)) return null;
    return { color, fill: reference && predicted ? "transparent" : `${color}b3`, width: 2 };
  }
  return reference && predicted ? { color, fill: `${color}cc`, width: 1 } : null;
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
