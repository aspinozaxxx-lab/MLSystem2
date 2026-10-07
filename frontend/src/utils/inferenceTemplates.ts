import type { DatasetInfo, InferenceTemplate } from "../api/types";

export function inferenceTemplateForDataset(templates: InferenceTemplate[], datasets: DatasetInfo[], key: string): InferenceTemplate | undefined {
  const classKey = datasets.find((item) => item.key === key)?.class_key || key;
  return templates.find((template) => template.is_active && template.class_keys.includes(classKey));
}
