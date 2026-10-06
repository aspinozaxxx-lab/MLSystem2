import type { PseudoMarkupViewInfo } from "../api/types";
import type { PseudoProperties } from "./pseudoViewer";
import { TEST_F1_LAYERS } from "./testF1Viewer";

export const COMPARISON_COLORS = [TEST_F1_LAYERS.fp.color, TEST_F1_LAYERS.fn.color, "#00d9ff", "#c084fc", "#ff9100", "#ff40c8",
  "#b2ff59", "#448aff", "#ffffff", "#00e5b0", "#ff80ab", "#b388ff"];

export function comparisonScenes(views: PseudoMarkupViewInfo[]) {
  const scenes = new Map<string, PseudoMarkupViewInfo["scenes"][number] & { resultIds: string[] }>();
  for (const view of views) for (const scene of view.scenes) {
    const previous = scenes.get(scene.id);
    if (previous) previous.resultIds.push(view.id);
    else scenes.set(scene.id, { ...scene, resultIds: [view.id] });
  }
  return [...scenes.values()].sort((left, right) => left.name.localeCompare(right.name, "ru"));
}

export function comparisonAppearance(properties: PseudoProperties, ids: string[], colors: Record<string, string>,
  availableCount: number, intersection: boolean, differences: boolean) {
  const kind = String(properties.comparison_kind);
  const id = String(properties.comparison_result_id ?? "");
  const color = kind === "intersection" ? TEST_F1_LAYERS.tp.color : colors[id];
  if (!color || (kind !== "intersection" && !ids.includes(id))) return null;
  const highlight = availableCount >= 2 && (intersection || differences);
  if (kind === "layer") return { color, fill: highlight ? "transparent" : `${color}b3`, width: 2 };
  if (kind === "intersection" && intersection && availableCount >= 2) return { color, fill: `${color}cc`, width: 1 };
  if (kind === "difference" && differences && availableCount >= 2) return { color, fill: `${color}cc`, width: 1 };
  return null;
}
