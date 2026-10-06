import type { TrainingResultInfo } from "../api/types";

export type TrainingResultFamily = { id: string; stages: TrainingResultInfo[] };

/** Собирает продолжения по связям результатов, сохраняя псевдоразметки у их сети. */
export function trainingResultFamilies(results: TrainingResultInfo[]): TrainingResultFamily[] {
  const byId = new Map(results.map(result => [result.id, result]));
  const groups = new Map<string, TrainingResultFamily>();
  const depths = new Map<string, number>();
  for (const result of results) {
    let rootId = result.id;
    let depth = 0;
    const seen = new Set<string>();
    while (!seen.has(rootId)) {
      seen.add(rootId);
      const parentId = byId.get(rootId)?.continued_from_result_id;
      if (!parentId) break;
      rootId = parentId;
      depth++;
    }
    depths.set(result.id, depth);
    if (!groups.has(rootId)) groups.set(rootId, { id: rootId, stages: [] });
    groups.get(rootId)!.stages.push(result);
  }
  for (const family of groups.values()) {
    family.stages.sort((a, b) => a.created_at.localeCompare(b.created_at) || depths.get(a.id)! - depths.get(b.id)!);
  }
  return [...groups.values()];
}
