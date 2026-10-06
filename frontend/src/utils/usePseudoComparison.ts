import { useCallback, useEffect, useMemo, useState } from "react";
import { apiJson } from "../api/client";
import type { PseudoMarkupComparisonCounts, PseudoMarkupComparisonLayers, PseudoMarkupComparisonViewport, PseudoMarkupPixelComparison } from "../api/types";
import { loadComparisonSceneCounts } from "./pseudoComparison";

export function usePseudoComparison(ids: string[], sceneId: string, available: boolean, sharedSceneIds: string[],
  revisions: Record<string, string>, attempt: number) {
  const contextKey = `${ids.join(",")}:${JSON.stringify(revisions)}:${attempt}`;
  const sceneKey = `${contextKey}:${sceneId}`;
  const [viewport, setViewport] = useState<{ sceneId: string; value: PseudoMarkupComparisonViewport } | null>(null);
  const [layers, setLayers] = useState<{ key: string; data: PseudoMarkupComparisonLayers } | null>(null);
  const [layerError, setLayerError] = useState<{ key: string; message: string } | null>(null);
  const [countsError, setCountsError] = useState<{ key: string; message: string } | null>(null);
  const [counts, setCounts] = useState<{ key: string; scenes: Record<string, PseudoMarkupPixelComparison>; warnings: string[] } | null>(null);
  const completed = useMemo(() => new Map<string, PseudoMarkupPixelComparison>(), [contextKey]);
  const layerCache = useMemo(() => new Map<string, PseudoMarkupComparisonLayers>(), [contextKey]);
  const onViewport = useCallback((value: PseudoMarkupComparisonViewport) => {
    setViewport((old) => old?.sceneId === sceneId && JSON.stringify(old.value) === JSON.stringify(value) ? old : { sceneId, value });
  }, [sceneId]);

  useEffect(() => {
    if (!available || !sceneId || viewport?.sceneId !== sceneId) return;
    const controller = new AbortController();
    const cacheKey = `${sceneKey}:${JSON.stringify(viewport.value)}`;
    setLayerError(null);
    const cached = layerCache.get(cacheKey);
    if (cached) {
      layerCache.delete(cacheKey); layerCache.set(cacheKey, cached);
      setLayers({ key: sceneKey, data: cached });
      return;
    }
    void apiJson<PseudoMarkupComparisonLayers>(`/results/pseudo-markup/compare/${sceneId}/layers`, {
      method: "POST", signal: controller.signal,
      body: { result_ids: ids, scene_revisions: revisions, viewport: viewport.value },
    }).then((data) => {
      if (controller.signal.aborted) return;
      layerCache.set(cacheKey, data);
      while (layerCache.size > 8) layerCache.delete(layerCache.keys().next().value!);
      setLayers({ key: sceneKey, data });
    }).catch((reason) => {
      if (!controller.signal.aborted) setLayerError({ key: sceneKey, message: reason instanceof Error ? reason.message : "Не удалось прочитать слои сравнения." });
    });
    return () => controller.abort();
  }, [sceneKey, available, viewport, layerCache]);

  const layersReady = layers?.key === sceneKey;
  useEffect(() => {
    // Первый кадр имеет приоритет перед любыми точными счётчиками, включая выбранную сцену.
    if (ids.length !== 2 || !sceneId || (available && !layersReady)) return;
    const controller = new AbortController();
    setCountsError(null);
    void loadComparisonSceneCounts(sharedSceneIds, sceneId, completed, controller.signal,
      (id, signal) => apiJson<PseudoMarkupComparisonCounts>("/results/pseudo-markup/compare/counts", {
        method: "POST", signal, body: { result_ids: ids, scene_revisions: revisions, scene_id: id },
      }),
      (data) => setCounts({ key: contextKey, scenes: Object.fromEntries(completed), warnings: data.warnings }),
    ).catch((reason) => {
      if (!controller.signal.aborted) setCountsError({ key: contextKey, message: reason instanceof Error ? reason.message : "Не удалось рассчитать пиксельные счётчики." });
    });
    return () => controller.abort();
  }, [contextKey, sceneId, layersReady, available, sharedSceneIds, completed]);

  return {
    layers: layersReady ? layers.data : null,
    counts: counts?.key === contextKey ? counts.scenes : {},
    warnings: counts?.key === contextKey ? counts.warnings : [],
    layerError: layerError?.key === sceneKey ? layerError.message : "",
    countsError: countsError?.key === contextKey ? countsError.message : "",
    onViewport,
  };
}
