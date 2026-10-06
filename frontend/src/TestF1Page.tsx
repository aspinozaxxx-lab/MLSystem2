import { ArrowLeft, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { apiJson } from "./api/client";
import type { PseudoMarkupViewInfo, TestF1SceneInfo, TestF1ViewInfo } from "./api/types";
import { PseudoMap, type ViewerGeoJson } from "./PseudoMarkupPage";
import { formatTestF1Percent } from "./utils/format";
import { sceneF1Score } from "./utils/testF1Viewer";
import "./styles/testF1Viewer.css";

const EMPTY_LAYERS: ViewerGeoJson = { type: "FeatureCollection", features: [] };

export function TestF1Page({ resultId, username }: { resultId: string; username: string }) {
  const [view, setView] = useState<TestF1ViewInfo | null>(null);
  const [sceneId, setSceneId] = useState("");
  const [classId, setClassId] = useState<number | null>(null);
  const [loaded, setLoaded] = useState<{ key: string; geojson: ViewerGeoJson } | null>(null);
  const [error, setError] = useState("");
  const [layerError, setLayerError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [preparing, setPreparing] = useState(false);
  const endpoint = `/results/training/${encodeURIComponent(resultId)}/test-f1/view`;
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setView(null); setError(""); setSceneId(""); setClassId(null);
    const poll = async () => {
      try {
        const next = await apiJson<TestF1ViewInfo>(endpoint);
        if (!active) return;
        setView(next);
        if (next.status === "queued" || next.status === "running") timer = setTimeout(() => void poll(), 5000);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "Не удалось открыть тестовый F1.");
      }
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [endpoint, attempt]);
  const scene = view?.scenes?.find((item) => item.id === sceneId) ?? view?.scenes?.[0];
  const effectiveClassId = scene?.target_class_id ?? (scene?.class_schema?.some((item) => Number(item.id) === classId) ? classId : null);
  const layerKey = `${scene?.id ?? ""}:${effectiveClassId ?? "foreground"}`;
  useEffect(() => {
    let active = true;
    setLayerError("");
    if (!scene) { setLoaded(null); return; }
    void apiJson<ViewerGeoJson>(`${scene.layers_url}${effectiveClassId === null ? "" : `?class_id=${effectiveClassId}`}`)
      .then((geojson) => {
        if (geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) throw new Error("Сохранённые слои повреждены.");
        if (active) setLoaded({ key: layerKey, geojson });
      }).catch((reason) => {
        if (active) setLayerError(reason instanceof Error ? reason.message : "Не удалось прочитать сохранённые слои.");
      });
    return () => { active = false; };
  }, [scene, effectiveClassId, layerKey, attempt]);
  const info = useMemo<PseudoMarkupViewInfo | null>(() => view && scene ? {
    id: view.training_result_id, training_result_id: view.training_result_id,
    model_name: view.model_name, training_dataset_name: view.training_dataset_name,
    source_dataset_name: `${scene.sample_name} · ${scene.name} · Ревизия ${scene.sample_revision}`,
    created_at: view.metric?.evaluated_at ?? "", geojson_url: scene.layers_url,
    scenes: scene.raster_available ? [scene] : [], warnings: view.warnings ?? [],
  } : null, [view, scene]);
  const quality = view?.metric?.quality_metric ?? "pixel";
  const prepare = async () => {
    setPreparing(true); setError("");
    try { await apiJson<TestF1ViewInfo>(endpoint, { method: "POST" }); setAttempt((value) => value + 1); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось подготовить визуализацию."); }
    finally { setPreparing(false); }
  };
  return <section className="panel pseudo-viewer test-f1-page">
    <a className="pseudo-back" href="#/results"><ArrowLeft size={16} /> К результатам</a>
    {error ? <div className="info-box" role="alert">{error} <button type="button" className="secondary compact-action" onClick={() => setAttempt((value) => value + 1)}>Повторить</button></div> : null}
    {!view && !error ? <p role="status">Загружаем сохранённый расчёт F1…</p> : null}
    {view ? <>
      {view.metric?.status === "stale" ? <p className="info-box">Показана сохранённая оценка прежней ревизии или настроек. {view.metric.error}</p> : null}
      {view.status === "queued" || view.status === "running" ? <p className="info-box" role="status"><RefreshCw size={15} className="spinning" /> {view.status === "queued" ? "Подготовка визуализации ожидает расчёта F1." : "Сеть обрабатывает тестовые снимки."}{view.metric?.progress?.total ? ` ${view.metric.progress.current ?? 0} из ${view.metric.progress.total} снимков.` : ""} {(view.scenes?.length ?? 0) > 0 ? "Ниже показан предыдущий сохранённый расчёт." : "Карта появится после завершения."}</p> : null}
      {view.status === "missing" || view.status === "error" || view.status === "unavailable" ? <div className="info-box">
        <p>{view.metric?.error || "Для этой исторической оценки предсказания не сохранены. Подготовка один раз пересчитает F1 текущей тестовой выборки и сохранит карту."}</p>
        <button className="primary compact-action" type="button" disabled={preparing} onClick={() => void prepare()}><RefreshCw size={15} className={preparing ? "spinning" : ""} /> {preparing ? "Подготовка…" : "Подготовить визуализацию"}</button>
      </div> : null}
      {layerError ? <p className="info-box" role="alert">{layerError} <button type="button" className="secondary compact-action" onClick={() => setAttempt((value) => value + 1)}>Повторить</button></p> : null}
      {info && scene ? <PseudoMap info={info} geojson={loaded?.key === layerKey ? loaded.geojson : EMPTY_LAYERS} username={username} onRetry={() => setAttempt((value) => value + 1)} comparison={{
          sidebar: <>
            <TestScenes scenes={view.scenes ?? []} selected={scene.id} quality={quality} onSelect={(item) => { setSceneId(item.id); setClassId(item.target_class_id ?? null); }} />
            {(scene.class_schema?.length ?? 0) > 0 ? <label className="test-f1-class">Тип объектов<select aria-label="Тип объектов для сравнения" value={effectiveClassId ?? "foreground"} disabled={scene.target_class_id != null} onChange={(event) => setClassId(event.target.value === "foreground" ? null : Number(event.target.value))}>
              {scene.target_class_id == null ? <option value="foreground">Все объекты — без различения типов</option> : null}
              {(scene.class_schema ?? []).filter((item) => scene.target_class_id == null || Number(item.id) === scene.target_class_id).map((item) => <option key={String(item.id)} value={String(item.id)}>{String(item.name)}</option>)}
            </select></label> : null}
            {loaded?.key !== layerKey ? <p className="test-f1-metric-note" role="status">Загружаем эталон и прогноз снимка…</p> : null}
            {quality === "objects" ? <p className="test-f1-metric-note">Объектовый F1: сопоставление при IoU ≥ 0,5. Цвета показывают пересечение пикселей; пиксельный F1 снимка — {formatTestF1Percent(sceneF1Score(scene, "pixel", effectiveClassId).f1)}.</p> : null}
          </>,
          summary: <div className="test-f1-summary" role="group" aria-label="Оценки тестового F1">
            <div className="test-f1-overall"><span title={view.metric?.aggregation === "macro" ? "Среднее F1 по классам всей выборки" : "F1 всей тестовой выборки"}>{quality === "objects" ? "Объектовый F1" : "Пиксельный F1"} · выборка</span><strong>{formatTestF1Percent(view.metric?.f1)}</strong></div>
            <div className="test-f1-image-score"><span>{quality === "objects" ? "Объектовый F1" : "Пиксельный F1"} · снимок</span><strong>{formatTestF1Percent(sceneF1Score(scene, quality, effectiveClassId).f1)}</strong>
              <small>TP {sceneF1Score(scene, quality, effectiveClassId).true_positive.toLocaleString("ru-RU")} · FP {sceneF1Score(scene, quality, effectiveClassId).false_positive.toLocaleString("ru-RU")} · FN {sceneF1Score(scene, quality, effectiveClassId).false_negative.toLocaleString("ru-RU")}</small>
            </div>
          </div>,
        }} /> : null}
    </> : null}
  </section>;
}

function TestScenes({ scenes, selected, quality, onSelect }: {
  scenes: TestF1SceneInfo[]; selected: string; quality: "pixel" | "objects"; onSelect: (scene: TestF1SceneInfo) => void;
}) {
  return <>
    <h2>Тестовые снимки <span>{scenes.length}</span></h2>
    <select className="test-f1-mobile-scenes" aria-label="Тестовый снимок" value={selected} onChange={(event) => { const scene = scenes.find((item) => item.id === event.target.value); if (scene) onSelect(scene); }}>
      {scenes.map((item) => <option key={item.id} value={item.id}>{item.name} · {quality === "objects" ? "Объектовый F1" : "Пиксельный F1"} {formatTestF1Percent(item[quality].f1)} · {item.sample_name}</option>)}
    </select>
    <div className="test-f1-scene-list">{scenes.map((item) => <button type="button" key={item.id} aria-pressed={selected === item.id} title={`${quality === "objects" ? "Объектовый F1" : "Пиксельный F1"}: ${formatTestF1Percent(item[quality].f1)}`} onClick={() => onSelect(item)}>
      <span><strong>{item.name}</strong><small>{item.sample_name}</small></span><b>{formatTestF1Percent(item[quality].f1)}</b>
    </button>)}</div>
  </>;
}
