import { ArrowLeft, Download, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { apiDownloadGet, apiJson, downloadBlob } from "./api/client";
import type { PseudoMarkupViewInfo, TestF1SceneInfo, TestF1ViewInfo } from "./api/types";
import { PseudoMap, type ViewerGeoJson } from "./PseudoMarkupPage";
import { formatTestF1Percent } from "./utils/format";
import { sceneF1Score, testF1PredictionDownloadUrl } from "./utils/testF1Viewer";
import "./styles/testF1Viewer.css";

const EMPTY_LAYERS: ViewerGeoJson = { type: "FeatureCollection", features: [] };

export function TestF1Page({ resultId, username }: { resultId: string; username: string }) {
  const [view, setView] = useState<TestF1ViewInfo | null>(null);
  const [sceneId, setSceneId] = useState("");
  const [classId, setClassId] = useState<number | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<"pixel" | "objects" | null>(null);
  const layerCache = useRef(new Map<string, ViewerGeoJson>());
  const [loaded, setLoaded] = useState<{ key: string; geojson: ViewerGeoJson } | null>(null);
  const [error, setError] = useState("");
  const [layerError, setLayerError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [preparing, setPreparing] = useState(false);
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState("");
  const endpoint = `/results/training/${encodeURIComponent(resultId)}/test-f1/view`;
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setView(null); setError("");
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
  useEffect(() => {
    setSceneId(""); setClassId(null); setSelectedMetric(null); layerCache.current.clear();
    setDownloadOpen(false); setDownloadError("");
  }, [endpoint]);
  const scene = view?.scenes?.find((item) => item.id === sceneId) ?? view?.scenes?.[0];
  const effectiveClassId = scene?.target_class_id ?? (scene?.class_schema?.some((item) => Number(item.id) === classId) ? classId : null);
  const quality = selectedMetric ?? (view?.metric?.quality_metric === "objects" ? "objects" : "pixel");
  const layerKey = `${scene?.id ?? ""}:${effectiveClassId ?? "foreground"}:${quality}`;
  useEffect(() => {
    let active = true;
    setLayerError("");
    if (!scene) { setLoaded(null); return; }
    const cached = layerCache.current.get(layerKey);
    if (cached) { setLoaded({ key: layerKey, geojson: cached }); return; }
    const controller = new AbortController();
    const parameters = new URLSearchParams({ metric: quality });
    if (effectiveClassId !== null) parameters.set("class_id", String(effectiveClassId));
    void apiJson<ViewerGeoJson>(`${scene.layers_url}?${parameters}`, { signal: controller.signal })
      .then((geojson) => {
        if (geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) throw new Error("Сохранённые слои повреждены.");
        if (active) {
          layerCache.current.set(layerKey, geojson);
          if (layerCache.current.size > 6) layerCache.current.delete(layerCache.current.keys().next().value!);
          setLoaded({ key: layerKey, geojson });
        }
      }).catch((reason) => {
        if (active) setLayerError(reason instanceof Error ? reason.message : "Не удалось прочитать сохранённые слои.");
      });
    return () => { active = false; controller.abort(); };
  }, [scene, effectiveClassId, layerKey, quality, attempt]);
  const info = useMemo<PseudoMarkupViewInfo | null>(() => view && scene ? {
    id: view.training_result_id, training_result_id: view.training_result_id,
    checkpoint_kind: "best",
    model_name: view.model_name, training_dataset_name: view.training_dataset_name,
    source_dataset_name: `${scene.sample_name} · ${scene.name} · Ревизия ${scene.sample_revision}`,
    created_at: view.metric?.evaluated_at ?? "", geojson_url: scene.layers_url,
    scenes: scene.raster_available ? [scene] : [], warnings: view.warnings ?? [],
  } : null, [view, scene]);
  const sceneScore = scene ? sceneF1Score(scene, quality, effectiveClassId) : null;
  const prepare = async () => {
    setPreparing(true); setError("");
    try { await apiJson<TestF1ViewInfo>(endpoint, { method: "POST" }); setAttempt((value) => value + 1); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось подготовить визуализацию."); }
    finally { setPreparing(false); }
  };
  const download = async (layout: "per_image" | "single") => {
    if (!view || downloading) return;
    setDownloading(true); setDownloadError("");
    try {
      const file = await apiDownloadGet(testF1PredictionDownloadUrl(resultId, view.scenes ?? [], layout));
      downloadBlob(file.blob, file.filename || `псевдоразметка.${layout === "single" ? "geojson" : "zip"}`);
      setDownloadOpen(false);
    } catch (reason) { setDownloadError(reason instanceof Error ? reason.message : "Не удалось скачать псевдоразметку."); }
    finally { setDownloading(false); }
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
          title: "Тестовый F1", metric: quality,
          titleActions: <button className="secondary compact-action test-f1-download" type="button"
            aria-label="Скачать псевдоразметку" title="Скачать прогноз всех снимков сохранённого расчёта"
            disabled={view.status !== "ready" || downloading}
            onClick={() => { setDownloadError(""); setDownloadOpen(true); }}>
            <Download size={16} /><span>Скачать псевдоразметку</span>
          </button>,
          dialog: downloadOpen ? <div className="modal-backdrop" role="presentation">
            <section className="modal-card panel" role="dialog" aria-modal="true" aria-labelledby="test-f1-download-title">
              <header className="modal-header"><h2 id="test-f1-download-title">Скачать псевдоразметку</h2>
                <button className="ghost" type="button" disabled={downloading} onClick={() => setDownloadOpen(false)}>Закрыть</button></header>
              <div className="modal-body">
                <p>Предсказания всех тестовых снимков и типов объектов сохранённого расчёта F1.</p>
                {downloadError ? <p className="error-box" role="alert">{downloadError}</p> : null}
                <TestF1DownloadOptions busy={downloading} onCancel={() => setDownloadOpen(false)} onSubmit={download} />
              </div>
            </section>
          </div> : null,
          layerControls: <>
            <div className="test-f1-metric-switch" role="group" aria-label="Режим подсветки F1">
              <button type="button" aria-label="Пиксельная F1" aria-pressed={quality === "pixel"} onClick={() => setSelectedMetric("pixel")}>Пиксельная</button>
              <button type="button" aria-label="Объектовая F1" aria-pressed={quality === "objects"} onClick={() => setSelectedMetric("objects")}>Объектовая</button>
            </div>
          </>,
          hint: quality === "objects" ? "Объектовая F1: сопоставление один к одному при IoU ≥ 0,5. Зелёным показаны целиком найденные объекты и их эталоны, красным — лишние прогнозы, жёлтым — пропущенные эталоны. Слои можно отключать независимо." : "Пиксельная F1: зелёный — пересечение масок, красный — лишние пиксели прогноза, жёлтый — пропущенные пиксели эталона. Слои можно отключать независимо.",
          sidebar: <>
            <TestScenes scenes={view.scenes ?? []} selected={scene.id} quality={quality} onSelect={(item) => { setSceneId(item.id); setClassId(item.target_class_id ?? null); }} />
            {(scene.class_schema?.length ?? 0) > 0 ? <label className="test-f1-class">Тип объектов<select aria-label="Тип объектов для сравнения" value={effectiveClassId ?? "foreground"} disabled={scene.target_class_id != null} onChange={(event) => setClassId(event.target.value === "foreground" ? null : Number(event.target.value))}>
              {scene.target_class_id == null ? <option value="foreground">Все объекты — без различения типов</option> : null}
              {(scene.class_schema ?? []).filter((item) => scene.target_class_id == null || Number(item.id) === scene.target_class_id).map((item) => <option key={String(item.id)} value={String(item.id)}>{String(item.name)}</option>)}
            </select></label> : null}
            {loaded?.key !== layerKey ? <p className="test-f1-metric-note" role="status">{quality === "objects" ? "Сопоставляем объекты сохранённого прогноза…" : "Загружаем эталон и прогноз снимка…"}</p> : null}
          </>,
          summary: <div className="test-f1-summary" role="group" aria-label="Оценки тестового F1">
            <div className="test-f1-summary-labels"><span>F1</span><span title={view.metric?.aggregation === "macro" ? "Среднее F1 по классам всей выборки" : "F1 всей тестовой выборки"}>Выборка</span><span>Снимок</span></div>
            {(["pixel", "objects"] as const).map((metric) => <div key={metric} className={quality === metric ? "selected" : ""}>
              <span>{metric === "objects" ? "Объектовая" : "Пиксельная"}</span>
              <strong>{formatTestF1Percent(metric === "pixel" ? view.metric?.pixel_f1 : view.metric?.object_f1)}</strong>
              <strong>{formatTestF1Percent(sceneF1Score(scene, metric, effectiveClassId).f1)}</strong>
            </div>)}
            <small className="test-f1-counts">Снимок · TP {sceneScore?.true_positive.toLocaleString("ru-RU")} · FP {sceneScore?.false_positive.toLocaleString("ru-RU")} · FN {sceneScore?.false_negative.toLocaleString("ru-RU")} {quality === "objects" ? "объектов" : "пикселей"}</small>
          </div>,
        }} /> : null}
    </> : null}
  </section>;
}

export function TestF1DownloadOptions({ busy, onCancel, onSubmit }: {
  busy: boolean; onCancel: () => void; onSubmit: (layout: "per_image" | "single") => void;
}) {
  const [layout, setLayout] = useState<"per_image" | "single">("per_image");
  return <form className="form-stack" onSubmit={event => { event.preventDefault(); if (!busy) onSubmit(layout); }}>
    <fieldset className="download-mode-fieldset" disabled={busy}>
      <legend>Формат выгрузки</legend>
      <label className="download-mode-choice"><input type="radio" name="test-f1-download-mode" value="per_image"
        checked={layout === "per_image"} onChange={() => setLayout("per_image")} />
        <span><strong>По снимкам</strong><small>ZIP с отдельным GeoJSON прогноза для каждого тестового снимка</small></span></label>
      <label className="download-mode-choice"><input type="radio" name="test-f1-download-mode" value="single"
        checked={layout === "single"} onChange={() => setLayout("single")} />
        <span><strong>Один общий GeoJSON</strong><small>Все предсказания в одном файле с указанием исходного снимка</small></span></label>
    </fieldset>
    <div className="button-row download-dialog-actions">
      <button className="secondary" type="button" disabled={busy} onClick={onCancel}>Отмена</button>
      <button className="primary" type="submit" disabled={busy}><Download size={16} />
        {busy ? "Формирование…" : layout === "per_image" ? "Скачать ZIP" : "Скачать GeoJSON"}</button>
    </div>
  </form>;
}

function TestScenes({ scenes, selected, quality, onSelect }: {
  scenes: TestF1SceneInfo[]; selected: string; quality: "pixel" | "objects"; onSelect: (scene: TestF1SceneInfo) => void;
}) {
  return <>
    <h2>Тестовые снимки <span>{scenes.length}</span></h2>
    <select className="test-f1-mobile-scenes" aria-label="Тестовый снимок" value={selected} onChange={(event) => { const scene = scenes.find((item) => item.id === event.target.value); if (scene) onSelect(scene); }}>
      {scenes.map((item) => <option key={item.id} value={item.id}>{item.name} · {quality === "objects" ? "Объектовый F1" : "Пиксельный F1"} {formatTestF1Percent(item[quality].f1)} · {item.sample_name}</option>)}
    </select>
    <div className="test-f1-scene-list">{scenes.map((item) => <button type="button" key={item.id} aria-pressed={selected === item.id} onClick={() => onSelect(item)}>
      <span><strong>{item.name}</strong><small>{item.sample_name}</small></span><span className="test-f1-scene-scores">
        <small className={quality === "pixel" ? "selected" : ""}>Пикс. <b>{formatTestF1Percent(item.pixel.f1)}</b></small>
        <small className={quality === "objects" ? "selected" : ""}>Объект. <b>{formatTestF1Percent(item.objects.f1)}</b></small>
      </span>
    </button>)}</div>
  </>;
}
