import { ArrowLeft, Blend, Eye, EyeOff, RefreshCw, Split } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { apiJson } from "./api/client";
import type { PseudoMarkupPixelComparison, PseudoMarkupViewInfo } from "./api/types";
import { PseudoMap, type ViewerGeoJson } from "./PseudoMarkupPage";
import { comparisonAppearance, comparisonScenes, comparisonTotal, COMPARISON_COLORS } from "./utils/pseudoComparison";
import { usePseudoComparison } from "./utils/usePseudoComparison";
import { TEST_F1_LAYERS } from "./utils/testF1Viewer";
import { formatDateTime } from "./utils/format";
import { useCompactLayout } from "./utils/useCompactLayout";
import "./styles/testF1Viewer.css";
import "./styles/pseudoComparison.css";

const EMPTY: ViewerGeoJson = { type: "FeatureCollection", features: [] };
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function PseudoComparisonPage({ ids, username }: { ids: string; username: string }) {
  const compact = useCompactLayout();
  const resultIds = useMemo(() => ids.split(","), [ids]);
  const validIds = resultIds.length >= 2 && resultIds.length <= 12 && resultIds.every((id) => UUID.test(id)) && new Set(resultIds).size === resultIds.length;
  const [views, setViews] = useState<PseudoMarkupViewInfo[]>([]);
  const [hidden, setHidden] = useState(new Set<string>());
  const [sceneId, setSceneId] = useState("");
  const [query, setQuery] = useState("");
  const [intersection, setIntersection] = useState(true);
  const [differences, setDifferences] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => { setHidden(new Set()); setSceneId(""); setQuery(""); }, [ids]);
  useEffect(() => {
    const controller = new AbortController();
    setViews([]); setError("");
    if (validIds) void Promise.all(resultIds.map((id) => apiJson<PseudoMarkupViewInfo>(`/results/pseudo-markup/${id}/view`, { signal: controller.signal })))
      .then((values) => { if (!controller.signal.aborted) setViews(values); })
      .catch((reason) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Не удалось открыть выбранные псевдоразметки."); });
    return () => controller.abort();
  }, [resultIds, validIds, attempt]);
  const scenes = useMemo(() => comparisonScenes(views), [views]);
  const scene = scenes.find((item) => item.id === sceneId) ?? scenes.find((item) => item.resultIds.length >= 2) ?? scenes[0];
  const activeIds = useMemo(() => views.filter((view) => !hidden.has(view.id)).map((view) => view.id), [views, hidden]);
  const revisions = useMemo(() => Object.fromEntries(scenes.map((item) => [item.id, new URL(item.raster_url, window.location.origin).searchParams.get("v") ?? ""])), [scenes]);
  const available = activeIds.filter((id) => scene?.resultIds.includes(id));
  const sharedSceneIds = useMemo(() => activeIds.length === 2 ? scenes.filter((item) => activeIds.every((id) => item.resultIds.includes(id))).map((item) => item.id) : [], [scenes, activeIds]);
  const { layers: currentLayers, counts, warnings, layerError, countsError, onViewport } = usePseudoComparison(
    activeIds, scene?.id ?? "", available.length > 0, sharedSceneIds, revisions, attempt);
  const complete = Object.keys(counts).length === sharedSceneIds.length;
  const total = useMemo(() => comparisonTotal(counts), [counts]);
  const colors = Object.fromEntries(views.map((view, index) => [view.id,
    activeIds.length === 2 && activeIds.includes(view.id) ? COMPARISON_COLORS[activeIds.indexOf(view.id)] : COMPARISON_COLORS[index]]));
  const numbers = activeIds.map((id) => views.findIndex((view) => view.id === id) + 1);
  const info = useMemo<PseudoMarkupViewInfo | null>(() => views[0] && scene ? { ...views[0], scenes: [scene],
    warnings: [...new Set(views.flatMap((view) => view.warnings))],
  } : null, [views, scene]);
  const sceneCounts = counts[scene?.id ?? ""];
  const retry = () => setAttempt((value) => value + 1);
  const toggleLayer = (id: string) => setHidden((old) => {
    const next = new Set(old); if (next.has(id)) next.delete(id); else next.add(id); return next;
  });
  return <section className="panel pseudo-viewer pseudo-comparison-page">
    <a className="pseudo-back" href="#/results"><ArrowLeft size={16} /> К результатам и списку сравнения</a>
    {!validIds ? <p className="info-box" role="alert">Выберите от двух до двенадцати разных готовых псевдоразметок в результатах обучения.</p> : null}
    {error || layerError || countsError ? <div className="info-box" role="alert">{error || layerError || countsError} <button className="secondary compact-action" type="button" onClick={retry}><RefreshCw size={14} /> Перечитать сравнение</button></div> : null}
    {validIds && !views.length && !error ? <p role="status">Загружаем выбранные псевдоразметки…</p> : null}
    {views.length && !scenes.length ? <p className="info-box">Исходные TIFF выбранных разметок недоступны. Пиксельное сравнение требует сохранённых снимков.</p> : null}
    {info && scene ? <PseudoMap info={info} geojson={(currentLayers?.geojson as ViewerGeoJson | undefined) ?? EMPTY} username={username} onRetry={retry} comparison={{
      title: "Сравнение псевдоразметок", subtitle: `${views.length} разметки · ${scenes.length} снимков · ${scene.name}`,
      onViewport,
      featureStyle: (properties) => comparisonAppearance(properties, activeIds, colors, available.length, intersection, differences),
      sidebar: <>
        <details className="pseudo-compare-layers" open={!compact}><summary>Разметки · включено {activeIds.length} из {views.length}</summary>
          <div>{views.map((view, index) => <div className="pseudo-compare-layer-card" key={view.id} data-active={!hidden.has(view.id)}><i style={{ backgroundColor: colors[view.id] }} /><span><strong>{index + 1}. {view.class_name || view.training_dataset_name}</strong><small title={`${view.training_dataset_name} · ${view.model_name}`}>{view.training_dataset_name} · {view.model_name}</small><small title={`${view.source_dataset_name} · ${formatDateTime(view.created_at)}${view.checkpoint_epoch != null ? ` · эпоха ${view.checkpoint_epoch}` : ""}`}>{view.source_dataset_name} · {formatDateTime(view.created_at)}{view.checkpoint_epoch != null ? ` · эпоха ${view.checkpoint_epoch}` : ""}</small></span></div>)}</div>
        </details>
        <h2>Снимки <span>{scenes.length}</span></h2>
        <select className="test-f1-mobile-scenes" aria-label="Снимок для сравнения" value={scene.id} onChange={(event) => setSceneId(event.target.value)}>{scenes.map((item) => <option key={item.id} value={item.id}>{item.name}{counts[item.id] ? ` · ∩ ${counts[item.id].intersection.toLocaleString("ru-RU")} · только ${numbers[0]}: ${counts[item.id].only_first.toLocaleString("ru-RU")} · только ${numbers[1]}: ${counts[item.id].only_second.toLocaleString("ru-RU")}` : ""}</option>)}</select>
        <input className="pseudo-compare-search" type="search" placeholder="Найти снимок" aria-label="Найти снимок для сравнения" value={query} onChange={(event) => setQuery(event.target.value)} />
        <div className="test-f1-scene-list">{scenes.filter((item) => item.name.toLocaleLowerCase().includes(query.toLocaleLowerCase())).map((item) => <button key={item.id} type="button" aria-pressed={scene.id === item.id} onClick={() => setSceneId(item.id)}>
          <span><strong>{item.name}</strong><small>Разметки: {item.resultIds.map((id) => views.findIndex((view) => view.id === id) + 1).join(", ")}</small>{counts[item.id] ? <PixelValues counts={counts[item.id]} numbers={numbers} /> : activeIds.length === 2 ? <small>{!sharedSceneIds.includes(item.id) ? "Нет общей пары на этом снимке" : scene.id === item.id ? "Считаем пиксели…" : "Ожидает фонового расчёта"}</small> : null}</span>
        </button>)}</div>
      </>,
      summary: <div className="pseudo-compare-summary" aria-label="Пиксельное пересечение и различия">{activeIds.length === 2 ? <>
        <div><span>Всего · {complete ? `${sharedSceneIds.length} общих снимков` : `в фоне ${Object.keys(counts).length}/${sharedSceneIds.length}`}</span>{complete && sharedSceneIds.length ? <PixelValues counts={total} numbers={numbers} /> : null}</div>
        <div><span>Этот снимок</span>{sceneCounts ? <PixelValues counts={sceneCounts} numbers={numbers} /> : <small>{available.length < 2 ? "Нет общей пары" : "Считаем пиксели…"}</small>}</div>
      </> : <small>Для счётчиков включите ровно две разметки</small>}</div>,
      layerControls: <div className="pseudo-compare-map-layers" role="group" aria-label="Переключатели псевдоразметок на снимке">{views.map((view, index) => <button type="button" key={view.id} className="secondary icon-button" aria-label={`${hidden.has(view.id) ? "Включить" : "Выключить"} разметку ${index + 1}`} aria-pressed={!hidden.has(view.id)} title={`${index + 1}. ${view.class_name || view.training_dataset_name} · ${view.model_name}${view.checkpoint_epoch != null ? ` · эпоха ${view.checkpoint_epoch}` : ""}${!scene.resultIds.includes(view.id) ? " · нет на этом снимке" : ""}`} style={{ borderColor: colors[view.id] }} onClick={() => toggleLayer(view.id)}>{hidden.has(view.id) ? <EyeOff size={16} /> : <Eye size={16} />}<b>{index + 1}</b><i style={{ backgroundColor: colors[view.id] }} /></button>)}</div>,
      controls: <>
        <button type="button" className={`${intersection ? "primary" : "secondary"} icon-button dataset-editor-map-control`} title="Пересечения активных разметок" aria-label="Подсветить пересечения" aria-pressed={intersection} onClick={() => setIntersection((value) => !value)}><Blend size={17} /></button>
        <button type="button" className={`${differences ? "primary" : "secondary"} icon-button dataset-editor-map-control`} title="Области только одной активной разметки" aria-label="Подсветить различия" aria-pressed={differences} onClick={() => setDifferences((value) => !value)}><Split size={17} /></button>
      </>,
      legend: <>{intersection && available.length >= 2 ? <span className="test-f1-legend-item"><i style={{ backgroundColor: TEST_F1_LAYERS.tp.color }} />Пересечение {available.length > 2 ? "≥2 слоёв" : ""}</span> : null}{available.map((id) => <span key={id} className="test-f1-legend-item"><i style={{ backgroundColor: colors[id] }} />{available.length >= 2 && differences ? "Только " : "Разметка "}{views.findIndex((view) => view.id === id) + 1}</span>)}</>,
      hint: "Слои уточняются по видимой области и масштабу карты. Счётчики — точные пиксели нативной сетки TIFF без nodata; выбранный снимок имеет приоритет, остальные считаются в фоне. «Только» — область одной разметки вне другой. Для трёх и более слоёв зелёным показаны пересечения любых двух. Это сравнение прогнозов, без оценки F1.",
    }} /> : null}
    {info && scene && !activeIds.length ? <p className="info-box">Все разметки выключены. Включите нужные слои кнопками с глазом и номером на снимке.</p> : null}
    {info && scene && activeIds.length > 0 && !available.length ? <p className="info-box">Этот снимок не входит в активные разметки. Выберите другой снимок или включите его разметку.</p> : null}
    {scene && available.length > 0 && !currentLayers && !layerError ? <p role="status" className="pseudo-hint">Загружаем разметки выбранного снимка…</p> : null}
    {warnings.map((warning, index) => <p className="info-box" key={index}>{warning}</p>)}
  </section>;
}

function PixelValues({ counts, numbers }: { counts: PseudoMarkupPixelComparison; numbers: number[] }) {
  return <span className="pseudo-compare-pixels"><span title="Пересечение в пикселях">∩ {counts.intersection.toLocaleString("ru-RU")}</span><span title={`Пиксели разметки ${numbers[0]} вне разметки ${numbers[1]}`}>Только {numbers[0]}: {counts.only_first.toLocaleString("ru-RU")}</span><span title={`Пиксели разметки ${numbers[1]} вне разметки ${numbers[0]}`}>Только {numbers[1]}: {counts.only_second.toLocaleString("ru-RU")}</span><small>пикс.</small></span>;
}
