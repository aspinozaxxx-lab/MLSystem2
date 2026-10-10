import { ArrowLeft, Blend, ChevronDown, ClipboardCheck, Download, Eye, EyeOff, Image, ImageOff, Layers, Maximize2, Minimize2, RefreshCw, Scan, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import OLMap from "ol/Map";
import View from "ol/View";
import { defaults as defaultControls } from "ol/control/defaults";
import { defaults as defaultInteractions } from "ol/interaction/defaults";
import GeoJSON from "ol/format/GeoJSON";
import { createEmpty, extend, intersects, isEmpty, type Extent } from "ol/extent";
import VectorImageLayer from "ol/layer/VectorImage";
import VectorLayer from "ol/layer/Vector";
import type Feature from "ol/Feature";
import WebGLTileLayer from "ol/layer/WebGLTile";
import GeoTIFF from "ol/source/GeoTIFF";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style } from "ol/style";
import { apiJson } from "./api/client";
import type { PseudoMarkupComparisonViewport, PseudoMarkupViewInfo } from "./api/types";
import { formatDateTime } from "./utils/format";
import { BAND_CHANNELS, type BandMode } from "./utils/datasetEditor";
import { pseudoClass, pseudoClasses, pseudoRasterCacheSizes, pseudoRasterScenes, pseudoRasterStyle, pseudoViewportScenes, type PseudoProperties } from "./utils/pseudoViewer";
import { rasterCache } from "./utils/rasterCache";
import { rasterBackdrop } from "./utils/rasterBackdrop";
import { rasterLoadCancelled, rasterResponseError, watchRasterLoading } from "./utils/rasterLoading";
import { useMapFullscreen } from "./utils/useMapFullscreen";
import { useCompactLayout } from "./utils/useCompactLayout";
import { comparisonLayerLabel, comparisonLayerStyle, TEST_F1_LAYERS } from "./utils/testF1Viewer";
import "ol/ol.css";
import "./styles/pseudoViewer.css";

export type ViewerGeoJson = { type: string; features: { properties?: PseudoProperties | null }[] };
type GeoJson = ViewerGeoJson;
type LoadedView = { info: PseudoMarkupViewInfo; geojson: GeoJson };
type Comparison = {
  sidebar: ReactNode; summary: ReactNode; title?: string; subtitle?: string; controls?: ReactNode;
  legend?: ReactNode; hint?: string; layerControls?: ReactNode;
  metric?: "pixel" | "objects";
  featureStyle?: (properties: PseudoProperties) => { color: string; fill: string; width: number } | null;
  onViewport?: (viewport: PseudoMarkupComparisonViewport) => void;
};

export function PseudoMarkupPage({ resultId, username }: { resultId: string; username: string }) {
  const [loaded, setLoaded] = useState<LoadedView | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    setLoaded(null);
    setError("");
    void apiJson<PseudoMarkupViewInfo>(`/results/pseudo-markup/${encodeURIComponent(resultId)}/view`)
      .then(async (info) => {
        const geojson = await apiJson<GeoJson>(info.geojson_url);
        if (geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) throw new Error("Файл псевдоразметки повреждён.");
        if (active) setLoaded({ info, geojson });
      })
      .catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : "Не удалось открыть псевдоразметку."); });
    return () => { active = false; };
  }, [resultId, attempt]);
  return <section className="panel pseudo-viewer">
    <a className="pseudo-back" href="#/results"><ArrowLeft size={16} /> К результатам</a>
    {error ? <div className="info-box" role="alert">{error} <button type="button" className="secondary compact-action" onClick={() => setAttempt((value) => value + 1)}><RefreshCw size={14} /> Повторить</button></div>
      : loaded ? <PseudoMap key={`${username}:${resultId}`} {...loaded} username={username} onRetry={() => setAttempt((value) => value + 1)} /> : <p role="status">Загружаем снимки и псевдоразметку…</p>}
  </section>;
}

export function PseudoMap({ info, geojson, username, onRetry, comparison }: LoadedView & {
  username: string; onRetry: () => void;
  comparison?: Comparison;
}) {
  const compact = useCompactLayout();
  const [scenesExpanded, setScenesExpanded] = useState(false);
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const target = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<OLMap | null>(null);
  const { fullscreen, toggleFullscreen } = useMapFullscreen(workspaceRef, mapRef);
  const resetBackdrop = useRef<() => void>(() => {});
  const refreshRasters = useRef<() => void>(() => {});
  const hiddenScenes = useRef(new Set<string>());
  const imagesEnabled = useRef(true);
  const outlineSource = useRef<VectorSource | null>(null);
  const footprints = useRef(new Map<string, Feature[]>());
  const nirLayer = useRef<WebGLTileLayer | null>(null);
  const markupLayer = useRef<VectorImageLayer | null>(null);
  const markupSource = useRef<VectorSource | null>(null);
  const comparisonLayers = useRef({ reference: true, predicted: true });
  const customStyle = useRef(comparison?.featureStyle);
  customStyle.current = comparison?.featureStyle;
  const viewportChanged = useRef(comparison?.onViewport);
  viewportChanged.current = comparison?.onViewport;
  const [referenceVisible, setReferenceVisible] = useState(true);
  const allBounds = useRef<Extent>(createEmpty());
  const hiddenClasses = useRef(new Set<string>());
  const [hidden, setHidden] = useState(new Set<string>());
  const [imagesVisible, setImagesVisible] = useState(true);
  const [markupVisible, setMarkupVisible] = useState(true);
  const [query, setQuery] = useState("");
  const [bandMode, setBandMode] = useState<BandMode>("RGB");
  const selectedBandMode = useRef<BandMode>("RGB");
  const [bandMenuOpen, setBandMenuOpen] = useState(false);
  const [rasterErrors, setRasterErrors] = useState<Record<string, string>>({});
  const [hiddenSceneIds, setHiddenSceneIds] = useState(new Set<string>());
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [hoverSceneId, setHoverSceneId] = useState<string | null>(null);
  const [outlineError, setOutlineError] = useState("");
  const [outlineAttempt, setOutlineAttempt] = useState(0);
  const [viewportSceneIds, setViewportSceneIds] = useState(new Set<string>());
  const classes = useMemo(() => pseudoClasses(geojson.features), [geojson]);
  const scenes = info.scenes.filter((scene) => scene.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const hasNir = info.scenes.some((scene) => scene.has_nir && !scene.has_alpha);

  useEffect(() => {
    if (!target.current) return;
    let active = true;
    const requests = new AbortController();
    const sources = new Map<string, GeoTIFF>();
    const sourceCleanup = new Map<string, () => void>();
    const sourceTimes = new Map<string, number>();
    const pinned = new Set<string>();
    const styles = new Map<string, Style>();
    const reportRasterError = (id: string, message: string | null) => {
      if (active) setRasterErrors((old) => {
        if ((old[id] ?? null) === message) return old;
        const next = { ...old };
        if (message) next[id] = message; else delete next[id];
        return next;
      });
    };
    // RGB, RGBA и RGB+NIR требуют разных правил цвета и прозрачности.
    const layouts = [{ alpha: false, nir: false }, { alpha: true, nir: false }, { alpha: false, nir: true }];
    const layoutScenes = layouts.map(({ alpha, nir }) => info.scenes.filter((scene) => Boolean(scene.has_alpha) === alpha
      && Boolean(scene.has_nir && !scene.has_alpha) === nir));
    const cacheSizes = pseudoRasterCacheSizes(layoutScenes.map((scenes) => scenes.length));
    const rasters = layouts.map(({ alpha, nir }, index) => {
      const scenes = layoutScenes[index];
      if (scenes[0]) pinned.add(scenes[0].id);
      return new WebGLTileLayer({
        className: "ol-layer pseudo-raster",
        cacheSize: cacheSizes[index],
        preload: 2,
        sources: (extent, resolution) => {
          // OpenLayers запрашивает весь мир для определения схемы каналов; достаточно первого снимка.
          const selected = pseudoRasterScenes(scenes, extent, resolution, hiddenScenes.current);
          return selected.map((scene) => {
            let source = sources.get(scene.id);
            if (!source) {
              const controller = new AbortController();
              let httpError: string | null = null;
              source = new GeoTIFF({
                sources: [{ url: scene.raster_url, bands: alpha || nir ? [1, 2, 3, 4] : [1, 2, 3], nodata: scene.nodata ?? NaN,
                  loader: async (url, headers, signal) => {
                    const response = await rasterCache.load(username, url, headers,
                      AbortSignal.any([controller.signal, requests.signal, ...(signal ? [signal] : [])]));
                    if (!response.ok) {
                      httpError = await rasterResponseError(response);
                      throw new Error(httpError);
                    }
                    return response;
                  },
                }],
                sourceOptions: { credentials: "same-origin", maxRanges: 1, cacheSize: 32 },
                normalize: true, interpolate: false, transition: 0,
              });
              sources.set(scene.id, source);
              const current = source;
              const isActive = () => active && !controller.signal.aborted && sources.get(scene.id) === current;
              const stopWatching = watchRasterLoading(source, {
                active: isActive,
                describe: () => httpError ?? (current.getState() === "error"
                  ? "Не удалось прочитать заголовок TIFF или его привязку."
                  : "Не удалось прочитать часть изображения после повторных попыток."),
                report: (message) => reportRasterError(scene.id, message),
              });
              sourceCleanup.set(scene.id, () => { stopWatching(); controller.abort(); current.clear(); current.dispose(); });
              void source.getView().catch((error: unknown) => {
                if (isActive() && !rasterLoadCancelled(error)) reportRasterError(scene.id,
                  httpError ?? "Не удалось прочитать заголовок TIFF или его привязку.");
              });
            }
            sourceTimes.set(scene.id, Date.now());
            return source;
          });
        },
        style: pseudoRasterStyle(nir ? selectedBandMode.current : "RGB", alpha, nir),
      });
    });
    const backdrop = rasterBackdrop();
    const outline = new VectorSource();
    outlineSource.current = outline;
    const contour = new VectorLayer({ source: outline, style: [
      new Style({ stroke: new Stroke({ color: "#111827", width: 6 }) }),
      new Style({ stroke: new Stroke({ color: "#fbbf24", width: 3 }) }),
    ] });
    const vector = new VectorSource({ features: new GeoJSON().readFeatures(geojson, {
      dataProjection: comparison ? "EPSG:3857" : "EPSG:4326", featureProjection: "EPSG:3857",
    }) });
    markupSource.current = vector;
    const markup = new VectorImageLayer({
      source: vector, opacity: comparison ? 1 : 0.8,
      style: (feature) => {
        if (comparison) {
          const appearance = customStyle.current ? customStyle.current(feature.getProperties())
            : comparisonLayerStyle(String(feature.get("test_f1_layer")), comparisonLayers.current.reference, comparisonLayers.current.predicted);
          if (!appearance) return undefined;
          const key = `${appearance.color}:${appearance.fill}:${appearance.width}`;
          let style = styles.get(key);
          if (!style) {
            style = new Style({ stroke: new Stroke({ color: appearance.color, width: appearance.width }),
              fill: new Fill({ color: appearance.fill }) });
            styles.set(key, style);
          }
          return style;
        }
        const item = pseudoClass(feature.getProperties());
        if (hiddenClasses.current.has(item.key)) return undefined;
        let style = styles.get(item.color);
        if (!style) {
          style = new Style({ stroke: new Stroke({ color: item.color, width: 1.5 }), fill: new Fill({ color: `${item.color}30` }) });
          styles.set(item.color, style);
        }
        return style;
      },
    });
    const view = new View({ projection: "EPSG:3857", center: [0, 0], zoom: 2, maxZoom: 26 });
    const map = new OLMap({
      target: target.current, layers: [backdrop.layer, ...rasters, markup, contour], view,
      controls: defaultControls({ zoom: false }),
      interactions: defaultInteractions({ onFocusOnly: false }),
    });
    mapRef.current = map;
    resetBackdrop.current = backdrop.reset;
    refreshRasters.current = () => {
      backdrop.reset();
      backdrop.layer.setVisible(imagesEnabled.current);
      rasters.forEach((layer, index) => {
        layer.setVisible(imagesEnabled.current && layoutScenes[index].some((scene) => !hiddenScenes.current.has(scene.id)));
        layer.changed();
      });
      map.render();
    };
    const detachBackdrop = backdrop.attach(map);
    nirLayer.current = rasters[2];
    markupLayer.current = markup;
    markup.setVisible(comparison ? true : markupVisible);
    refreshRasters.current();
    const bounds = createEmpty();
    info.scenes.forEach((scene) => extend(bounds, scene.bounds));
    const vectorBounds = vector.getExtent();
    if (isEmpty(bounds) && vectorBounds) extend(bounds, vectorBounds);
    allBounds.current = bounds;
    if (!isEmpty(bounds)) view.fit(bounds, { padding: [32, 32, 32, 32], maxZoom: 20 });
    let previousVisible: Set<string> | null = null;
    map.on("postrender", ({ frameState }) => {
      if (!active || !frameState) return;
      const visible = new Set(pseudoViewportScenes(info.scenes, frameState.viewState, frameState.size)
        .filter((scene) => !hiddenScenes.current.has(scene.id)).map((scene) => scene.id));
      // Следим за текущим кадром, включая анимацию и полный экран; список не перерисовывается без смены состава.
      if (previousVisible?.size === visible.size && [...visible].every((id) => previousVisible!.has(id))) return;
      previousVisible = visible;
      setViewportSceneIds(visible);
    });
    map.on("moveend", () => {
      const extent = view.calculateExtent(map.getSize());
      // Сохраняем недавно просмотренные источники при небольших перемещениях.
      // Видимые снимки и опорные схемы каналов не вытесняются.
      const offscreen = info.scenes.filter((scene) => sources.has(scene.id) && !pinned.has(scene.id) && !intersects(extent, scene.bounds))
        .sort((a, b) => (sourceTimes.get(a.id) ?? 0) - (sourceTimes.get(b.id) ?? 0));
      for (const scene of offscreen) {
        if (sources.size > Math.max(64, pinned.size)) {
          sourceCleanup.get(scene.id)?.();
          sourceCleanup.delete(scene.id);
          sources.delete(scene.id);
          sourceTimes.delete(scene.id);
          reportRasterError(scene.id, null);
        }
      }
    });
    let viewportTimer: ReturnType<typeof setTimeout> | undefined;
    const reportViewport = () => {
      clearTimeout(viewportTimer);
      viewportTimer = setTimeout(() => {
        if (!active || !viewportChanged.current) return;
        const size = map.getSize();
        if (!size || size[0] <= 0 || size[1] <= 0) return;
        const bounds = view.calculateExtent(size).map((value) => Math.round(value * 1000) / 1000) as [number, number, number, number];
        viewportChanged.current({ bounds, width: Math.min(1536, Math.ceil(size[0])), height: Math.min(1536, Math.ceil(size[1])) });
      }, 60);
    };
    map.on("moveend", reportViewport);
    const resize = new ResizeObserver(() => { map.updateSize(); reportViewport(); });
    resize.observe(target.current);
    return () => {
      active = false;
      clearTimeout(viewportTimer);
      requests.abort();
      resize.disconnect();
      detachBackdrop();
      map.setTarget(undefined);
      map.dispose();
      rasters.forEach((layer) => layer.dispose());
      sourceCleanup.forEach((cleanup) => cleanup());
      sourceCleanup.clear();
      sources.clear();
      contour.dispose();
      markup.dispose();
      vector.clear();
      markupSource.current = null;
      outlineSource.current = null;
      mapRef.current = null;
      nirLayer.current = null;
    };
  }, [info, username]);

  useEffect(() => {
    const source = markupSource.current;
    if (!source) return;
    source.clear();
    source.addFeatures(new GeoJSON().readFeatures(geojson, {
      dataProjection: comparison ? "EPSG:3857" : "EPSG:4326", featureProjection: "EPSG:3857",
    }));
    const bounds = source.getExtent();
    if (!info.scenes.length && bounds && !isEmpty(bounds)) {
      allBounds.current = bounds;
      mapRef.current?.getView().fit(allBounds.current, { padding: [32, 32, 32, 32], maxZoom: 20 });
    }
  }, [geojson, info]);

  useEffect(() => { markupLayer.current?.changed(); }, [comparison?.featureStyle]);

  const highlightId = hoverSceneId ?? selectedSceneId;
  useEffect(() => {
    const scene = info.scenes.find((item) => item.id === highlightId);
    const source = outlineSource.current;
    source?.clear();
    setOutlineError("");
    if (!scene || !source) return;
    const cached = footprints.current.get(scene.id);
    if (cached) { source.addFeatures(cached); return; }
    const controller = new AbortController();
    // При быстром движении по списку не рассчитываем контуры всех промежуточных строк.
    const timer = setTimeout(() => {
      void fetch(scene.footprint_url, { credentials: "same-origin", signal: controller.signal })
        .then(async (response) => {
          if (!response.ok) throw new Error(await rasterResponseError(response));
          const features = new GeoJSON().readFeatures(await response.json(), { dataProjection: "EPSG:3857", featureProjection: "EPSG:3857" });
          if (controller.signal.aborted) return;
          footprints.current.set(scene.id, features);
          if (footprints.current.size > 32) footprints.current.delete(footprints.current.keys().next().value!);
          source.addFeatures(features);
        }).catch((error: unknown) => {
          if (!controller.signal.aborted) setOutlineError(`Контур ${scene.name} не загрузился. ${error instanceof Error ? error.message : "Повторите выбор снимка."}`);
        });
    }, 120);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [highlightId, info, outlineAttempt]);

  const fit = (bounds: Extent) => {
    if (!isEmpty(bounds)) mapRef.current?.getView().fit(bounds, { padding: [32, 32, 32, 32], maxZoom: 22, duration: 250 });
  };
  const toggleClass = (key: string) => {
    const next = new Set(hidden);
    if (next.has(key)) next.delete(key); else next.add(key);
    hiddenClasses.current = next;
    setHidden(next);
    markupLayer.current?.changed();
  };
  const toggleScene = (id: string) => {
    const next = new Set(hiddenScenes.current);
    if (next.has(id)) next.delete(id); else next.add(id);
    hiddenScenes.current = next;
    setHiddenSceneIds(next);
    refreshRasters.current();
  };
  const visibleErrors = info.scenes.filter((scene) => rasterErrors[scene.id] && !hiddenSceneIds.has(scene.id));
  return <div ref={workspaceRef} className={`pseudo-workspace${comparison ? " test-f1-workspace" : ""}${fullscreen ? " fullscreen" : ""}`}>
    <header className="pseudo-viewer-heading">
      <div className="pseudo-viewer-title"><h1>{comparison?.title ?? (comparison ? "Тестовый F1" : "Просмотр псевдоразметки")}</h1><p title={comparison?.subtitle ?? `${info.source_dataset_name} · ${formatDateTime(info.created_at)}`}>{comparison?.subtitle ?? `${info.source_dataset_name} · ${formatDateTime(info.created_at)}`}</p>
        {!comparison?.featureStyle ? <div className="pseudo-model" title={`Сеть: ${info.model_name}. Обучена на: ${info.training_dataset_name}`}><span>Сеть: <strong>{info.model_name}</strong></span><span>Обучена на: <strong>{info.training_dataset_name}</strong></span>{!comparison && info.checkpoint_epoch != null ? <span>Лучшие веса · эпоха <strong>{info.checkpoint_epoch}</strong></span> : null}{!comparison ? <span>{geojson.features.length.toLocaleString("ru-RU")} объектов</span> : null}</div> : null}
      </div>
      {comparison?.summary}
      {!comparison ? <a className="secondary compact-action pseudo-download" href={info.geojson_url} aria-label="Скачать GeoJSON" title="Скачать GeoJSON"><Download size={16} /><span>Скачать GeoJSON</span></a> : null}
    </header>
    {info.warnings.length > 0 ? <details className="info-box"><summary>Не все исходные снимки доступны: подробности</summary><ul>{info.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details> : null}
    {imagesVisible && visibleErrors.length > 0 ? <div className="info-box" role="alert">
      <ul>{visibleErrors.map((scene) => <li key={scene.id}><strong>{scene.name}</strong>: {rasterErrors[scene.id]}</li>)}</ul>
      <button type="button" className="secondary compact-action" onClick={onRetry}><RefreshCw size={14} /> Повторить загрузку</button>
    </div> : null}
    <div className="pseudo-viewer-body">
      {comparison ? <aside className="test-f1-sidebar">{comparison.sidebar}</aside> : <aside className={`pseudo-sidebar${scenesExpanded ? "" : " collapsed"}`}>
        <div className="pseudo-sidebar-heading">
          <h2>Снимки <span>{info.scenes.length}{info.expected_image_count != null && info.expected_image_count !== info.scenes.length ? ` из ${info.expected_image_count}` : ""}</span></h2>
          <span className="mobile-only muted">На экране: {imagesVisible ? viewportSceneIds.size : 0}</span>
          <button className="secondary icon-button mobile-only" type="button" aria-label="Список снимков" aria-expanded={scenesExpanded} aria-controls="pseudo-scene-content" onClick={() => setScenesExpanded((value) => !value)}><ChevronDown size={16} style={{ transform: scenesExpanded ? "rotate(180deg)" : undefined }} /></button>
        </div>
        <div className="pseudo-sidebar-content" id="pseudo-scene-content">
        <p className="pseudo-scene-count">{imagesVisible ? `На экране: ${viewportSceneIds.size}` : "Снимки скрыты"}{hiddenSceneIds.size > 0 ? ` · Выключено: ${hiddenSceneIds.size}` : ""}</p>
        <input type="search" placeholder="Найти снимок" aria-label="Найти снимок" value={query} onChange={(event) => setQuery(event.target.value)} />
        <div className="pseudo-scene-list">{scenes.map((scene) => {
          const enabled = !hiddenSceneIds.has(scene.id);
          const visible = imagesVisible && enabled && viewportSceneIds.has(scene.id);
          const selected = selectedSceneId === scene.id;
          return <div key={scene.id} className={`pseudo-scene-row${visible ? " in-view" : ""}${selected ? " selected" : ""}${enabled ? "" : " hidden-scene"}`}
            onMouseEnter={() => setHoverSceneId(scene.id)} onMouseLeave={() => setHoverSceneId(null)}
            onFocus={() => setHoverSceneId(scene.id)} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setHoverSceneId(null); }}>
            <button type="button" className="pseudo-scene-toggle" aria-label={`${enabled ? "Скрыть" : "Показать"} снимок ${scene.name}`}
              aria-pressed={enabled} title={enabled ? "Скрыть снимок" : "Показать снимок"} onClick={() => toggleScene(scene.id)}>
              {enabled ? <Eye size={15} /> : <EyeOff size={15} />}
            </button>
            <button type="button" className="pseudo-scene-name" aria-label={`${scene.name}${visible ? " · На экране" : ""}`}
              aria-pressed={selected} title={`${visible ? "На экране. " : ""}${selected ? "Снять выделение контура" : "Приблизить и выделить контур"} ${scene.name}`}
              onClick={() => {
                if (selected) {
                  setSelectedSceneId(null);
                  setHoverSceneId(null);
                  return;
                }
                setSelectedSceneId(scene.id);
                setOutlineAttempt((value) => value + 1);
                fit(scene.bounds);
                if (compact) setScenesExpanded(false);
              }}><span>{scene.name}</span></button>
          </div>;
        })}</div>
        {!info.scenes.length ? <p>Исходные снимки недоступны. Слой псевдоразметки можно просматривать отдельно.</p> : null}
        </div>
      </aside>}
      <div className="pseudo-map-area">
        <div className="pseudo-map" ref={target} aria-label={comparison?.title ?? (comparison ? "Тестовый снимок с эталоном, прогнозом и областями TP, FP, FN" : "Мозаика снимков с псевдоразметкой")} tabIndex={0} />
        {comparison?.layerControls}
        <div className="dataset-editor-map-controls">
          <button className="secondary icon-button dataset-editor-map-control" type="button" aria-label={comparison ? "Весь снимок" : "Все снимки"} title={comparison ? "Показать весь снимок" : "Показать все снимки"} onClick={() => fit(allBounds.current)}><Scan size={17} /></button>
          <button className={`${imagesVisible ? "primary" : "secondary"} icon-button dataset-editor-map-control`} type="button" aria-label="Снимки" aria-pressed={imagesVisible} title={imagesVisible ? "Скрыть снимки" : "Показать снимки"} onClick={() => { imagesEnabled.current = !imagesVisible; setImagesVisible(!imagesVisible); refreshRasters.current(); }}>{imagesVisible ? <Image size={17} /> : <ImageOff size={17} />}</button>
          {comparison?.featureStyle ? comparison.controls : <>
            {comparison ? <button className={`${referenceVisible ? "primary" : "secondary"} icon-button dataset-editor-map-control`} type="button" aria-label="Эталон" aria-pressed={referenceVisible} title={referenceVisible ? "Скрыть тестовую разметку" : "Показать тестовую разметку"} onClick={() => { setReferenceVisible(!referenceVisible); comparisonLayers.current.reference = !referenceVisible; markupLayer.current?.changed(); }}><ClipboardCheck size={17} /></button> : null}
            <button className={`${markupVisible ? "primary" : "secondary"} icon-button dataset-editor-map-control`} type="button" aria-label={comparison ? "Прогноз" : "Псевдоразметка"} aria-pressed={markupVisible} title={`${markupVisible ? "Скрыть" : "Показать"} ${comparison ? "предсказанную разметку" : "псевдоразметку"}`} onClick={() => { setMarkupVisible(!markupVisible); if (comparison) { comparisonLayers.current.predicted = !markupVisible; markupLayer.current?.changed(); } else markupLayer.current?.setVisible(!markupVisible); }}>{comparison ? <Sparkles size={17} /> : <Layers size={17} />}</button>
          </>}
          {hasNir ? <div className={`dataset-editor-band-picker${bandMenuOpen ? " open" : ""}`} onKeyDown={(event) => { if (event.key === "Escape") setBandMenuOpen(false); }} onBlur={(event) => { if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) setBandMenuOpen(false); }}>
            <button className="secondary icon-button dataset-editor-map-control" type="button" aria-label={`Сочетание каналов ${bandMode}`} aria-haspopup="menu" aria-expanded={bandMenuOpen} title={`Сочетание каналов снимка: ${bandMode}. Открыть варианты RGB, NRG и NGB`} onClick={() => setBandMenuOpen((open) => !open)}><Blend size={17} /></button>
            <div className="dataset-editor-band-menu" role="menu">{Object.keys(BAND_CHANNELS).map((mode) => <button className={bandMode === mode ? "active" : ""} type="button" role="menuitemradio" aria-checked={bandMode === mode} key={mode} onClick={() => { const selected = mode as BandMode; selectedBandMode.current = selected; setBandMode(selected); setBandMenuOpen(false); resetBackdrop.current(); nirLayer.current?.setStyle(pseudoRasterStyle(selected, false, true)); }}>{mode}</button>)}</div>
          </div> : null}
          <button
            className={`${fullscreen ? "primary" : "secondary"} icon-button dataset-editor-map-control`}
            type="button"
            aria-label={fullscreen ? "Выйти из полноэкранного режима" : "Открыть просмотр на весь экран"}
            aria-pressed={fullscreen}
            title={fullscreen ? "Выйти из полноэкранного режима (Esc)" : "Открыть просмотр на весь экран"}
            onClick={() => void toggleFullscreen()}
          >{fullscreen ? <Minimize2 size={17} /> : <Maximize2 size={17} />}</button>
        </div>
        <div className="pseudo-map-legend">{comparison?.featureStyle ? comparison.legend : comparison
          ? Object.entries(TEST_F1_LAYERS).filter(([layer]) => comparisonLayerStyle(layer, referenceVisible, markupVisible)).map(([layer, item]) => <span className="test-f1-legend-item" key={layer}><i style={{ backgroundColor: item.color }} />{comparisonLayerLabel(layer, comparison.metric)}</span>)
          : classes.map((item) => <button type="button" key={item.key} aria-pressed={!hidden.has(item.key)} onClick={() => toggleClass(item.key)}><span style={{ backgroundColor: item.color }} />{item.name} <small>{item.count.toLocaleString("ru-RU")}</small></button>)}</div>
        {!geojson.features.length && !comparison ? <div className="pseudo-empty">На этих снимках сеть не нашла объектов</div> : null}
        {outlineError ? <div className="pseudo-outline-error" role="status">{outlineError}</div> : null}
      </div>
    </div>
    <p className="pseudo-hint">{comparison?.hint ?? (comparison ? "Два включённых слоя показывают пиксельное сравнение: зелёный — совпадение, красный — лишнее, жёлтый — пропуск. Эталон и прогноз можно отключать независимо. " : `${compact ? "Перемещение — одним пальцем, масштаб — двумя. Список снимков раскрывается стрелкой. " : "Колесо — масштаб, перетаскивание — перемещение. "}Снимки на экране подсвечены в списке. Глаз скрывает и показывает снимок; наведение на строку выделяет его контур без nodata. Нажмите на название, чтобы приблизить снимок и оставить контур выделенным; нажмите повторно, чтобы убрать рамку.`)}{hasNir ? " NRG и NGB используют NIR; снимки без него остаются в RGB." : ""} Просмотр не изменяет разметку датасета.</p>
  </div>;
}
