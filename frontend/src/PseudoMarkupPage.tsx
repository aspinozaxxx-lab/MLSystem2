import { ArrowLeft, Download, Eye, EyeOff, Maximize, Maximize2, Minimize2, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
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
import type { PseudoMarkupViewInfo } from "./api/types";
import { formatDateTime } from "./utils/format";
import { BAND_CHANNELS, type BandMode } from "./utils/datasetEditor";
import { pseudoClass, pseudoClasses, pseudoRasterCacheSizes, pseudoRasterScenes, pseudoRasterStyle, pseudoViewportScenes, type PseudoProperties } from "./utils/pseudoViewer";
import { rasterCache } from "./utils/rasterCache";
import { rasterBackdrop } from "./utils/rasterBackdrop";
import { rasterLoadCancelled, rasterResponseError, watchRasterLoading } from "./utils/rasterLoading";
import { useMapFullscreen } from "./utils/useMapFullscreen";
import "ol/ol.css";
import "./styles/pseudoViewer.css";

type GeoJson = { type: string; features: { properties?: PseudoProperties | null }[] };
type LoadedView = { info: PseudoMarkupViewInfo; geojson: GeoJson };

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

function PseudoMap({ info, geojson, username, onRetry }: LoadedView & { username: string; onRetry: () => void }) {
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
  const allBounds = useRef<Extent>(createEmpty());
  const hiddenClasses = useRef(new Set<string>());
  const [hidden, setHidden] = useState(new Set<string>());
  const [imagesVisible, setImagesVisible] = useState(true);
  const [markupVisible, setMarkupVisible] = useState(true);
  const [opacity, setOpacity] = useState(0.8);
  const [query, setQuery] = useState("");
  const [bandMode, setBandMode] = useState<BandMode>("RGB");
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
        style: pseudoRasterStyle("RGB", alpha, nir),
      });
    });
    const backdrop = rasterBackdrop();
    const outline = new VectorSource();
    outlineSource.current = outline;
    const contour = new VectorLayer({ source: outline, style: [
      new Style({ stroke: new Stroke({ color: "#111827", width: 6 }) }),
      new Style({ stroke: new Stroke({ color: "#fbbf24", width: 3 }) }),
    ] });
    const vector = new VectorSource({ features: new GeoJSON().readFeatures(geojson, { featureProjection: "EPSG:3857" }) });
    const markup = new VectorImageLayer({
      source: vector, opacity: 0.8,
      style: (feature) => {
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
    const resize = new ResizeObserver(() => map.updateSize());
    resize.observe(target.current);
    return () => {
      active = false;
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
      outlineSource.current = null;
      mapRef.current = null;
      nirLayer.current = null;
    };
  }, [info, geojson, username]);

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
  return <div ref={workspaceRef} className={`pseudo-workspace${fullscreen ? " fullscreen" : ""}`}>
    <header className="pseudo-viewer-heading">
      <div><h1>Просмотр псевдоразметки</h1><p>{info.source_dataset_name} · {formatDateTime(info.created_at)}</p></div>
      <a className="secondary compact-action" href={info.geojson_url}><Download size={16} /> Скачать GeoJSON</a>
    </header>
    <div className="pseudo-model"><span>Сеть: <strong>{info.model_name}</strong></span><span>Обучена на: <strong>{info.training_dataset_name}</strong></span><span>{geojson.features.length.toLocaleString("ru-RU")} объектов</span></div>
    <div className="pseudo-toolbar">
      <button type="button" className="secondary compact-action" onClick={() => fit(allBounds.current)}><Maximize size={15} /> Все снимки</button>
      {hasNir ? <label title="Как в редакторе датасета: RGB, NRG или NGB. Снимки без NIR остаются в RGB.">Каналы
        <select aria-label="Сочетание каналов" value={bandMode} onChange={(event) => {
          const mode = event.target.value as BandMode;
          setBandMode(mode);
          resetBackdrop.current();
          nirLayer.current?.setStyle(pseudoRasterStyle(mode, false, true));
        }}>{Object.keys(BAND_CHANNELS).map((mode) => <option key={mode} value={mode}>{mode}</option>)}</select>
      </label> : null}
      <label><input type="checkbox" checked={imagesVisible} onChange={(event) => { imagesEnabled.current = event.target.checked; setImagesVisible(event.target.checked); refreshRasters.current(); }} /> Снимки</label>
      <label><input type="checkbox" checked={markupVisible} onChange={(event) => { setMarkupVisible(event.target.checked); markupLayer.current?.setVisible(event.target.checked); }} /> Псевдоразметка</label>
      <label>Непрозрачность <input aria-label="Непрозрачность псевдоразметки" type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(event) => { const value = Number(event.target.value); setOpacity(value); markupLayer.current?.setOpacity(value); }} /></label>
    </div>
    {info.warnings.length > 0 ? <details className="info-box"><summary>Не все исходные снимки доступны: подробности</summary><ul>{info.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details> : null}
    {imagesVisible && visibleErrors.length > 0 ? <div className="info-box" role="alert">
      <ul>{visibleErrors.map((scene) => <li key={scene.id}><strong>{scene.name}</strong>: {rasterErrors[scene.id]}</li>)}</ul>
      <button type="button" className="secondary compact-action" onClick={onRetry}><RefreshCw size={14} /> Повторить загрузку</button>
    </div> : null}
    <div className="pseudo-viewer-body">
      <aside className="pseudo-sidebar">
        <h2>Снимки <span>{info.scenes.length}{info.expected_image_count != null && info.expected_image_count !== info.scenes.length ? ` из ${info.expected_image_count}` : ""}</span></h2>
        <p className="pseudo-scene-count">{imagesVisible ? `На экране: ${viewportSceneIds.size}` : "Снимки скрыты"}{hiddenSceneIds.size > 0 ? ` · Выключено: ${hiddenSceneIds.size}` : ""}</p>
        <input type="search" placeholder="Найти снимок" aria-label="Найти снимок" value={query} onChange={(event) => setQuery(event.target.value)} />
        <div className="pseudo-scene-list">{scenes.map((scene) => {
          const enabled = !hiddenSceneIds.has(scene.id);
          const visible = imagesVisible && enabled && viewportSceneIds.has(scene.id);
          return <div key={scene.id} className={`pseudo-scene-row${visible ? " in-view" : ""}${selectedSceneId === scene.id ? " selected" : ""}${enabled ? "" : " hidden-scene"}`}
            onMouseEnter={() => setHoverSceneId(scene.id)} onMouseLeave={() => setHoverSceneId(null)}
            onFocus={() => setHoverSceneId(scene.id)} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setHoverSceneId(null); }}>
            <button type="button" className="pseudo-scene-toggle" aria-label={`${enabled ? "Скрыть" : "Показать"} снимок ${scene.name}`}
              aria-pressed={enabled} title={enabled ? "Скрыть снимок" : "Показать снимок"} onClick={() => toggleScene(scene.id)}>
              {enabled ? <Eye size={15} /> : <EyeOff size={15} />}
            </button>
            <button type="button" className="pseudo-scene-name" aria-label={`${scene.name}${visible ? " · На экране" : ""}`}
              aria-pressed={selectedSceneId === scene.id} title={`${visible ? "На экране. " : ""}Приблизить и выделить контур ${scene.name}`}
              onClick={() => { setSelectedSceneId(scene.id); setOutlineAttempt((value) => value + 1); fit(scene.bounds); }}><span>{scene.name}</span></button>
          </div>;
        })}</div>
        {!info.scenes.length ? <p>Исходные снимки недоступны. Слой псевдоразметки можно просматривать отдельно.</p> : null}
      </aside>
      <div className="pseudo-map-area">
        <div className="pseudo-map" ref={target} aria-label="Мозаика снимков с псевдоразметкой" tabIndex={0} />
        <div className="dataset-editor-map-controls">
          <button
            className={`${fullscreen ? "primary" : "secondary"} icon-button dataset-editor-map-control`}
            type="button"
            aria-label={fullscreen ? "Выйти из полноэкранного режима" : "Открыть просмотр на весь экран"}
            aria-pressed={fullscreen}
            title={fullscreen ? "Выйти из полноэкранного режима (Esc)" : "Открыть просмотр на весь экран"}
            onClick={() => void toggleFullscreen()}
          >{fullscreen ? <Minimize2 size={17} /> : <Maximize2 size={17} />}</button>
        </div>
        <div className="pseudo-map-legend">{classes.map((item) => <button type="button" key={item.key} aria-pressed={!hidden.has(item.key)} onClick={() => toggleClass(item.key)}><span style={{ backgroundColor: item.color }} />{item.name} <small>{item.count.toLocaleString("ru-RU")}</small></button>)}</div>
        {!geojson.features.length ? <div className="pseudo-empty">На этих снимках сеть не нашла объектов</div> : null}
        {outlineError ? <div className="pseudo-outline-error" role="status">{outlineError}</div> : null}
      </div>
    </div>
    <p className="pseudo-hint">Колесо — масштаб, перетаскивание — перемещение. Снимки на экране подсвечены в списке. Глаз скрывает и показывает снимок; наведение на строку выделяет его контур без nodata. Нажмите на название, чтобы приблизить снимок и оставить контур выделенным.{hasNir ? " NRG и NGB используют NIR; снимки без него остаются в RGB." : ""} Просмотр не изменяет разметку датасета.</p>
  </div>;
}
