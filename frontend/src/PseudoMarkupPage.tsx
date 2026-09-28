import { ArrowLeft, Download, Maximize, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import OLMap from "ol/Map";
import View from "ol/View";
import { defaults as defaultControls } from "ol/control/defaults";
import { defaults as defaultInteractions } from "ol/interaction/defaults";
import GeoJSON from "ol/format/GeoJSON";
import { createEmpty, extend, intersects, isEmpty, type Extent } from "ol/extent";
import VectorImageLayer from "ol/layer/VectorImage";
import WebGLTileLayer from "ol/layer/WebGLTile";
import GeoTIFF from "ol/source/GeoTIFF";
import type BaseLayer from "ol/layer/Base";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style } from "ol/style";
import { apiJson } from "./api/client";
import type { PseudoMarkupViewInfo } from "./api/types";
import { formatDateTime } from "./utils/format";
import { BAND_CHANNELS, type BandMode } from "./utils/datasetEditor";
import { pseudoClass, pseudoClasses, pseudoRasterStyle, type PseudoProperties } from "./utils/pseudoViewer";
import { rasterCache } from "./utils/rasterCache";
import { rasterBackdrop } from "./utils/rasterBackdrop";
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
      : loaded ? <PseudoMap key={`${username}:${resultId}`} {...loaded} username={username} /> : <p role="status">Загружаем снимки и псевдоразметку…</p>}
  </section>;
}

function PseudoMap({ info, geojson, username }: LoadedView & { username: string }) {
  const target = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<OLMap | null>(null);
  const rasterLayers = useRef<BaseLayer[]>([]);
  const resetBackdrop = useRef<() => void>(() => {});
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
  const [rasterErrors, setRasterErrors] = useState<string[]>([]);
  const classes = useMemo(() => pseudoClasses(geojson.features), [geojson]);
  const scenes = info.scenes.filter((scene) => scene.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const hasNir = info.scenes.some((scene) => scene.has_nir && !scene.has_alpha);

  useEffect(() => {
    if (!target.current) return;
    let active = true;
    const requests = new AbortController();
    const sources = new Map<string, GeoTIFF>();
    const sourceTimes = new Map<string, number>();
    const pinned = new Set<string>();
    const styles = new Map<string, Style>();
    const reportRasterError = (name: string) => {
      if (active) setRasterErrors((old) => old.includes(name) ? old : [...old, name]);
    };
    // RGB, RGBA и RGB+NIR требуют разных правил цвета и прозрачности.
    const layouts = [{ alpha: false, nir: false }, { alpha: true, nir: false }, { alpha: false, nir: true }];
    const activeLayouts = new Set(info.scenes.map((scene) => scene.has_alpha ? "alpha" : scene.has_nir ? "nir" : "rgb")).size;
    const rasters = layouts.map(({ alpha, nir }) => {
      const scenes = info.scenes.filter((scene) => Boolean(scene.has_alpha) === alpha
        && Boolean(scene.has_nir && !scene.has_alpha) === nir);
      if (scenes[0]) pinned.add(scenes[0].id);
      return new WebGLTileLayer({
        className: "ol-layer pseudo-raster",
        cacheSize: scenes.length ? 32 + Math.floor((256 - 32 * activeLayouts) * scenes.length / info.scenes.length) : 1,
        preload: 2,
        sources: (extent, resolution) => {
          // OpenLayers запрашивает весь мир для определения схемы каналов; достаточно первого снимка.
          const selected = resolution === Number.MAX_VALUE ? scenes.slice(0, 1) : scenes.filter((scene) => intersects(extent, scene.bounds));
          return selected.map((scene) => {
            let source = sources.get(scene.id);
            if (!source) {
              source = new GeoTIFF({
                sources: [{ url: scene.raster_url, bands: alpha || nir ? [1, 2, 3, 4] : [1, 2, 3], nodata: scene.nodata ?? NaN,
                  loader: (url, headers, signal) => rasterCache.load(username, url, headers,
                    signal ? AbortSignal.any([signal, requests.signal]) : requests.signal),
                }],
                sourceOptions: { credentials: "same-origin", maxRanges: 1, cacheSize: 32 },
                normalize: true, interpolate: false, transition: 0,
              });
              sources.set(scene.id, source);
              source.on("tileloaderror", () => reportRasterError(scene.name));
              void source.getView().catch(() => reportRasterError(scene.name));
            }
            sourceTimes.set(scene.id, Date.now());
            return source;
          });
        },
        style: pseudoRasterStyle("RGB", alpha, nir),
      });
    });
    const backdrop = rasterBackdrop();
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
      target: target.current, layers: [backdrop.layer, ...rasters, markup], view,
      controls: defaultControls({ zoom: false }),
      interactions: defaultInteractions({ onFocusOnly: false }),
    });
    mapRef.current = map;
    rasterLayers.current = [backdrop.layer, ...rasters];
    resetBackdrop.current = backdrop.reset;
    const detachBackdrop = backdrop.attach(map);
    nirLayer.current = rasters[2];
    markupLayer.current = markup;
    const bounds = createEmpty();
    info.scenes.forEach((scene) => extend(bounds, scene.bounds));
    const vectorBounds = vector.getExtent();
    if (isEmpty(bounds) && vectorBounds) extend(bounds, vectorBounds);
    allBounds.current = bounds;
    if (!isEmpty(bounds)) view.fit(bounds, { padding: [32, 32, 32, 32], maxZoom: 20 });
    map.on("moveend", () => {
      const extent = view.calculateExtent(map.getSize());
      // Сохраняем недавно просмотренные источники при небольших перемещениях.
      // Видимые снимки и опорные схемы каналов не вытесняются.
      const offscreen = info.scenes.filter((scene) => sources.has(scene.id) && !pinned.has(scene.id) && !intersects(extent, scene.bounds))
        .sort((a, b) => (sourceTimes.get(a.id) ?? 0) - (sourceTimes.get(b.id) ?? 0));
      for (const scene of offscreen) {
        if (sources.size > Math.max(64, pinned.size)) {
          sources.get(scene.id)?.clear();
          sources.get(scene.id)?.dispose();
          sources.delete(scene.id);
          sourceTimes.delete(scene.id);
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
      sources.forEach((source) => { source.clear(); source.dispose(); });
      sources.clear();
      mapRef.current = null;
      nirLayer.current = null;
    };
  }, [info, geojson, username]);

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
  return <>
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
      <label><input type="checkbox" checked={imagesVisible} onChange={(event) => { setImagesVisible(event.target.checked); rasterLayers.current.forEach((layer) => layer.setVisible(event.target.checked)); }} /> Снимки</label>
      <label><input type="checkbox" checked={markupVisible} onChange={(event) => { setMarkupVisible(event.target.checked); markupLayer.current?.setVisible(event.target.checked); }} /> Псевдоразметка</label>
      <label>Непрозрачность <input aria-label="Непрозрачность псевдоразметки" type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(event) => { const value = Number(event.target.value); setOpacity(value); markupLayer.current?.setOpacity(value); }} /></label>
    </div>
    {info.warnings.length > 0 ? <details className="info-box"><summary>Не все исходные снимки доступны: подробности</summary><ul>{info.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details> : null}
    {rasterErrors.length > 0 ? <div className="info-box" role="alert">Не удалось загрузить изображение: {rasterErrors.join(", ")}. Обновите страницу, чтобы повторить загрузку.</div> : null}
    <div className="pseudo-viewer-body">
      <aside className="pseudo-sidebar">
        <h2>Снимки <span>{info.scenes.length}{info.expected_image_count != null && info.expected_image_count !== info.scenes.length ? ` из ${info.expected_image_count}` : ""}</span></h2>
        <input type="search" placeholder="Найти снимок" aria-label="Найти снимок" value={query} onChange={(event) => setQuery(event.target.value)} />
        <div className="pseudo-scene-list">{scenes.map((scene) => <button type="button" key={scene.id} title={`Приблизить ${scene.name}`} onClick={() => fit(scene.bounds)}>{scene.name}</button>)}</div>
        {!info.scenes.length ? <p>Исходные снимки недоступны. Слой псевдоразметки можно просматривать отдельно.</p> : null}
      </aside>
      <div className="pseudo-map-area">
        <div className="pseudo-map" ref={target} aria-label="Мозаика снимков с псевдоразметкой" tabIndex={0} />
        <div className="pseudo-map-legend">{classes.map((item) => <button type="button" key={item.key} aria-pressed={!hidden.has(item.key)} onClick={() => toggleClass(item.key)}><span style={{ backgroundColor: item.color }} />{item.name} <small>{item.count.toLocaleString("ru-RU")}</small></button>)}</div>
        {!geojson.features.length ? <div className="pseudo-empty">На этих снимках сеть не нашла объектов</div> : null}
      </div>
    </div>
    <p className="pseudo-hint">Наведите курсор на карту: колесо — масштаб, перетаскивание — перемещение. Нажмите на название снимка, чтобы приблизить его.{hasNir ? " NRG и NGB используют NIR; снимки без него остаются в RGB." : ""} Просмотр не изменяет разметку датасета.</p>
  </>;
}
