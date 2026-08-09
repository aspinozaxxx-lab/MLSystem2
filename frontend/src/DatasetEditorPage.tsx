import { Folder, FolderOpen, MousePointer2, Pencil, Plus, Save, Trash2 } from "lucide-react";
import Feature from "ol/Feature";
import GeoJSON from "ol/format/GeoJSON";
import type Geometry from "ol/geom/Geometry";
import Draw from "ol/interaction/Draw";
import Modify from "ol/interaction/Modify";
import Select from "ol/interaction/Select";
import Snap from "ol/interaction/Snap";
import WebGLTileLayer from "ol/layer/WebGLTile";
import VectorLayer from "ol/layer/Vector";
import OLMap from "ol/Map";
import GeoTIFF from "ol/source/GeoTIFF";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style } from "ol/style";
import { containsExtent, type Extent } from "ol/extent";
import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "ol/ol.css";

import { apiJson } from "./api/client";

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;
type JsonObject = Record<string, unknown>;
type Role = "positive" | "hard_negative";
type EditMode = "select" | "draw";

type EditorDataset = {
  key: string;
  name: string;
  class_key: string;
  class_name: string;
  dataset_name: string;
  imagery_type: "kanopus" | "ortho";
  scene_count: number;
};

type EditorScene = {
  scene_id: string;
  annotation_name: string;
  image_name: string;
  raster_url: string;
  total_count: number;
  positive_count: number;
  hard_negative_count: number;
  revision: string;
};

type SceneDetail = { scene: EditorScene; geojson: JsonObject };
type RasterFolder = { name: string; path: string };
type RasterInfo = { name: string; path: string; annotation_name: string; size_bytes: number };
type RasterBrowser = {
  folder: string;
  parent: string | null;
  folders: RasterFolder[];
  rasters: RasterInfo[];
};
type MutationResult = {
  commit: string;
  publication_status: "publishing" | "published";
  scenes: EditorScene[];
};
type PublicationInfo = {
  commit: string;
  live_commit: string | null;
  status: "publishing" | "published";
};

const ROLE_PROPERTY = "_mlsystem2_role";
const positiveStyle = new Style({
  stroke: new Stroke({ color: "#f3c623", width: 3 }),
  fill: new Fill({ color: "rgba(243, 198, 35, 0.22)" }),
});
const hardNegativeStyle = new Style({
  stroke: new Stroke({ color: "#ef4444", width: 3, lineDash: [8, 5] }),
  fill: new Fill({ color: "rgba(239, 68, 68, 0.18)" }),
});
const selectedStyle = new Style({
  stroke: new Stroke({ color: "#38bdf8", width: 4 }),
  fill: new Fill({ color: "rgba(56, 189, 248, 0.18)" }),
});

export function DatasetEditorPage({
  run,
  registerRouteGuard,
}: {
  run: Runner;
  registerRouteGuard: (guard: (() => boolean) | null) => void;
}) {
  const [datasets, setDatasets] = useState<EditorDataset[]>([]);
  const [classKey, setClassKey] = useState("");
  const [datasetKey, setDatasetKey] = useState("");
  const [scenes, setScenes] = useState<EditorScene[]>([]);
  const [annotationName, setAnnotationName] = useState("");
  const [detail, setDetail] = useState<SceneDetail | null>(null);
  const [role, setRole] = useState<Role>("positive");
  const [editMode, setEditMode] = useState<EditMode>("select");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [browser, setBrowser] = useState<RasterBrowser | null>(null);
  const [selectedRasters, setSelectedRasters] = useState<Set<string>>(new Set());
  const [publication, setPublication] = useState<PublicationInfo | null>(null);
  const mapTargetRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<OLMap | null>(null);
  const vectorSourceRef = useRef<VectorSource<Feature<Geometry>> | null>(null);
  const selectRef = useRef<Select | null>(null);
  const modifyRef = useRef<Modify | null>(null);
  const drawRef = useRef<Draw | null>(null);
  const rasterExtentRef = useRef<Extent | null>(null);
  const dirtyRef = useRef(false);
  const roleRef = useRef<Role>(role);

  const setChanged = useCallback((value: boolean) => {
    dirtyRef.current = value;
    setDirty(value);
  }, []);

  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  useEffect(() => {
    const guard = () =>
      !dirtyRef.current || window.confirm("Есть несохранённые изменения. Покинуть редактор?");
    registerRouteGuard(guard);
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirtyRef.current) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => {
      registerRouteGuard(null);
      window.removeEventListener("beforeunload", beforeUnload);
    };
  }, [registerRouteGuard]);

  const loadDatasets = useCallback(async () => {
    const payload = await run(() =>
      apiJson<{ datasets: EditorDataset[] }>("/dataset-editor/datasets"),
    );
    if (!payload) return;
    setDatasets(payload.datasets);
    const firstClass = payload.datasets[0]?.class_key || "";
    setClassKey((current) =>
      payload.datasets.some((item) => item.class_key === current) ? current : firstClass,
    );
  }, [run]);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  const classDatasets = useMemo(
    () => datasets.filter((item) => item.class_key === classKey),
    [classKey, datasets],
  );
  const classes = useMemo(
    () =>
      Array.from(new Map(datasets.map((item) => [item.class_key, item.class_name])).entries()).map(
        ([key, name]) => ({ key, name }),
      ),
    [datasets],
  );

  useEffect(() => {
    if (!classDatasets.some((item) => item.key === datasetKey)) {
      setDatasetKey(classDatasets[0]?.key || "");
    }
  }, [classDatasets, datasetKey]);

  const loadScenes = useCallback(
    async (key: string, preferredAnnotation = "") => {
      if (!key) {
        setScenes([]);
        setAnnotationName("");
        setDetail(null);
        return;
      }
      const payload = await run(() =>
        apiJson<{ dataset: EditorDataset; scenes: EditorScene[] }>(
          `/dataset-editor/datasets/${encodeURIComponent(key)}/scenes`,
        ),
      );
      if (!payload) return;
      setScenes(payload.scenes);
      const next =
        payload.scenes.find((item) => item.annotation_name === preferredAnnotation)
          ?.annotation_name || payload.scenes[0]?.annotation_name || "";
      setAnnotationName(next);
      if (!next) setDetail(null);
    },
    [run],
  );

  useEffect(() => {
    setChanged(false);
    setBrowser(null);
    setPublication(null);
    void loadScenes(datasetKey);
  }, [datasetKey, loadScenes, setChanged]);

  const loadScene = useCallback(
    async (key: string, name: string) => {
      if (!key || !name) {
        setDetail(null);
        return;
      }
      const payload = await run(() =>
        apiJson<SceneDetail>(
          `/dataset-editor/datasets/${encodeURIComponent(key)}/scenes/${encodeURIComponent(name)}`,
        ),
      );
      if (!payload) return;
      setEditMode("select");
      setDetail(payload);
      setChanged(false);
    },
    [run, setChanged],
  );

  useEffect(() => {
    void loadScene(datasetKey, annotationName);
  }, [annotationName, datasetKey, loadScene]);

  useEffect(() => {
    const target = mapTargetRef.current;
    if (!target || !detail) return;
    const format = new GeoJSON();
    const crs = geojsonCrs(detail.geojson);
    const vectorSource = new VectorSource<Feature<Geometry>>();
    const features = format.readFeatures(detail.geojson, {
      dataProjection: crs,
      featureProjection: crs,
    }) as Feature<Geometry>[];
    for (const feature of features) {
      if (!feature.get(ROLE_PROPERTY)) feature.set(ROLE_PROPERTY, "positive", true);
    }
    vectorSource.addFeatures(features);
    vectorSourceRef.current = vectorSource;

    const rasterSource = new GeoTIFF({
      sources: [{ url: detail.scene.raster_url }],
      normalize: true,
    });
    void rasterSource.getView().then((viewOptions) => {
      rasterExtentRef.current = viewOptions.extent || null;
    });
    const vectorLayer = new VectorLayer({
      source: vectorSource,
      style: (feature) =>
        feature.get(ROLE_PROPERTY) === "hard_negative" ? hardNegativeStyle : positiveStyle,
    });
    const select = new Select({ style: selectedStyle });
    const modify = new Modify({ source: vectorSource });
    const snap = new Snap({ source: vectorSource });
    const draw = new Draw({ source: vectorSource, type: "Polygon" });
    draw.setActive(false);
    const backups = new Map<Feature<Geometry>, Geometry>();

    draw.on("drawend", (event) => {
      event.feature.set(ROLE_PROPERTY, roleRef.current);
      if (!insideRaster(event.feature, rasterExtentRef.current)) {
        window.setTimeout(() => vectorSource.removeFeature(event.feature), 0);
        window.alert("Полигон должен целиком находиться внутри снимка.");
        return;
      }
      setChanged(true);
    });
    modify.on("modifystart", (event) => {
      backups.clear();
      event.features.forEach((feature) => {
        const geometry = feature.getGeometry();
        if (geometry) backups.set(feature, geometry.clone());
      });
    });
    modify.on("modifyend", (event) => {
      const outside = event.features.getArray().some((feature) =>
        !insideRaster(feature, rasterExtentRef.current),
      );
      if (outside) {
        for (const [feature, geometry] of backups) feature.setGeometry(geometry);
        window.alert("Геометрия не может выходить за границы снимка.");
        return;
      }
      setChanged(true);
    });

    const map = new OLMap({
      target,
      layers: [new WebGLTileLayer({ source: rasterSource }), vectorLayer],
      interactions: undefined,
      view: rasterSource.getView(),
    });
    map.addInteraction(select);
    map.addInteraction(modify);
    map.addInteraction(draw);
    map.addInteraction(snap);
    mapRef.current = map;
    selectRef.current = select;
    modifyRef.current = modify;
    drawRef.current = draw;

    return () => {
      map.setTarget(undefined);
      mapRef.current = null;
      vectorSourceRef.current = null;
      selectRef.current = null;
      modifyRef.current = null;
      drawRef.current = null;
      rasterExtentRef.current = null;
    };
  }, [detail, setChanged]);

  useEffect(() => {
    const drawing = editMode === "draw";
    drawRef.current?.setActive(drawing);
    selectRef.current?.setActive(!drawing);
    modifyRef.current?.setActive(!drawing);
    if (drawing) selectRef.current?.getFeatures().clear();
  }, [editMode]);

  useEffect(() => {
    if (!publication || publication.status === "published") return;
    const timer = window.setInterval(() => {
      void apiJson<PublicationInfo>(
        `/dataset-editor/publication/${encodeURIComponent(publication.commit)}`,
      )
        .then(setPublication)
        .catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [publication]);

  const selectClass = (event: ChangeEvent<HTMLSelectElement>) => {
    if (dirtyRef.current && !window.confirm("Отменить несохранённые изменения?")) return;
    setChanged(false);
    setClassKey(event.target.value);
  };

  const selectDataset = (event: ChangeEvent<HTMLSelectElement>) => {
    if (dirtyRef.current && !window.confirm("Отменить несохранённые изменения?")) return;
    setChanged(false);
    setDatasetKey(event.target.value);
  };

  const selectScene = (name: string) => {
    if (name === annotationName) return;
    if (dirtyRef.current && !window.confirm("Отменить несохранённые изменения?")) return;
    setChanged(false);
    setAnnotationName(name);
  };

  const applyRole = () => {
    const selected = selectRef.current?.getFeatures().getArray() || [];
    if (!selected.length) {
      window.alert("Сначала выберите объект на карте.");
      return;
    }
    selected.forEach((feature) => feature.set(ROLE_PROPERTY, role));
    vectorSourceRef.current?.changed();
    setChanged(true);
  };

  const deleteSelected = () => {
    const selected = selectRef.current?.getFeatures();
    const source = vectorSourceRef.current;
    if (!selected || !source || selected.getLength() === 0) return;
    selected.getArray().forEach((feature) => source.removeFeature(feature));
    selected.clear();
    setChanged(true);
  };

  const save = async () => {
    if (!detail || !vectorSourceRef.current) return;
    setBusy(true);
    const crs = geojsonCrs(detail.geojson);
    const features = vectorSourceRef.current.getFeatures();
    for (const feature of features) {
      if (!feature.get(ROLE_PROPERTY)) feature.set(ROLE_PROPERTY, "positive", true);
    }
    const collection = new GeoJSON().writeFeaturesObject(features, {
      dataProjection: crs,
      featureProjection: crs,
    }) as JsonObject;
    const geojson = { ...detail.geojson, ...collection, crs: detail.geojson.crs };
    const result = await run(() =>
      apiJson<MutationResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/scenes/${encodeURIComponent(detail.scene.annotation_name)}`,
        { method: "PUT", body: { revision: detail.scene.revision, geojson } },
      ),
    );
    setBusy(false);
    if (!result) return;
    const updatedScene = result.scenes[0] || detail.scene;
    setDetail({ scene: updatedScene, geojson });
    setScenes((current) =>
      current.map((item) =>
        item.annotation_name === updatedScene.annotation_name ? updatedScene : item,
      ),
    );
    setChanged(false);
    setPublication({
      commit: result.commit,
      live_commit: null,
      status: result.publication_status,
    });
  };

  const removeScene = async () => {
    if (!detail || !window.confirm(`Удалить снимок ${detail.scene.image_name} из датасета?`)) return;
    setBusy(true);
    const result = await run(() =>
      apiJson<MutationResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/scenes/${encodeURIComponent(detail.scene.annotation_name)}`,
        { method: "DELETE", body: { revision: detail.scene.revision } },
      ),
    );
    setBusy(false);
    if (!result) return;
    setChanged(false);
    setPublication({ commit: result.commit, live_commit: null, status: result.publication_status });
    await loadScenes(datasetKey);
  };

  const loadBrowser = async (folder: string) => {
    if (!datasetKey) return;
    const payload = await run(() =>
      apiJson<RasterBrowser>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/rasters?folder=${encodeURIComponent(folder)}`,
      ),
    );
    if (payload) {
      setBrowser(payload);
      setSelectedRasters(new Set());
    }
  };

  const addRasters = async (wholeFolder: boolean) => {
    if (!browser) return;
    const body = wholeFolder
      ? { folder_path: browser.folder }
      : { image_paths: Array.from(selectedRasters) };
    if (!wholeFolder && selectedRasters.size === 0) return;
    setBusy(true);
    const result = await run(() =>
      apiJson<MutationResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/scenes`,
        { method: "POST", body },
      ),
    );
    setBusy(false);
    if (!result) return;
    const preferred = result.scenes[0]?.annotation_name || "";
    setPublication({ commit: result.commit, live_commit: null, status: result.publication_status });
    setBrowser(null);
    await loadScenes(datasetKey, preferred);
  };

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <h1>Редактор датасетов</h1>
          <p>Один GeoJSON на снимок · изменения сохраняются через отдельный клон MLMarkup</p>
        </div>
        {publication ? (
          <span className={`badge ${publication.status === "published" ? "ok" : "running"}`}>
            {publication.status === "published" ? "опубликовано" : "публикуется"} · {publication.commit.slice(0, 8)}
          </span>
        ) : null}
      </header>

      <section className="dataset-editor-selectors panel">
        <label className="field">
          <span>Класс</span>
          <select value={classKey} onChange={selectClass}>
            {classes.map((item) => <option key={item.key} value={item.key}>{item.name}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Датасет</span>
          <select value={datasetKey} onChange={selectDataset}>
            {classDatasets.map((item) => (
              <option key={item.key} value={item.key}>{item.dataset_name} ({item.scene_count})</option>
            ))}
          </select>
        </label>
        <button className="secondary" type="button" disabled={!datasetKey} onClick={() => void loadBrowser("")}>
          <Plus size={16} /> Добавить снимки
        </button>
      </section>

      {!datasets.length ? (
        <section className="panel empty-state">Per-image датасеты в editor-клоне не найдены.</section>
      ) : (
        <section className="dataset-editor-layout">
          <aside className="panel dataset-editor-scenes">
            <div className="panel-header"><div><h2>Размеченные снимки</h2><p>{scenes.length} шт.</p></div></div>
            <div className="dataset-editor-scene-list">
              {scenes.map((scene) => (
                <button
                  className={scene.annotation_name === annotationName ? "active" : ""}
                  type="button"
                  key={scene.annotation_name}
                  onClick={() => selectScene(scene.annotation_name)}
                >
                  <strong>{scene.image_name}</strong>
                  <span>{scene.total_count} объектов · +{scene.positive_count} / −{scene.hard_negative_count}</span>
                </button>
              ))}
              {!scenes.length ? <p className="muted">Снимки ещё не добавлены.</p> : null}
            </div>
          </aside>

          <section className="panel dataset-editor-workspace">
            {detail ? (
              <>
                <div className="dataset-editor-toolbar">
                  <span className="source-lines"><strong>{detail.scene.image_name}</strong><small className="muted">{detail.scene.annotation_name}</small></span>
                  <label className="field compact-field">
                    <span>Роль</span>
                    <select value={role} onChange={(event) => setRole(event.target.value as Role)}>
                      <option value="positive">Положительный</option>
                      <option value="hard_negative">Сложный отрицательный</option>
                    </select>
                  </label>
                  <button className={editMode === "select" ? "primary" : "secondary"} type="button" onClick={() => setEditMode("select")}><MousePointer2 size={15} /> Выбор и правка</button>
                  <button className={editMode === "draw" ? "primary" : "secondary"} type="button" onClick={() => setEditMode("draw")}><Pencil size={15} /> Новый полигон</button>
                  <button className="secondary" type="button" onClick={applyRole}><Pencil size={15} /> Роль выбранного</button>
                  <button className="danger" type="button" onClick={deleteSelected}><Trash2 size={15} /> Удалить объект</button>
                  <button className="primary" type="button" disabled={!dirty || busy} onClick={() => void save()}><Save size={15} /> Сохранить</button>
                  <button className="danger" type="button" disabled={busy} onClick={() => void removeScene()}><Trash2 size={15} /> Удалить снимок</button>
                </div>
                <div className="dataset-editor-help"><MousePointer2 size={14} /> Клик — выбор, перетаскивание вершин — изменение, новый полигон рисуется кликами и завершается двойным кликом.</div>
                <div className="dataset-editor-map" ref={mapTargetRef} />
              </>
            ) : (
              <div className="empty-state">Выберите или добавьте снимок.</div>
            )}
          </section>
        </section>
      )}

      {browser ? (
        <section className="panel dataset-editor-browser">
          <div className="panel-header">
            <div><h2>Серверные TIFF</h2><p>Папка: /{browser.folder}</p></div>
            <div className="button-row">
              <button className="secondary" type="button" disabled={browser.parent === null} onClick={() => void loadBrowser(browser.parent || "")}><FolderOpen size={15} /> Выше</button>
              <button className="secondary" type="button" disabled={!browser.rasters.length || busy} onClick={() => void addRasters(true)}><Plus size={15} /> Добавить всю папку</button>
              <button className="primary" type="button" disabled={!selectedRasters.size || busy} onClick={() => void addRasters(false)}><Plus size={15} /> Добавить выбранные</button>
              <button className="ghost" type="button" onClick={() => setBrowser(null)}>Закрыть</button>
            </div>
          </div>
          <div className="dataset-editor-file-grid">
            {browser.folders.map((folder) => (
              <button type="button" className="dataset-editor-file folder" key={folder.path} onClick={() => void loadBrowser(folder.path)}><Folder size={17} /><span>{folder.name}</span></button>
            ))}
            {browser.rasters.map((raster) => (
              <label className="dataset-editor-file" key={raster.path}>
                <input
                  type="checkbox"
                  checked={selectedRasters.has(raster.path)}
                  onChange={(event) => {
                    setSelectedRasters((current) => {
                      const next = new Set(current);
                      if (event.target.checked) next.add(raster.path); else next.delete(raster.path);
                      return next;
                    });
                  }}
                />
                <span><strong>{raster.name}</strong><small>{raster.annotation_name}</small></span>
              </label>
            ))}
          </div>
        </section>
      ) : null}
    </>
  );
}

function geojsonCrs(payload: JsonObject): string {
  const crs = payload.crs;
  if (crs && typeof crs === "object") {
    const properties = (crs as JsonObject).properties;
    if (properties && typeof properties === "object") {
      const name = (properties as JsonObject).name;
      if (typeof name === "string" && name) return name;
    }
    const name = (crs as JsonObject).name;
    if (typeof name === "string" && name) return name;
  }
  return "EPSG:4326";
}

function insideRaster(feature: Feature<Geometry>, extent: Extent | null): boolean {
  const geometry = feature.getGeometry();
  return Boolean(geometry && (!extent || containsExtent(extent, geometry.getExtent())));
}
