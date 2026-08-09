import {
  ArrowDownNarrowWide,
  ArrowUpNarrowWide,
  Blend,
  CloudUpload,
  Folder,
  FolderOpen,
  MousePointer2,
  PaintBucket,
  PencilLine,
  Plus,
  Trash2,
  Undo2,
} from "lucide-react";
import Feature from "ol/Feature";
import { defaults as defaultControls } from "ol/control/defaults";
import { containsExtent, type Extent } from "ol/extent";
import GeoJSON from "ol/format/GeoJSON";
import type Geometry from "ol/geom/Geometry";
import Draw from "ol/interaction/Draw";
import Modify from "ol/interaction/Modify";
import Select from "ol/interaction/Select";
import Snap from "ol/interaction/Snap";
import WebGLTileLayer, { type Style as WebGLTileStyle } from "ol/layer/WebGLTile";
import VectorLayer from "ol/layer/Vector";
import OLMap from "ol/Map";
import GeoTIFF from "ol/source/GeoTIFF";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style } from "ol/style";
import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "ol/ol.css";

import { apiJson } from "./api/client";
import {
  acceptPublishedDraft,
  appendHistory,
  cloneSnapshot,
  draftChanged,
  publishScenes as buildPublishScenes,
  sceneCounts,
  sortEditorScenes,
  undoDraft,
  type DraftSnapshot,
  type DraftState,
  type JsonObject,
  type SortDirection,
} from "./utils/datasetEditor";

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;
type Role = "positive" | "hard_negative";
type EditMode = "select" | "draw";
type BandMode = "RGB" | "NRG" | "NGB";

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
type SceneDraft = DraftState & {
  scene: EditorScene;
  normalized: boolean;
};
type DraftMap = Record<string, SceneDraft>;
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
const BAND_CHANNELS: Record<BandMode, [number, number, number]> = {
  RGB: [1, 2, 3],
  NRG: [4, 1, 2],
  NGB: [4, 2, 3],
};
const styleCache = new Map<string, Style>();

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
  const [drafts, setDrafts] = useState<DraftMap>({});
  const [role, setRole] = useState<Role>("positive");
  const [editMode, setEditMode] = useState<EditMode>("select");
  const [sortDirection, setSortDirection] = useState<SortDirection>("descending");
  const [fillEnabled, setFillEnabled] = useState(true);
  const [bandMode, setBandMode] = useState<BandMode>("RGB");
  const [bandMenuOpen, setBandMenuOpen] = useState(false);
  const [drawInProgress, setDrawInProgress] = useState(false);
  const [busy, setBusy] = useState(false);
  const [browser, setBrowser] = useState<RasterBrowser | null>(null);
  const [selectedRasters, setSelectedRasters] = useState<Set<string>>(new Set());
  const [publication, setPublication] = useState<PublicationInfo | null>(null);
  const mapTargetRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<OLMap | null>(null);
  const vectorSourceRef = useRef<VectorSource<Feature<Geometry>> | null>(null);
  const vectorLayerRef = useRef<VectorLayer<VectorSource<Feature<Geometry>>> | null>(null);
  const rasterLayerRef = useRef<WebGLTileLayer | null>(null);
  const selectRef = useRef<Select | null>(null);
  const modifyRef = useRef<Modify | null>(null);
  const drawRef = useRef<Draw | null>(null);
  const rasterExtentRef = useRef<Extent | null>(null);
  const draftsRef = useRef<DraftMap>({});
  const dirtyRef = useRef(false);
  const roleRef = useRef<Role>(role);
  const fillEnabledRef = useRef(fillEnabled);
  const bandModeRef = useRef<BandMode>(bandMode);
  const activeAnnotationRef = useRef(annotationName);
  const drawInProgressRef = useRef(false);
  const newFeaturesRef = useRef<WeakSet<Feature<Geometry>>>(new WeakSet());
  const sceneLoadRequestRef = useRef(0);

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.key === datasetKey) || null,
    [datasetKey, datasets],
  );
  const hasDirtyDrafts = useMemo(
    () => Object.values(drafts).some(draftChanged),
    [drafts],
  );
  const dirtyDraftCount = useMemo(
    () => Object.values(drafts).filter(draftChanged).length,
    [drafts],
  );
  const sortedScenes = useMemo(
    () => sortEditorScenes(scenes, drafts, sortDirection),
    [drafts, scenes, sortDirection],
  );
  const activeDraft = annotationName ? drafts[annotationName] : undefined;

  const changeDrafts = useCallback((updater: (current: DraftMap) => DraftMap) => {
    const next = updater(draftsRef.current);
    draftsRef.current = next;
    setDrafts(next);
  }, []);

  const updateDraft = useCallback(
    (name: string, updater: (current: SceneDraft) => SceneDraft) => {
      changeDrafts((current) => {
        const draft = current[name];
        if (!draft) return current;
        return { ...current, [name]: updater(draft) };
      });
    },
    [changeDrafts],
  );

  const resetDrafts = useCallback(() => {
    draftsRef.current = {};
    setDrafts({});
  }, []);

  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  useEffect(() => {
    activeAnnotationRef.current = annotationName;
  }, [annotationName]);

  useEffect(() => {
    dirtyRef.current = hasDirtyDrafts;
  }, [hasDirtyDrafts]);

  useEffect(() => {
    const guard = () =>
      !dirtyRef.current ||
      window.confirm("Есть неопубликованные изменения. Покинуть редактор и потерять черновики?");
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
          ?.annotation_name ||
        payload.scenes.find((item) => item.annotation_name === activeAnnotationRef.current)
          ?.annotation_name ||
        payload.scenes[0]?.annotation_name ||
        "";
      setAnnotationName(next);
      if (!next) setDetail(null);
    },
    [run],
  );

  useEffect(() => {
    resetDrafts();
    setDetail(null);
    setBrowser(null);
    setPublication(null);
    setBandMode("RGB");
    bandModeRef.current = "RGB";
    setBandMenuOpen(false);
    void loadScenes(datasetKey);
  }, [datasetKey, loadScenes, resetDrafts]);

  const loadScene = useCallback(
    async (key: string, name: string) => {
      const requestId = ++sceneLoadRequestRef.current;
      if (!key || !name) {
        setDetail(null);
        return;
      }
      const cached = draftsRef.current[name];
      if (cached) {
        setEditMode("select");
        setDetail({ scene: cached.scene, geojson: cached.current.geojson });
        return;
      }
      const payload = await run(() =>
        apiJson<SceneDetail>(
          `/dataset-editor/datasets/${encodeURIComponent(key)}/scenes/${encodeURIComponent(name)}`,
        ),
      );
      if (!payload || requestId !== sceneLoadRequestRef.current) return;
      const initial = { geojson: payload.geojson, newFeatureIndexes: [] };
      changeDrafts((current) => ({
        ...current,
        [name]: {
          scene: payload.scene,
          baseline: cloneSnapshot(initial),
          current: cloneSnapshot(initial),
          history: [],
          normalized: false,
        },
      }));
      setEditMode("select");
      setDetail(payload);
    },
    [changeDrafts, run],
  );

  useEffect(() => {
    void loadScene(datasetKey, annotationName);
  }, [annotationName, datasetKey, loadScene]);

  const captureActiveSnapshot = useCallback((): DraftSnapshot | null => {
    const name = activeAnnotationRef.current;
    const draft = draftsRef.current[name];
    const source = vectorSourceRef.current;
    if (!draft || !source) return null;
    return vectorSnapshot(
      source,
      geojsonCrs(draft.current.geojson),
      draft.current.geojson,
      newFeaturesRef.current,
    );
  }, []);

  const recordCurrentChange = useCallback(
    (before: DraftSnapshot) => {
      const name = activeAnnotationRef.current;
      const current = captureActiveSnapshot();
      if (!name || !current) return;
      updateDraft(name, (draft) => ({
        ...draft,
        current,
        history: appendHistory(draft.history, before),
        normalized: true,
      }));
    },
    [captureActiveSnapshot, updateDraft],
  );

  const setDrawingState = useCallback((value: boolean) => {
    drawInProgressRef.current = value;
    setDrawInProgress(value);
  }, []);

  useEffect(() => {
    const target = mapTargetRef.current;
    if (!target || !detail) return;
    const name = detail.scene.annotation_name;
    const draft = draftsRef.current[name];
    if (!draft) return;
    const format = new GeoJSON();
    const crs = geojsonCrs(draft.current.geojson);
    const vectorSource = new VectorSource<Feature<Geometry>>();
    const features = format.readFeatures(draft.current.geojson, {
      dataProjection: crs,
      featureProjection: crs,
    }) as Feature<Geometry>[];
    const newFeatures = new WeakSet<Feature<Geometry>>();
    for (const [index, feature] of features.entries()) {
      if (!feature.get(ROLE_PROPERTY)) feature.set(ROLE_PROPERTY, "positive", true);
      if (draft.current.newFeatureIndexes.includes(index)) newFeatures.add(feature);
    }
    newFeaturesRef.current = newFeatures;
    vectorSource.addFeatures(features);
    vectorSourceRef.current = vectorSource;

    const rasterSource = new GeoTIFF({
      sources: [{ url: detail.scene.raster_url }],
      normalize: true,
    });
    let active = true;
    void rasterSource.getView().then((viewOptions) => {
      if (active) rasterExtentRef.current = viewOptions.extent || null;
    });
    const rasterLayer = new WebGLTileLayer({
      source: rasterSource,
      style: rasterBandStyle(
        selectedDataset?.imagery_type === "kanopus" ? bandModeRef.current : "RGB",
      ),
    });
    const vectorLayer = new VectorLayer({
      source: vectorSource,
      style: (feature) =>
        featureStyle(
          feature as Feature<Geometry>,
          false,
          fillEnabledRef.current,
          newFeaturesRef.current,
        ),
    });
    const select = new Select({
      style: (feature) =>
        featureStyle(
          feature as Feature<Geometry>,
          true,
          fillEnabledRef.current,
          newFeaturesRef.current,
        ),
    });
    const modify = new Modify({ source: vectorSource });
    const snap = new Snap({ source: vectorSource });
    const draw = new Draw({ source: vectorSource, type: "Polygon" });
    draw.setActive(false);
    const geometryBackups = new Map<Feature<Geometry>, Geometry>();
    let drawBefore: DraftSnapshot | null = null;
    let modifyBefore: DraftSnapshot | null = null;
    let drawCommitTimer: number | null = null;

    const normalized = vectorSnapshot(
      vectorSource,
      crs,
      draft.current.geojson,
      newFeaturesRef.current,
    );
    if (!draft.normalized) {
      updateDraft(name, (current) => ({
        ...current,
        baseline: cloneSnapshot(normalized),
        current: cloneSnapshot(normalized),
        normalized: true,
      }));
    }

    select.on("select", (event) => {
      const selected = event.selected[0] as Feature<Geometry> | undefined;
      if (!selected) return;
      const selectedRole = selected.get(ROLE_PROPERTY) === "hard_negative" ? "hard_negative" : "positive";
      roleRef.current = selectedRole;
      setRole(selectedRole);
    });
    draw.on("drawstart", () => {
      drawBefore = captureActiveSnapshot();
      setDrawingState(true);
    });
    draw.on("drawabort", () => {
      drawBefore = null;
      setDrawingState(false);
    });
    draw.on("drawend", (event) => {
      setDrawingState(false);
      event.feature.set(ROLE_PROPERTY, roleRef.current);
      if (!insideRaster(event.feature, rasterExtentRef.current)) {
        drawBefore = null;
        window.setTimeout(() => vectorSource.removeFeature(event.feature), 0);
        window.alert("Полигон должен целиком находиться внутри снимка.");
        return;
      }
      newFeaturesRef.current.add(event.feature);
      const before = drawBefore;
      drawBefore = null;
      if (before) {
        drawCommitTimer = window.setTimeout(() => {
          if (vectorSourceRef.current === vectorSource) recordCurrentChange(before);
        }, 0);
      }
    });
    modify.on("modifystart", (event) => {
      modifyBefore = captureActiveSnapshot();
      geometryBackups.clear();
      event.features.forEach((feature) => {
        const geometry = feature.getGeometry();
        if (geometry) geometryBackups.set(feature, geometry.clone());
      });
    });
    modify.on("modifyend", (event) => {
      const outside = event.features.getArray().some((feature) =>
        !insideRaster(feature, rasterExtentRef.current),
      );
      if (outside) {
        for (const [feature, geometry] of geometryBackups) feature.setGeometry(geometry);
        modifyBefore = null;
        window.alert("Геометрия не может выходить за границы снимка.");
        return;
      }
      const before = modifyBefore;
      modifyBefore = null;
      if (before) recordCurrentChange(before);
    });

    const map = new OLMap({
      target,
      controls: defaultControls({ zoom: false }),
      layers: [rasterLayer, vectorLayer],
      interactions: undefined,
      view: rasterSource.getView(),
    });
    map.addInteraction(select);
    map.addInteraction(modify);
    map.addInteraction(draw);
    map.addInteraction(snap);
    mapRef.current = map;
    vectorLayerRef.current = vectorLayer;
    rasterLayerRef.current = rasterLayer;
    selectRef.current = select;
    modifyRef.current = modify;
    drawRef.current = draw;

    return () => {
      active = false;
      if (drawCommitTimer !== null) window.clearTimeout(drawCommitTimer);
      map.setTarget(undefined);
      mapRef.current = null;
      vectorSourceRef.current = null;
      vectorLayerRef.current = null;
      rasterLayerRef.current = null;
      selectRef.current = null;
      modifyRef.current = null;
      drawRef.current = null;
      rasterExtentRef.current = null;
      newFeaturesRef.current = new WeakSet();
      setDrawingState(false);
    };
  }, [
    captureActiveSnapshot,
    detail,
    recordCurrentChange,
    selectedDataset?.imagery_type,
    setDrawingState,
    updateDraft,
  ]);

  useEffect(() => {
    const drawing = editMode === "draw";
    drawRef.current?.setActive(drawing);
    selectRef.current?.setActive(!drawing);
    modifyRef.current?.setActive(!drawing);
    if (drawing) selectRef.current?.getFeatures().clear();
  }, [editMode]);

  useEffect(() => {
    const effectiveMode = selectedDataset?.imagery_type === "kanopus" ? bandMode : "RGB";
    rasterLayerRef.current?.setStyle(rasterBandStyle(effectiveMode));
  }, [bandMode, selectedDataset?.imagery_type]);

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

  const confirmDiscardDrafts = () =>
    !dirtyRef.current || window.confirm("Отменить все неопубликованные изменения датасета?");

  const selectClass = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!confirmDiscardDrafts()) return;
    resetDrafts();
    setClassKey(event.target.value);
  };

  const selectDataset = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!confirmDiscardDrafts()) return;
    resetDrafts();
    setDatasetKey(event.target.value);
  };

  const selectScene = (name: string) => {
    if (name !== annotationName) setAnnotationName(name);
  };

  const changeRole = (nextRole: Role) => {
    const selected = selectRef.current?.getFeatures().getArray() || [];
    if (selected.length && selected.some((feature) => feature.get(ROLE_PROPERTY) !== nextRole)) {
      const before = captureActiveSnapshot();
      selected.forEach((feature) => feature.set(ROLE_PROPERTY, nextRole));
      vectorSourceRef.current?.changed();
      mapRef.current?.render();
      if (before) recordCurrentChange(before);
    }
    roleRef.current = nextRole;
    setRole(nextRole);
  };

  const deleteSelected = () => {
    const selected = selectRef.current?.getFeatures();
    const source = vectorSourceRef.current;
    if (!selected || !source || selected.getLength() === 0) return;
    const before = captureActiveSnapshot();
    selected.getArray().forEach((feature) => source.removeFeature(feature));
    selected.clear();
    if (before) recordCurrentChange(before);
  };

  const restoreActiveSnapshot = useCallback((snapshot: DraftSnapshot) => {
    const source = vectorSourceRef.current;
    if (!source) return;
    restoreVectorSnapshot(source, snapshot, geojsonCrs(snapshot.geojson), newFeaturesRef);
    selectRef.current?.getFeatures().clear();
    vectorLayerRef.current?.changed();
    mapRef.current?.render();
  }, []);

  const undoCurrent = useCallback(() => {
    if (drawInProgressRef.current) {
      drawRef.current?.removeLastPoint();
      return;
    }
    const name = activeAnnotationRef.current;
    const draft = draftsRef.current[name];
    if (!draft) return;
    const undone = undoDraft(draft);
    if (!undone) return;
    const restored: SceneDraft = {
      ...draft,
      baseline: undone.baseline,
      current: undone.current,
      history: undone.history,
      normalized: true,
    };
    changeDrafts((current) => ({ ...current, [name]: restored }));
    restoreActiveSnapshot(restored.current);
  }, [changeDrafts, restoreActiveSnapshot]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLowerCase() !== "z") return;
      if (isTextInput(event.target)) return;
      event.preventDefault();
      undoCurrent();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [undoCurrent]);

  const toggleFill = () => {
    const next = !fillEnabledRef.current;
    fillEnabledRef.current = next;
    setFillEnabled(next);
    vectorLayerRef.current?.changed();
    mapRef.current?.render();
  };

  const selectBandMode = (next: BandMode) => {
    bandModeRef.current = next;
    setBandMode(next);
    setBandMenuOpen(false);
  };

  const publish = async () => {
    const items = buildPublishScenes(draftsRef.current);
    if (!datasetKey || !items.length || drawInProgressRef.current) return;
    setBusy(true);
    const result = await run(() =>
      apiJson<MutationResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/scenes`,
        { method: "PUT", body: { scenes: items } },
      ),
    );
    setBusy(false);
    if (!result) return;
    const updatedByName = new Map(result.scenes.map((scene) => [scene.annotation_name, scene]));
    setScenes((current) =>
      current.map((scene) => updatedByName.get(scene.annotation_name) || scene),
    );
    changeDrafts((current) => {
      const next = { ...current };
      for (const item of items) {
        const draft = next[item.annotation_name];
        const updatedScene = updatedByName.get(item.annotation_name);
        if (!draft || !updatedScene) continue;
        next[item.annotation_name] = {
          ...acceptPublishedDraft(draft, updatedScene),
          normalized: true,
        };
      }
      return next;
    });
    const refreshedActive = draftsRef.current[activeAnnotationRef.current];
    if (refreshedActive && updatedByName.has(refreshedActive.scene.annotation_name)) {
      setDetail({ scene: refreshedActive.scene, geojson: refreshedActive.current.geojson });
    }
    setPublication({
      commit: result.commit,
      live_commit: null,
      status: result.publication_status,
    });
  };

  const removeScene = async () => {
    const draft = draftsRef.current[activeAnnotationRef.current];
    if (!draft) return;
    const warning = draftChanged(draft)
      ? `Удалить снимок ${draft.scene.image_name}? Его неопубликованные изменения будут потеряны.`
      : `Удалить снимок ${draft.scene.image_name} из датасета?`;
    if (!window.confirm(warning)) return;
    setBusy(true);
    const result = await run(() =>
      apiJson<MutationResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/scenes/${encodeURIComponent(draft.scene.annotation_name)}`,
        { method: "DELETE", body: { revision: draft.scene.revision } },
      ),
    );
    setBusy(false);
    if (!result) return;
    changeDrafts((current) => {
      const next = { ...current };
      delete next[draft.scene.annotation_name];
      return next;
    });
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
          <p>Один GeoJSON на снимок · изменения публикуются через отдельный клон MLMarkup</p>
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
        <button
          className="secondary"
          type="button"
          disabled={!datasetKey}
          title="Открыть список серверных TIFF для добавления в датасет"
          onClick={() => void loadBrowser("")}
        >
          <Plus size={16} /> Добавить снимки
        </button>
      </section>

      {!datasets.length ? (
        <section className="panel empty-state">Per-image датасеты в editor-клоне не найдены.</section>
      ) : (
        <section className="dataset-editor-layout">
          <aside className="panel dataset-editor-scenes">
            <div className="dataset-editor-scenes-header">
              <div><h2>Снимки</h2><p>{scenes.length} шт.</p></div>
              <button
                className="secondary icon-button dataset-editor-small-icon"
                type="button"
                aria-label={sortDirection === "descending" ? "Сортировать по возрастанию" : "Сортировать по убыванию"}
                title={sortDirection === "descending" ? "Сейчас сначала снимки с большим числом объектов. Переключить на возрастание" : "Сейчас сначала снимки с малым числом объектов. Переключить на убывание"}
                onClick={() => setSortDirection((current) => current === "descending" ? "ascending" : "descending")}
              >
                {sortDirection === "descending" ? <ArrowDownNarrowWide size={16} /> : <ArrowUpNarrowWide size={16} />}
              </button>
            </div>
            <button
              className="primary dataset-editor-publish"
              type="button"
              disabled={!dirtyDraftCount || busy || drawInProgress}
              title={dirtyDraftCount ? `Опубликовать изменения ${dirtyDraftCount} снимков одним коммитом` : "Нет неопубликованных изменений"}
              onClick={() => void publish()}
            >
              <CloudUpload size={16} /> Опубликовать
            </button>
            <div className="dataset-editor-scene-list">
              {sortedScenes.map((scene) => {
                const draft = drafts[scene.annotation_name];
                const counts = sceneCounts(scene, draft);
                const changed = Boolean(draft && draftChanged(draft));
                return (
                  <button
                    className={scene.annotation_name === annotationName ? "active" : ""}
                    type="button"
                    key={scene.annotation_name}
                    title={`Открыть снимок ${scene.image_name}. ${counts.total} объектов: ${counts.positive} positive, ${counts.hardNegative} hard negative${changed ? ". Есть неопубликованные изменения" : ""}`}
                    onClick={() => selectScene(scene.annotation_name)}
                  >
                    {changed ? <span className="dataset-editor-dirty-dot" aria-label="Есть неопубликованные изменения" /> : null}
                    <span>{counts.total} · +{counts.positive} / −{counts.hardNegative}</span>
                  </button>
                );
              })}
              {!scenes.length ? <p className="muted">Снимки ещё не добавлены.</p> : null}
            </div>
          </aside>

          <section className="panel dataset-editor-workspace">
            {detail && activeDraft ? (
              <>
                <div className="dataset-editor-toolbar">
                  <span className="source-lines">
                    <strong>{activeDraft.scene.image_name}</strong>
                    <small className="muted">{activeDraft.scene.annotation_name}</small>
                  </span>
                  <div className="dataset-editor-map-actions">
                    <div className="dataset-editor-mode-toggle" role="group" aria-label="Режим редактирования">
                      <button
                        className={`${editMode === "select" ? "primary" : "secondary"} icon-button dataset-editor-icon-button`}
                        type="button"
                        aria-label="Выбор и правка полигонов"
                        aria-pressed={editMode === "select"}
                        title="Выбор и правка: выбрать полигон и перемещать его вершины"
                        onClick={() => setEditMode("select")}
                      >
                        <MousePointer2 size={17} />
                      </button>
                      <button
                        className={`${editMode === "draw" ? "primary" : "secondary"} icon-button dataset-editor-icon-button`}
                        type="button"
                        aria-label="Нарисовать новый полигон"
                        aria-pressed={editMode === "draw"}
                        title="Новый полигон: ставьте вершины кликами, завершите двойным кликом"
                        onClick={() => setEditMode("draw")}
                      >
                        <PencilLine size={17} />
                      </button>
                    </div>
                    <label
                      className={`dataset-editor-role-switch ${role === "hard_negative" ? "negative" : "positive"}`}
                      title={role === "positive" ? "Роль positive. Переключить вправо на hard negative" : "Роль hard negative. Переключить влево на positive"}
                    >
                      <span aria-hidden="true">+</span>
                      <input
                        type="checkbox"
                        checked={role === "hard_negative"}
                        aria-label="Роль полигона: слева positive, справа hard negative"
                        onChange={(event) => changeRole(event.target.checked ? "hard_negative" : "positive")}
                      />
                      <span className="dataset-editor-role-track" aria-hidden="true"><span /></span>
                      <span aria-hidden="true">−</span>
                    </label>
                    <button
                      className="secondary icon-button dataset-editor-icon-button"
                      type="button"
                      disabled={!drawInProgress && !activeDraft.history.length}
                      aria-label="Отменить последнее действие"
                      title="Отменить последнее действие (Ctrl+Z)"
                      onClick={undoCurrent}
                    >
                      <Undo2 size={17} />
                    </button>
                    <button
                      className="danger icon-button dataset-editor-icon-button"
                      type="button"
                      aria-label="Удалить выбранный полигон"
                      title="Удалить выбранный полигон; действие можно отменить через Ctrl+Z"
                      onClick={deleteSelected}
                    >
                      <Trash2 size={17} />
                    </button>
                    <button
                      className="danger icon-button dataset-editor-icon-button"
                      type="button"
                      disabled={busy}
                      aria-label="Удалить снимок из датасета"
                      title="Удалить текущий снимок и его GeoJSON из датасета"
                      onClick={() => void removeScene()}
                    >
                      <Trash2 size={17} />
                    </button>
                  </div>
                </div>
                <div className="dataset-editor-help">
                  <MousePointer2 size={14} /> Клик — выбор, перетаскивание вершин — изменение, Ctrl+Z — отмена.
                </div>
                <div className="dataset-editor-map-shell">
                  <div className="dataset-editor-map" ref={mapTargetRef} />
                  <div className="dataset-editor-map-controls">
                    <button
                      className={`${fillEnabled ? "primary" : "secondary"} icon-button dataset-editor-map-control`}
                      type="button"
                      aria-label={fillEnabled ? "Выключить заливку полигонов" : "Включить заливку полигонов"}
                      aria-pressed={fillEnabled}
                      title={fillEnabled ? "Скрыть заливку полигонов, оставив контуры" : "Показать заливку полигонов"}
                      onClick={toggleFill}
                    >
                      <PaintBucket size={17} />
                    </button>
                    {selectedDataset?.imagery_type === "kanopus" ? (
                      <div
                        className={`dataset-editor-band-picker ${bandMenuOpen ? "open" : ""}`}
                        onBlur={(event) => {
                          if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) {
                            setBandMenuOpen(false);
                          }
                        }}
                      >
                        <button
                          className="secondary icon-button dataset-editor-map-control"
                          type="button"
                          aria-label={`Сочетание каналов ${bandMode}`}
                          aria-haspopup="menu"
                          aria-expanded={bandMenuOpen}
                          title={`Сочетание каналов снимка: ${bandMode}. Открыть варианты RGB, NRG и NGB`}
                          onClick={() => setBandMenuOpen((current) => !current)}
                        >
                          <Blend size={17} />
                        </button>
                        <div className="dataset-editor-band-menu" role="menu">
                          {(Object.keys(BAND_CHANNELS) as BandMode[]).map((mode) => (
                            <button
                              className={mode === bandMode ? "active" : ""}
                              type="button"
                              role="menuitemradio"
                              aria-checked={mode === bandMode}
                              title={`Показать снимок в сочетании каналов ${mode}`}
                              key={mode}
                              onClick={() => selectBandMode(mode)}
                            >
                              {mode}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
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
              <button
                className="secondary"
                type="button"
                disabled={browser.parent === null}
                title="Перейти в родительскую папку"
                onClick={() => void loadBrowser(browser.parent || "")}
              ><FolderOpen size={15} /> Выше</button>
              <button
                className="secondary"
                type="button"
                disabled={!browser.rasters.length || busy}
                title="Добавить все TIFF непосредственно из текущей папки"
                onClick={() => void addRasters(true)}
              ><Plus size={15} /> Добавить всю папку</button>
              <button
                className="primary"
                type="button"
                disabled={!selectedRasters.size || busy}
                title="Добавить отмеченные TIFF в датасет"
                onClick={() => void addRasters(false)}
              ><Plus size={15} /> Добавить выбранные</button>
              <button
                className="ghost"
                type="button"
                title="Закрыть выбор серверных TIFF"
                onClick={() => setBrowser(null)}
              >Закрыть</button>
            </div>
          </div>
          <div className="dataset-editor-file-grid">
            {browser.folders.map((folder) => (
              <button
                type="button"
                className="dataset-editor-file folder"
                key={folder.path}
                title={`Открыть папку ${folder.name}`}
                onClick={() => void loadBrowser(folder.path)}
              ><Folder size={17} /><span>{folder.name}</span></button>
            ))}
            {browser.rasters.map((raster) => (
              <label className="dataset-editor-file" key={raster.path} title={`Добавить TIFF ${raster.name}`}>
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

function vectorSnapshot(
  source: VectorSource<Feature<Geometry>>,
  crs: string,
  template: JsonObject,
  newFeatures: WeakSet<Feature<Geometry>>,
): DraftSnapshot {
  const features = source.getFeatures();
  for (const feature of features) {
    if (!feature.get(ROLE_PROPERTY)) feature.set(ROLE_PROPERTY, "positive", true);
  }
  const collection = new GeoJSON().writeFeaturesObject(features, {
    dataProjection: crs,
    featureProjection: crs,
  }) as JsonObject;
  return {
    geojson: { ...template, ...collection, crs: template.crs },
    newFeatureIndexes: features.flatMap((feature, index) => newFeatures.has(feature) ? [index] : []),
  };
}

function restoreVectorSnapshot(
  source: VectorSource<Feature<Geometry>>,
  snapshot: DraftSnapshot,
  crs: string,
  newFeaturesRef: { current: WeakSet<Feature<Geometry>> },
): void {
  const features = new GeoJSON().readFeatures(snapshot.geojson, {
    dataProjection: crs,
    featureProjection: crs,
  }) as Feature<Geometry>[];
  const newFeatures = new WeakSet<Feature<Geometry>>();
  for (const [index, feature] of features.entries()) {
    if (!feature.get(ROLE_PROPERTY)) feature.set(ROLE_PROPERTY, "positive", true);
    if (snapshot.newFeatureIndexes.includes(index)) newFeatures.add(feature);
  }
  source.clear();
  source.addFeatures(features);
  newFeaturesRef.current = newFeatures;
}

function featureStyle(
  feature: Feature<Geometry>,
  selected: boolean,
  filled: boolean,
  newFeatures: WeakSet<Feature<Geometry>>,
): Style {
  const role: Role = feature.get(ROLE_PROPERTY) === "hard_negative" ? "hard_negative" : "positive";
  const isNew = newFeatures.has(feature);
  const key = `${selected ? "selected" : role}:${isNew ? "new" : "saved"}:${filled ? "filled" : "outline"}`;
  const cached = styleCache.get(key);
  if (cached) return cached;
  const strokeColor = selected ? "#38bdf8" : role === "hard_negative" ? "#ef4444" : "#f3c623";
  const fillColor = selected
    ? "rgba(56, 189, 248, 0.18)"
    : role === "hard_negative"
      ? "rgba(239, 68, 68, 0.18)"
      : "rgba(243, 198, 35, 0.22)";
  const style = new Style({
    stroke: new Stroke({ color: strokeColor, width: selected ? 4 : 3, lineDash: isNew ? [8, 5] : undefined }),
    fill: filled ? new Fill({ color: fillColor }) : undefined,
  });
  styleCache.set(key, style);
  return style;
}

function rasterBandStyle(mode: BandMode): WebGLTileStyle {
  const [red, green, blue] = BAND_CHANNELS[mode];
  return {
    color: [
      "color",
      ["*", ["band", red], 255],
      ["*", ["band", green], 255],
      ["*", ["band", blue], 255],
    ],
  };
}

function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.tagName === "INPUT" ||
    target.tagName === "SELECT" ||
    target.tagName === "TEXTAREA"
  );
}
