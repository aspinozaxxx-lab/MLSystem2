import {
  ArrowDownNarrowWide,
  ArrowUpNarrowWide,
  Blend,
  CloudUpload,
  Eye,
  EyeOff,
  Folder,
  FolderOpen,
  Layers,
  MousePointer2,
  PaintBucket,
  PencilLine,
  Plus,
  RefreshCw,
  Trash2,
  Undo2,
} from "lucide-react";
import Feature from "ol/Feature";
import { defaults as defaultControls } from "ol/control/defaults";
import GeoJSON from "ol/format/GeoJSON";
import type Geometry from "ol/geom/Geometry";
import MultiPoint from "ol/geom/MultiPoint";
import { defaults as defaultInteractions } from "ol/interaction/defaults";
import DragBox from "ol/interaction/DragBox";
import DragPan from "ol/interaction/DragPan";
import Draw from "ol/interaction/Draw";
import Modify from "ol/interaction/Modify";
import Select from "ol/interaction/Select";
import Snap from "ol/interaction/Snap";
import WebGLTileLayer, { type Style as WebGLTileStyle } from "ol/layer/WebGLTile";
import VectorLayer from "ol/layer/Vector";
import OLMap from "ol/Map";
import GeoTIFF from "ol/source/GeoTIFF";
import VectorSource from "ol/source/Vector";
import { Circle as CircleStyle, Fill, Stroke, Style } from "ol/style";
import type { ViewOptions } from "ol/View";
import { type ChangeEvent, type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "ol/ol.css";

import { apiJson } from "./api/client";
import {
  appendHistory,
  cloneSnapshot,
  deleteEditableVertices,
  draftChanged,
  editableVertexCoordinates,
  editableVertices,
  editableVerticesInExtent,
  extendRasterResolutions,
  featureClassCounts,
  featureCounts,
  geometryInsideFootprint,
  preventMapMiddleButtonDefault,
  RASTER_CONTRAST,
  sceneClassCounts,
  sceneCounts,
  snapshotsEqual,
  sortEditorScenes,
  undoDraft,
  type DraftSnapshot,
  type DraftState,
  type EditableVertex,
  type JsonObject,
  type SortDirection,
} from "./utils/datasetEditor";

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;
type ObjectSelection = string;
type EditMode = "select" | "draw";
type BandMode = "RGB" | "NRG" | "NGB";
type VertexSelection = {
  feature: Feature<Geometry>;
  vertex: EditableVertex;
};

type EditorDataset = {
  key: string;
  name: string;
  class_key: string;
  class_name: string;
  dataset_name: string;
  imagery_type: "kanopus" | "ortho";
  scene_count: number;
  task: "binary" | "multiclass";
  object_types: EditorObjectType[];
  combined: boolean;
  source_status: "current" | "stale" | "unknown" | "unavailable";
  source_changes: string[];
  class_counts: Record<string, number>;
  hard_negative_count: number;
  primary_training_result_id: string | null;
};

type EditorObjectType = {
  id: number;
  slug: string;
  name: string;
  color: string;
  priority: number;
};

type EditorScene = {
  scene_id: string;
  annotation_name: string;
  image_name: string;
  raster_url: string;
  total_count: number;
  positive_count: number;
  hard_negative_count: number;
  class_counts: Record<string, number>;
  revision: string;
  draft: DraftSummary | null;
};

type DraftSummary = {
  annotation_name: string;
  base_revision: string;
  deleted: boolean;
  stale: boolean;
  total_count: number;
  positive_count: number;
  hard_negative_count: number;
  class_counts: Record<string, number>;
  updated_at: string;
};

type DraftInfo = DraftSummary & { geojson: JsonObject };

type SceneDetail = {
  scene: EditorScene;
  geojson: JsonObject;
  valid_data_footprint: JsonObject;
  draft: DraftInfo | null;
};
type PseudoMarkupInfo = {
  status: "unavailable" | "ready" | "queued" | "running" | "failed";
  source: "dataset" | "scene" | null;
  training_result_id: string | null;
  model_name: string | null;
  job_id: string | null;
  progress_current: number | null;
  progress_total: number | null;
  object_count: number;
  message: string | null;
  geojson: JsonObject | null;
  can_retry: boolean;
};
type SceneDraft = DraftState & {
  scene: EditorScene;
  validDataFootprint: JsonObject;
  normalized: boolean;
  baseRevision: string;
  serverSaved: DraftSnapshot;
  hasServerDraft: boolean;
  serverDraftStale: boolean;
  serverUpdatedAt: string | null;
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
type RebuildChange = {
  kind: "added" | "edited" | "deleted" | "source_added" | "source_edited" | "source_deleted";
  annotation_name: string;
  origin_key: string | null;
  detail: string | null;
};
type RebuildPreview = {
  preview_token: string;
  dataset_key: string;
  source_status: "current" | "stale" | "unknown" | "unavailable";
  source_changes: string[];
  local_changes: RebuildChange[];
  conflicts: RebuildChange[];
  replacement_scene_count: number;
  replacement_class_counts: Record<string, number>;
  replacement_hard_negative_count: number;
  warnings: string[];
};
type RebuildResult = MutationResult & {
  mode: "merge" | "replace";
  conflicts: RebuildChange[];
  warnings: string[];
};
type DraftSaveStatus = "saved" | "saving" | "unsaved" | "error";

const ROLE_PROPERTY = "_mlsystem2_role";
const CLASS_PROPERTY = "_mlsystem2_class";
const POSITIVE_COLOR = "#F3C623";
const HARD_NEGATIVE_COLOR = "#EF4444";
const BAND_CHANNELS: Record<BandMode, [number, number, number]> = {
  RGB: [1, 2, 3],
  NRG: [4, 1, 2],
  NGB: [4, 2, 3],
};
const styleCache = new Map<string, Style>();
const pseudoStyleCache = new Map<string, Style>();
const EDITABLE_VERTICES_STYLE = new Style({
  geometry: (feature) => new MultiPoint(
    editableVertexCoordinates((feature as Feature<Geometry>).getGeometry()),
  ),
  image: new CircleStyle({
    radius: 5,
    fill: new Fill({ color: "#FFFFFF" }),
    stroke: new Stroke({ color: "#0284C7", width: 2 }),
  }),
});
const SELECTED_VERTEX_IMAGE = new CircleStyle({
  radius: 7,
  fill: new Fill({ color: "#38BDF8" }),
  stroke: new Stroke({ color: "#FFFFFF", width: 2 }),
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
  const [drafts, setDrafts] = useState<DraftMap>({});
  const [role, setRole] = useState<ObjectSelection>("positive");
  const [editMode, setEditMode] = useState<EditMode>("select");
  const [sortDirection, setSortDirection] = useState<SortDirection>("descending");
  const [fillEnabled, setFillEnabled] = useState(true);
  const [annotationsVisible, setAnnotationsVisible] = useState(true);
  const [pseudoVisible, setPseudoVisible] = useState(false);
  const [pseudoMarkup, setPseudoMarkup] = useState<PseudoMarkupInfo | null>(null);
  const [pseudoRequestPending, setPseudoRequestPending] = useState(false);
  const [draftSaveStatuses, setDraftSaveStatuses] = useState<Record<string, DraftSaveStatus>>({});
  const [bandMode, setBandMode] = useState<BandMode>("RGB");
  const [bandMenuOpen, setBandMenuOpen] = useState(false);
  const [drawInProgress, setDrawInProgress] = useState(false);
  const [busy, setBusy] = useState(false);
  const [browser, setBrowser] = useState<RasterBrowser | null>(null);
  const [selectedRasters, setSelectedRasters] = useState<Set<string>>(new Set());
  const [publication, setPublication] = useState<PublicationInfo | null>(null);
  const [rebuildPreview, setRebuildPreview] = useState<RebuildPreview | null>(null);
  const mapTargetRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<OLMap | null>(null);
  const vectorSourceRef = useRef<VectorSource<Feature<Geometry>> | null>(null);
  const vectorLayerRef = useRef<VectorLayer<VectorSource<Feature<Geometry>>> | null>(null);
  const pseudoSourceRef = useRef<VectorSource<Feature<Geometry>> | null>(null);
  const pseudoLayerRef = useRef<VectorLayer<VectorSource<Feature<Geometry>>> | null>(null);
  const rasterLayerRef = useRef<WebGLTileLayer | null>(null);
  const selectRef = useRef<Select | null>(null);
  const vertexBoxRef = useRef<DragBox | null>(null);
  const modifyRef = useRef<Modify | null>(null);
  const drawRef = useRef<Draw | null>(null);
  const rasterFootprintRef = useRef<Geometry | null>(null);
  const draftsRef = useRef<DraftMap>({});
  const dirtyRef = useRef(false);
  const roleRef = useRef<ObjectSelection>(role);
  const fillEnabledRef = useRef(fillEnabled);
  const annotationsVisibleRef = useRef(annotationsVisible);
  const bandModeRef = useRef<BandMode>(bandMode);
  const activeAnnotationRef = useRef(annotationName);
  const drawInProgressRef = useRef(false);
  const selectedVerticesRef = useRef<VertexSelection[]>([]);
  const newFeaturesRef = useRef<WeakSet<Feature<Geometry>>>(new WeakSet());
  const scenesLoadRequestRef = useRef(0);
  const sceneLoadRequestRef = useRef(0);
  const pseudoLoadRequestRef = useRef(0);
  const pseudoCacheRef = useRef<Map<string, PseudoMarkupInfo>>(new Map());
  const draftSaveTimersRef = useRef<Map<string, number>>(new Map());
  const draftSaveInFlightRef = useRef<Map<string, Promise<boolean>>>(new Map());
  const datasetKeyRef = useRef(datasetKey);
  datasetKeyRef.current = datasetKey;

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.key === datasetKey) || null,
    [datasetKey, datasets],
  );
  const draftSceneNames = useMemo(() => new Set([
    ...scenes.filter((scene) => scene.draft).map((scene) => scene.annotation_name),
    ...Object.values(drafts)
      .filter((draft) => draftChanged(draft) || draft.hasServerDraft)
      .map((draft) => draft.scene.annotation_name),
  ]), [drafts, scenes]);
  const hasDirtyDrafts = draftSceneNames.size > 0;
  const dirtyDraftCount = draftSceneNames.size;
  const hasUnsavedLocalDrafts = useMemo(
    () => Object.values(drafts).some(
      (draft) => !snapshotsEqual(draft.current, draft.serverSaved),
    ) || Object.values(draftSaveStatuses).some((status) =>
      status === "saving" || status === "unsaved" || status === "error"
    ),
    [draftSaveStatuses, drafts],
  );
  const sortedScenes = useMemo(
    () => sortEditorScenes(scenes, drafts, sortDirection),
    [drafts, scenes, sortDirection],
  );
  const addedAnnotationNames = useMemo(
    () => new Set(scenes.map((scene) => scene.annotation_name.toLocaleLowerCase("ru"))),
    [scenes],
  );
  const activeDraft = annotationName ? drafts[annotationName] : undefined;
  const objectTypeChoices = useMemo(
    () => selectedDataset?.object_types || [],
    [selectedDataset],
  );
  const activeClassCounts = useMemo(
    () => activeDraft ? featureClassCounts(activeDraft.current.geojson) : {},
    [activeDraft],
  );
  const activePseudoCacheKey = useMemo(
    () => pseudoCacheKey(
      datasetKey,
      annotationName,
      selectedDataset?.primary_training_result_id || "none",
    ),
    [annotationName, datasetKey, selectedDataset?.primary_training_result_id],
  );
  const activeDraftSaveStatus: DraftSaveStatus | undefined = annotationName
    ? draftSaveStatuses[annotationName] || (
        activeDraft && !snapshotsEqual(activeDraft.current, activeDraft.serverSaved)
          ? "unsaved"
          : "saved"
      )
    : undefined;

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
    draftSaveTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    draftSaveTimersRef.current.clear();
    draftsRef.current = {};
    setDrafts({});
    setDraftSaveStatuses({});
  }, []);

  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  useEffect(() => {
    activeAnnotationRef.current = annotationName;
  }, [annotationName]);

  useEffect(() => {
    dirtyRef.current = hasUnsavedLocalDrafts;
  }, [hasUnsavedLocalDrafts]);

  useEffect(() => {
    const guard = () =>
      !dirtyRef.current ||
      window.confirm("Последние изменения ещё не сохранены на сервере. Покинуть редактор?");
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
      const requestId = ++scenesLoadRequestRef.current;
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
      if (
        !payload
        || requestId !== scenesLoadRequestRef.current
        || datasetKeyRef.current !== key
      ) return;
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
    const defaultRole = selectedDataset?.task === "multiclass"
      ? selectedDataset.object_types[0]?.slug || "positive"
      : "positive";
    roleRef.current = defaultRole;
    setRole(defaultRole);
    setRebuildPreview(null);
    void loadScenes(datasetKey);
  }, [datasetKey, loadScenes, resetDrafts, selectedDataset]);

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
        setDetail({
          scene: cached.scene,
          geojson: cached.current.geojson,
          valid_data_footprint: cached.validDataFootprint,
          draft: null,
        });
        return;
      }
      const payload = await run(() =>
        apiJson<SceneDetail>(
          `/dataset-editor/datasets/${encodeURIComponent(key)}/scenes/${encodeURIComponent(name)}`,
        ),
      );
      if (
        !payload
        || requestId !== sceneLoadRequestRef.current
        || datasetKeyRef.current !== key
        || activeAnnotationRef.current !== name
      ) return;
      const baseline = { geojson: payload.geojson, newFeatureIndexes: [], deleted: false };
      const draftSnapshot = payload.draft
        ? {
            geojson: payload.draft.geojson,
            newFeatureIndexes: draftNewFeatureIndexes(payload.geojson, payload.draft.geojson),
            deleted: payload.draft.deleted,
          }
        : baseline;
      const scene = {
        ...payload.scene,
        draft: payload.draft ? draftSummary(payload.draft) : payload.scene.draft || null,
      };
      changeDrafts((existing) => ({
        ...existing,
        [name]: {
          scene,
          validDataFootprint: payload.valid_data_footprint,
          baseline: cloneSnapshot(baseline),
          current: cloneSnapshot(draftSnapshot),
          serverSaved: cloneSnapshot(draftSnapshot),
          baseRevision: payload.draft?.base_revision || payload.scene.revision,
          hasServerDraft: Boolean(payload.draft),
          serverDraftStale: Boolean(payload.draft?.stale),
          serverUpdatedAt: payload.draft?.updated_at || null,
          history: [],
          normalized: Boolean(payload.draft),
        },
      }));
      setEditMode("select");
      setDetail({ ...payload, scene });
    },
    [changeDrafts, run],
  );

  useEffect(() => {
    void loadScene(datasetKey, annotationName);
  }, [annotationName, datasetKey, loadScene]);

  const saveDraftNow = useCallback((name: string): Promise<boolean> => {
    const key = datasetKey;
    const operationKey = `${key}\u0000${name}`;
    const existing = draftSaveInFlightRef.current.get(operationKey);
    if (existing) return existing;
    const operation = (async () => {
      const draft = draftsRef.current[name];
      if (
        !key ||
        !draft ||
        snapshotsEqual(draft.current, draft.serverSaved)
      ) return true;

      if (!draftChanged(draft)) {
        setDraftSaveStatuses((current) => ({ ...current, [name]: "saving" }));
        try {
          await apiJson<{ deleted_count: number }>(
            `/dataset-editor/datasets/${encodeURIComponent(key)}/drafts/${encodeURIComponent(name)}`,
            { method: "DELETE" },
          );
          if (datasetKeyRef.current !== key) return true;
          changeDrafts((current) => {
            const currentDraft = current[name];
            if (!currentDraft) return current;
            const hasNewerChanges = !snapshotsEqual(currentDraft.current, draft.current);
            return {
              ...current,
              [name]: {
                ...currentDraft,
                scene: { ...currentDraft.scene, draft: null },
                serverSaved: cloneSnapshot(draft.baseline),
                hasServerDraft: false,
                serverDraftStale: false,
                serverUpdatedAt: null,
                current: hasNewerChanges
                  ? currentDraft.current
                  : cloneSnapshot(draft.baseline),
              },
            };
          });
          setScenes((current) => current.map((scene) =>
            scene.annotation_name === name ? { ...scene, draft: null } : scene
          ));
          const latest = draftsRef.current[name];
          setDraftSaveStatuses((current) => ({
            ...current,
            [name]: latest && !snapshotsEqual(latest.current, latest.serverSaved)
              ? "unsaved"
              : "saved",
          }));
          return true;
        } catch {
          if (datasetKeyRef.current === key) {
            setDraftSaveStatuses((current) => ({ ...current, [name]: "error" }));
          }
          return false;
        }
      }

      const sent = cloneSnapshot(draft.current);
      setDraftSaveStatuses((current) => ({ ...current, [name]: "saving" }));
      try {
        const saved = await apiJson<DraftInfo>(
          `/dataset-editor/datasets/${encodeURIComponent(key)}/drafts/${encodeURIComponent(name)}`,
          {
            method: "PUT",
            body: {
              base_revision: draft.baseRevision,
              geojson: sent.geojson,
              deleted: Boolean(sent.deleted),
            },
          },
        );
        if (datasetKeyRef.current !== key) return true;
        const latest = draftsRef.current[name];
        const hasNewerChanges = Boolean(latest && !snapshotsEqual(latest.current, sent));
        const savedSnapshot = {
          geojson: saved.geojson,
          newFeatureIndexes: [...sent.newFeatureIndexes],
          deleted: saved.deleted,
        };
        changeDrafts((current) => {
          const currentDraft = current[name];
          if (!currentDraft) return current;
          const summary = draftSummary(saved);
          return {
            ...current,
            [name]: {
              ...currentDraft,
              scene: { ...currentDraft.scene, draft: summary },
              current: hasNewerChanges
                ? currentDraft.current
                : cloneSnapshot(savedSnapshot),
              serverSaved: cloneSnapshot(savedSnapshot),
              baseRevision: saved.base_revision,
              hasServerDraft: true,
              serverDraftStale: saved.stale,
              serverUpdatedAt: saved.updated_at,
              normalized: true,
            },
          };
        });
        setScenes((current) => current.map((scene) =>
          scene.annotation_name === name
            ? { ...scene, draft: draftSummary(saved) }
            : scene
        ));
        setDraftSaveStatuses((current) => ({
          ...current,
          [name]: hasNewerChanges ? "unsaved" : "saved",
        }));
        return true;
      } catch {
        if (datasetKeyRef.current === key) {
          setDraftSaveStatuses((current) => ({ ...current, [name]: "error" }));
        }
        return false;
      }
    })();
    draftSaveInFlightRef.current.set(operationKey, operation);
    void operation.finally(() => {
      if (draftSaveInFlightRef.current.get(operationKey) === operation) {
        draftSaveInFlightRef.current.delete(operationKey);
        const latest = draftsRef.current[name];
        if (
          datasetKeyRef.current === key &&
          latest &&
          !snapshotsEqual(latest.current, latest.serverSaved) &&
          !draftSaveTimersRef.current.has(name)
        ) {
          const timer = window.setTimeout(() => {
            draftSaveTimersRef.current.delete(name);
            void saveDraftNow(name);
          }, 1000);
          draftSaveTimersRef.current.set(name, timer);
        }
      }
    });
    return operation;
  }, [changeDrafts, datasetKey]);

  useEffect(() => {
    const nextStatuses: Record<string, DraftSaveStatus> = {};
    for (const [name, draft] of Object.entries(drafts)) {
      if (snapshotsEqual(draft.current, draft.serverSaved)) continue;
      nextStatuses[name] = "unsaved";
      if (
        draftSaveTimersRef.current.has(name) ||
        draftSaveInFlightRef.current.has(`${datasetKey}\u0000${name}`)
      ) continue;
      const timer = window.setTimeout(() => {
        draftSaveTimersRef.current.delete(name);
        void saveDraftNow(name);
      }, 1000);
      draftSaveTimersRef.current.set(name, timer);
    }
    if (Object.keys(nextStatuses).length) {
      setDraftSaveStatuses((current) => {
        const updated = { ...current };
        for (const [name, status] of Object.entries(nextStatuses)) {
          if (updated[name] !== "saving" && updated[name] !== "error") {
            updated[name] = status;
          }
        }
        return updated;
      });
    }
    setDraftSaveStatuses((current) => {
      const updated = { ...current };
      let changed = false;
      for (const [name, status] of Object.entries(updated)) {
        const draft = drafts[name];
        if (!draft) {
          delete updated[name];
          changed = true;
        } else if (
          status !== "saving" &&
          snapshotsEqual(draft.current, draft.serverSaved) &&
          status !== "saved"
        ) {
          updated[name] = "saved";
          changed = true;
        }
      }
      return changed ? updated : current;
    });
  }, [drafts, saveDraftNow]);

  useEffect(() => () => {
    draftSaveTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    draftSaveTimersRef.current.clear();
  }, []);

  const flushDrafts = useCallback(async (): Promise<boolean> => {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const names = Object.entries(draftsRef.current)
        .filter(([, draft]) => !snapshotsEqual(draft.current, draft.serverSaved))
        .map(([name]) => name);
      if (!names.length) return true;
      names.forEach((name) => {
        const timer = draftSaveTimersRef.current.get(name);
        if (timer !== undefined) window.clearTimeout(timer);
        draftSaveTimersRef.current.delete(name);
      });
      const results = await Promise.all(names.map(saveDraftNow));
      if (results.some((value) => !value)) return false;
    }
    return false;
  }, [saveDraftNow]);

  useEffect(() => {
    pseudoLoadRequestRef.current += 1;
    setPseudoVisible(false);
    setPseudoRequestPending(false);
    setPseudoMarkup(pseudoCacheRef.current.get(activePseudoCacheKey) || null);
  }, [activePseudoCacheKey]);

  const loadPseudoMarkup = useCallback(
    async (ensure: boolean, retry = false) => {
      if (!datasetKey || !annotationName) return;
      const requestId = ++pseudoLoadRequestRef.current;
      const cacheKey = activePseudoCacheKey;
      const path = `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/scenes/${encodeURIComponent(annotationName)}/pseudo-markup${retry ? "?retry=true" : ""}`;
      setPseudoRequestPending(true);
      try {
        const payload = await apiJson<PseudoMarkupInfo>(
          path,
          ensure ? { method: "POST" } : undefined,
        );
        pseudoCacheRef.current.set(cacheKey, payload);
        if (requestId === pseudoLoadRequestRef.current) setPseudoMarkup(payload);
      } catch (error) {
        if (requestId !== pseudoLoadRequestRef.current) return;
        const failed: PseudoMarkupInfo = {
          status: "failed",
          source: null,
          training_result_id: null,
          model_name: null,
          job_id: null,
          progress_current: null,
          progress_total: null,
          object_count: 0,
          message: error instanceof Error ? error.message : "Не удалось получить псевдоразметку.",
          geojson: null,
          can_retry: true,
        };
        pseudoCacheRef.current.set(cacheKey, failed);
        setPseudoMarkup(failed);
      } finally {
        if (requestId === pseudoLoadRequestRef.current) setPseudoRequestPending(false);
      }
    },
    [activePseudoCacheKey, annotationName, datasetKey],
  );

  const pollPseudoMarkup = useCallback(async (jobId: string) => {
    const requestId = pseudoLoadRequestRef.current;
    const cacheKey = activePseudoCacheKey;
    try {
      const payload = await apiJson<PseudoMarkupInfo>(
        `/dataset-editor/pseudo-markup/${encodeURIComponent(jobId)}`,
      );
      pseudoCacheRef.current.set(cacheKey, payload);
      if (requestId === pseudoLoadRequestRef.current) setPseudoMarkup(payload);
    } catch {
      // Временная ошибка polling не должна создавать новое задание или стирать готовый кеш.
    }
  }, [activePseudoCacheKey]);

  useEffect(() => {
    if (
      !pseudoVisible ||
      !pseudoMarkup?.job_id ||
      (pseudoMarkup.status !== "queued" && pseudoMarkup.status !== "running")
    ) return;
    const timer = window.setInterval(
      () => void pollPseudoMarkup(pseudoMarkup.job_id as string),
      1500,
    );
    return () => window.clearInterval(timer);
  }, [pollPseudoMarkup, pseudoMarkup?.job_id, pseudoMarkup?.status, pseudoVisible]);

  const togglePseudoMarkup = () => {
    if (pseudoVisible) {
      setPseudoVisible(false);
      return;
    }
    setPseudoVisible(true);
    if (!pseudoMarkup) void loadPseudoMarkup(true);
  };

  const captureActiveSnapshot = useCallback((): DraftSnapshot | null => {
    const name = activeAnnotationRef.current;
    const draft = draftsRef.current[name];
    const source = vectorSourceRef.current;
    if (!draft || !source) return null;
    return {
      ...vectorSnapshot(
        source,
        geojsonCrs(draft.current.geojson),
        draft.current.geojson,
        newFeaturesRef.current,
      ),
      deleted: Boolean(draft.current.deleted),
    };
  }, []);

  const recordCurrentChange = useCallback(
    (before: DraftSnapshot) => {
      const name = activeAnnotationRef.current;
      const current = captureActiveSnapshot();
      if (!name || !current || snapshotsEqual(before, current)) return;
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

  const setSelectedVertices = useCallback((selections: VertexSelection[]) => {
    const changedFeatures = new Set<Feature<Geometry>>([
      ...selectedVerticesRef.current.map((selection) => selection.feature),
      ...selections.map((selection) => selection.feature),
    ]);
    selectedVerticesRef.current = selections;
    changedFeatures.forEach((feature) => feature.changed());
    mapRef.current?.render();
  }, []);

  useEffect(() => {
    const target = mapTargetRef.current;
    if (!target || !detail) return;
    const name = detail.scene.annotation_name;
    const draft = draftsRef.current[name];
    if (!draft) return;
    const format = new GeoJSON();
    const crs = geojsonCrs(draft.current.geojson);
    rasterFootprintRef.current = format.readGeometry(detail.valid_data_footprint, {
      dataProjection: crs,
      featureProjection: crs,
    });
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
      interpolate: false,
      transition: 0,
    });
    const rasterLayer = new WebGLTileLayer({
      className: "dataset-editor-raster-layer",
      source: rasterSource,
      style: rasterBandStyle(
        selectedDataset?.imagery_type === "kanopus" ? bandModeRef.current : "RGB",
      ),
    });
    const pseudoSource = new VectorSource<Feature<Geometry>>();
    const pseudoLayer = new VectorLayer({
      source: pseudoSource,
      visible: false,
      style: (feature) => pseudoMarkupStyle(
        feature as Feature<Geometry>,
        objectTypeChoices,
      ),
    });
    const vectorLayer = new VectorLayer({
      source: vectorSource,
      visible: annotationsVisibleRef.current,
      style: (feature) =>
        featureStyle(
          feature as Feature<Geometry>,
          false,
          fillEnabledRef.current,
          newFeaturesRef.current,
          objectTypeChoices,
        ),
    });
    const select = new Select({
      layers: [vectorLayer],
      style: (feature) =>
        selectedFeatureStyles(
          feature as Feature<Geometry>,
          fillEnabledRef.current,
          newFeaturesRef.current,
          objectTypeChoices,
          selectedVerticesRef.current,
        ),
    });
    const modify = new Modify({
      features: select.getFeatures(),
      deleteCondition: () => false,
    });
    const vertexBox = new DragBox({
      className: "dataset-editor-vertex-box",
      minArea: 16,
      condition: (event) => {
        const originalEvent = event.originalEvent;
        return originalEvent instanceof PointerEvent &&
          originalEvent.pointerType !== "touch" &&
          originalEvent.button === 0 &&
          !originalEvent.altKey &&
          !originalEvent.ctrlKey &&
          !originalEvent.metaKey &&
          !originalEvent.shiftKey;
      },
    });
    const middleDragPan = new DragPan({
      condition: (event) => {
        const originalEvent = event.originalEvent;
        if (!(originalEvent instanceof PointerEvent)) return false;
        if (originalEvent.pointerType === "touch") return true;
        return preventMapMiddleButtonDefault(originalEvent);
      },
    });
    const snap = new Snap({ source: vectorSource });
    const draw = new Draw({ source: vectorSource, type: "Polygon" });
    draw.setActive(false);
    const geometryBackups = new Map<Feature<Geometry>, Geometry>();
    let drawBefore: DraftSnapshot | null = null;
    let modifyBefore: DraftSnapshot | null = null;
    let drawCommitTimer: number | null = null;
    let vertexSelectionTimer: number | null = null;

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
        serverSaved: cloneSnapshot(normalized),
        normalized: true,
      }));
    }

    const selectCurrentModifyVertex = () => {
      const coordinate = modify.getPoint();
      if (!coordinate || !modify.canRemovePoint()) {
        setSelectedVertices([]);
        return;
      }
      for (const feature of select.getFeatures().getArray() as Feature<Geometry>[]) {
        const vertex = editableVertices(feature.getGeometry()).find((candidate) =>
          coordinatesEqual(candidate.coordinate, coordinate),
        );
        if (vertex) {
          setSelectedVertices([{ feature, vertex }]);
          return;
        }
      }
      setSelectedVertices([]);
    };

    const selectCurrentModifyVertexAfterInteractions = () => {
      // Map вызывает свои singleclick-listeners до interactions; ждём обновления точки Modify.
      if (vertexSelectionTimer !== null) window.clearTimeout(vertexSelectionTimer);
      vertexSelectionTimer = window.setTimeout(() => {
        vertexSelectionTimer = null;
        selectCurrentModifyVertex();
      }, 0);
    };

    const syncRoleFromFeature = (selected: Feature<Geometry> | undefined) => {
      if (!selected) return;
      const selectedRole = selected.get(ROLE_PROPERTY) === "hard_negative"
        ? "hard_negative"
        : typeof selected.get(CLASS_PROPERTY) === "string"
          ? String(selected.get(CLASS_PROPERTY))
          : "positive";
      roleRef.current = selectedRole;
      setRole(selectedRole);
    };

    select.on("select", (event) => {
      setSelectedVertices([]);
      syncRoleFromFeature(event.selected[0] as Feature<Geometry> | undefined);
    });
    vertexBox.on("boxstart", () => {
      setSelectedVertices([]);
    });
    vertexBox.on("boxend", () => {
      const extent = vertexBox.getGeometry().getExtent();
      const selections = vectorSource.getFeatures().flatMap((feature) =>
        editableVerticesInExtent(feature.getGeometry(), extent).map((vertex) => ({
          feature,
          vertex,
        })),
      );
      const features = [...new Set(selections.map((selection) => selection.feature))];
      const selectedFeatures = select.getFeatures();
      selectedFeatures.clear();
      selectedFeatures.extend(features);
      setSelectedVertices(selections);
      syncRoleFromFeature(features[0]);
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
      if (event.feature.getId() === undefined) event.feature.setId(crypto.randomUUID());
      applyObjectSelection(event.feature, roleRef.current, selectedDataset);
      if (!geometryInsideFootprint(event.feature.getGeometry(), rasterFootprintRef.current)) {
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
        !geometryInsideFootprint(feature.getGeometry(), rasterFootprintRef.current),
      );
      if (outside) {
        for (const [feature, geometry] of geometryBackups) feature.setGeometry(geometry);
        modifyBefore = null;
        setSelectedVertices([]);
        window.alert("Геометрия не может выходить за границы снимка.");
        return;
      }
      const before = modifyBefore;
      modifyBefore = null;
      if (before) recordCurrentChange(before);
      selectCurrentModifyVertex();
    });

    const map = new OLMap({
      target,
      controls: defaultControls({ zoom: false }),
      layers: [rasterLayer, pseudoLayer, vectorLayer],
      interactions: defaultInteractions({ dragPan: false, shiftDragZoom: false }),
      view: rasterViewWithOverzoom(rasterSource),
    });
    const preventMiddleButtonBrowserAction = (event: MouseEvent) => {
      preventMapMiddleButtonDefault(event);
    };
    target.addEventListener("mousedown", preventMiddleButtonBrowserAction, true);
    target.addEventListener("auxclick", preventMiddleButtonBrowserAction, true);
    map.addInteraction(middleDragPan);
    map.addInteraction(select);
    map.addInteraction(vertexBox);
    map.addInteraction(modify);
    map.addInteraction(draw);
    map.addInteraction(snap);
    map.on("singleclick", selectCurrentModifyVertexAfterInteractions);
    mapRef.current = map;
    pseudoSourceRef.current = pseudoSource;
    pseudoLayerRef.current = pseudoLayer;
    vectorLayerRef.current = vectorLayer;
    rasterLayerRef.current = rasterLayer;
    selectRef.current = select;
    vertexBoxRef.current = vertexBox;
    modifyRef.current = modify;
    drawRef.current = draw;

    return () => {
      if (drawCommitTimer !== null) window.clearTimeout(drawCommitTimer);
      if (vertexSelectionTimer !== null) window.clearTimeout(vertexSelectionTimer);
      target.removeEventListener("mousedown", preventMiddleButtonBrowserAction, true);
      target.removeEventListener("auxclick", preventMiddleButtonBrowserAction, true);
      map.un("singleclick", selectCurrentModifyVertexAfterInteractions);
      map.setTarget(undefined);
      mapRef.current = null;
      pseudoSourceRef.current = null;
      pseudoLayerRef.current = null;
      vectorSourceRef.current = null;
      vectorLayerRef.current = null;
      rasterLayerRef.current = null;
      selectRef.current = null;
      vertexBoxRef.current = null;
      modifyRef.current = null;
      drawRef.current = null;
      rasterFootprintRef.current = null;
      selectedVerticesRef.current = [];
      newFeaturesRef.current = new WeakSet();
      setDrawingState(false);
    };
  }, [
    captureActiveSnapshot,
    detail,
    recordCurrentChange,
    selectedDataset?.imagery_type,
    selectedDataset,
    setSelectedVertices,
    setDrawingState,
    updateDraft,
    objectTypeChoices,
  ]);

  useEffect(() => {
    const source = pseudoSourceRef.current;
    const layer = pseudoLayerRef.current;
    if (!source || !layer || !detail) return;
    source.clear();
    if (pseudoMarkup?.status === "ready" && pseudoMarkup.geojson) {
      const rasterCrs = geojsonCrs(detail.geojson);
      const pseudoCrs = geojsonCrs(pseudoMarkup.geojson);
      const features = new GeoJSON().readFeatures(pseudoMarkup.geojson, {
        dataProjection: pseudoCrs,
        featureProjection: rasterCrs,
      }) as Feature<Geometry>[];
      source.addFeatures(features);
    }
    layer.setVisible(pseudoVisible && pseudoMarkup?.status === "ready");
  }, [detail, pseudoMarkup, pseudoVisible]);

  useEffect(() => {
    const sceneDeleted = Boolean(activeDraft?.current.deleted);
    const drawing = annotationsVisible && !sceneDeleted && editMode === "draw";
    drawRef.current?.setActive(drawing);
    selectRef.current?.setActive(annotationsVisible && !sceneDeleted && !drawing);
    vertexBoxRef.current?.setActive(annotationsVisible && !sceneDeleted && !drawing);
    modifyRef.current?.setActive(annotationsVisible && !sceneDeleted && !drawing);
    vectorLayerRef.current?.setVisible(annotationsVisible && !sceneDeleted);
    if (drawing || !annotationsVisible || sceneDeleted) {
      setSelectedVertices([]);
      selectRef.current?.getFeatures().clear();
    }
  }, [activeDraft?.current.deleted, annotationsVisible, detail, editMode, setSelectedVertices]);

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
    !dirtyRef.current || window.confirm(
      "Последние изменения ещё не сохранены на сервере. Перейти и оставить их несохранёнными?",
    );

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

  const changeRole = (nextRole: ObjectSelection) => {
    const selected = selectRef.current?.getFeatures().getArray() || [];
    if (selected.length && selected.some((feature) => !matchesObjectSelection(feature, nextRole))) {
      const before = captureActiveSnapshot();
      selected.forEach((feature) => applyObjectSelection(feature, nextRole, selectedDataset));
      vectorSourceRef.current?.changed();
      mapRef.current?.render();
      if (before) recordCurrentChange(before);
    }
    roleRef.current = nextRole;
    setRole(nextRole);
  };

  const previewDatasetRebuild = async () => {
    if (!datasetKey || hasDirtyDrafts) return;
    setBusy(true);
    const preview = await run(() =>
      apiJson<RebuildPreview>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/rebuild/preview`,
        { method: "POST" },
      ),
    );
    setBusy(false);
    if (preview) setRebuildPreview(preview);
  };

  const applyDatasetRebuild = async (mode: "merge" | "replace") => {
    if (!rebuildPreview) return;
    setBusy(true);
    const result = await run(() =>
      apiJson<RebuildResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/rebuild`,
        {
          method: "POST",
          body: { preview_token: rebuildPreview.preview_token, mode },
        },
      ),
    );
    setBusy(false);
    if (!result) return;
    setRebuildPreview(null);
    resetDrafts();
    setPublication({ commit: result.commit, live_commit: null, status: result.publication_status });
    await loadDatasets();
    await loadScenes(datasetKey);
  };

  const deleteSelected = () => {
    const selected = selectRef.current?.getFeatures();
    const source = vectorSourceRef.current;
    if (!selected || !source || selected.getLength() === 0) return;
    const before = captureActiveSnapshot();
    setSelectedVertices([]);
    selected.getArray().forEach((feature) => source.removeFeature(feature));
    selected.clear();
    if (before) recordCurrentChange(before);
  };

  const restoreActiveSnapshot = useCallback((snapshot: DraftSnapshot) => {
    const source = vectorSourceRef.current;
    if (!source) return;
    setSelectedVertices([]);
    restoreVectorSnapshot(source, snapshot, geojsonCrs(snapshot.geojson), newFeaturesRef);
    selectRef.current?.getFeatures().clear();
    vectorLayerRef.current?.changed();
    mapRef.current?.render();
  }, [setSelectedVertices]);

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

  const deleteSelectedVertices = useCallback((): boolean => {
    const selections = selectedVerticesRef.current;
    const modify = modifyRef.current;
    const source = vectorSourceRef.current;
    if (!selections.length || !modify?.getActive() || !source) return false;

    const before = captureActiveSnapshot();
    const selectionsByFeature = new Map<Feature<Geometry>, EditableVertex[]>();
    for (const selection of selections) {
      const vertices = selectionsByFeature.get(selection.feature) || [];
      vertices.push(selection.vertex);
      selectionsByFeature.set(selection.feature, vertices);
    }

    let removedCount = 0;
    let blockedRingCount = 0;
    for (const [feature, vertices] of selectionsByFeature) {
      const geometry = feature.getGeometry();
      if (!geometry) continue;
      const result = deleteEditableVertices(geometry, vertices);
      removedCount += result.removedCount;
      blockedRingCount += result.blockedRingCount;
      if (result.removedCount) feature.setGeometry(result.geometry);
    }

    setSelectedVertices([]);
    if (removedCount) {
      source.changed();
      mapRef.current?.render();
      if (before) recordCurrentChange(before);
    }
    if (blockedRingCount) {
      window.alert(
        `В ${blockedRingCount} кольцах вершины не удалены: в каждом кольце должно остаться не менее трёх вершин.`,
      );
    }
    return removedCount > 0;
  }, [captureActiveSnapshot, recordCurrentChange, setSelectedVertices]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTextInput(event.target)) return;
      if (event.key === "Delete" || event.key === "Backspace") {
        if (!selectedVerticesRef.current.length) return;
        event.preventDefault();
        deleteSelectedVertices();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === "z") {
        event.preventDefault();
        undoCurrent();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteSelectedVertices, undoCurrent]);

  const toggleFill = () => {
    const next = !fillEnabledRef.current;
    fillEnabledRef.current = next;
    setFillEnabled(next);
    vectorLayerRef.current?.changed();
    mapRef.current?.render();
  };

  const toggleAnnotations = () => {
    const next = !annotationsVisibleRef.current;
    if (!next) {
      if (drawInProgressRef.current) drawRef.current?.abortDrawing();
      setSelectedVertices([]);
      selectRef.current?.getFeatures().clear();
    }
    annotationsVisibleRef.current = next;
    setAnnotationsVisible(next);
    vectorLayerRef.current?.setVisible(next);
    mapRef.current?.render();
  };

  const selectBandMode = (next: BandMode) => {
    bandModeRef.current = next;
    setBandMode(next);
    setBandMenuOpen(false);
  };

  const discardAllDrafts = async () => {
    if (!datasetKey || !hasDirtyDrafts) return;
    if (!window.confirm(
      "Удалить все ваши черновики этого датасета и вернуться к опубликованной разметке MLMarkup?",
    )) return;
    setBusy(true);
    draftSaveTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    draftSaveTimersRef.current.clear();
    await Promise.allSettled([...draftSaveInFlightRef.current.values()]);
    draftSaveTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    draftSaveTimersRef.current.clear();
    const result = await run(() =>
      apiJson<{ deleted_count: number }>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/drafts`,
        { method: "DELETE" },
      ),
    );
    setBusy(false);
    if (!result) return;
    const active = activeAnnotationRef.current;
    resetDrafts();
    setDetail(null);
    await loadScenes(datasetKey, active);
    if (active) await loadScene(datasetKey, active);
  };

  const publish = async () => {
    if (!datasetKey || !hasDirtyDrafts || drawInProgressRef.current) return;
    setBusy(true);
    const saved = await flushDrafts();
    if (!saved) {
      setBusy(false);
      window.alert("Не удалось сохранить часть черновиков. Публикация не выполнена.");
      return;
    }
    const result = await run(() =>
      apiJson<MutationResult>(
        `/dataset-editor/datasets/${encodeURIComponent(datasetKey)}/drafts/publish`,
        { method: "POST" },
      ),
    );
    setBusy(false);
    if (!result) return;
    const active = activeAnnotationRef.current;
    resetDrafts();
    setDetail(null);
    setPublication({
      commit: result.commit,
      live_commit: null,
      status: result.publication_status,
    });
    await loadScenes(datasetKey, active);
    if (active) await loadScene(datasetKey, active);
  };

  const removeScene = async () => {
    const draft = draftsRef.current[activeAnnotationRef.current];
    if (!draft || drawInProgressRef.current) return;
    const deleting = !draft.current.deleted;
    if (deleting && !window.confirm(
      `Пометить снимок ${draft.scene.image_name} на удаление? Файл останется в MLMarkup до публикации, действие можно отменить.`,
    )) return;
    const before = captureActiveSnapshot() || cloneSnapshot(draft.current);
    updateDraft(draft.scene.annotation_name, (current) => ({
      ...current,
      current: { ...cloneSnapshot(before), deleted: deleting },
      history: appendHistory(current.history, before),
      normalized: true,
    }));
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
          <p>Один GeoJSON на снимок · черновики сохраняются на сервере и публикуются отдельно</p>
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
        {selectedDataset?.combined ? (
          <button
            className={selectedDataset.source_status === "stale" ? "danger" : "secondary"}
            type="button"
            disabled={busy || hasDirtyDrafts}
            title={hasDirtyDrafts
              ? "Сначала опубликуйте или отмените черновики"
              : "Сравнить исходные main-датасеты и безопасно пересобрать комбинированный датасет"}
            onClick={() => void previewDatasetRebuild()}
          >
            <RefreshCw size={16} />
            {selectedDataset.source_status === "stale" ? "Источники изменились" : "Пересобрать"}
          </button>
        ) : null}
      </section>

      {selectedDataset?.combined && selectedDataset.source_status === "stale" ? (
        <section className="panel dataset-editor-source-warning" role="status">
          <strong>Комбинированный датасет устарел.</strong>
          <span>{selectedDataset.source_changes.join(" · ") || "Изменились исходные папки main."}</span>
        </section>
      ) : null}

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
            <div className="button-row dataset-editor-draft-actions">
              <button
                className="ghost"
                type="button"
                disabled={!hasDirtyDrafts || busy}
                title="Удалить ваши черновики и вернуться к опубликованной разметке MLMarkup"
                onClick={() => void discardAllDrafts()}
              >
                <Undo2 size={15} /> Отменить
              </button>
            </div>
            <div className="dataset-editor-scene-list">
              {sortedScenes.map((scene) => {
                const draft = drafts[scene.annotation_name];
                const summary = scene.draft;
                const counts = draft
                  ? sceneCounts(scene, draft)
                  : summary
                    ? {
                        total: summary.total_count,
                        positive: summary.positive_count,
                        hardNegative: summary.hard_negative_count,
                      }
                    : sceneCounts(scene, undefined);
                const classCounts = draft
                  ? sceneClassCounts(scene, draft)
                  : summary?.class_counts || sceneClassCounts(scene, undefined);
                const changed = Boolean(summary || (draft && (draftChanged(draft) || draft.hasServerDraft)));
                const deleted = Boolean(draft?.current.deleted ?? summary?.deleted);
                const countDescription = selectedDataset?.task === "multiclass"
                  ? objectTypeChoices
                    .map((item) => `${item.name}: ${classCounts[item.slug] || 0}`)
                    .join(", ")
                  : `positive: ${counts.positive}`;
                return (
                  <button
                    className={`${scene.annotation_name === annotationName ? "active" : ""}${deleted ? " pending-delete" : ""}`}
                    type="button"
                    key={scene.annotation_name}
                    title={`Открыть снимок ${scene.image_name}. ${counts.total} объектов: ${countDescription}, hard negative: ${counts.hardNegative}${changed ? ". Есть неопубликованные изменения" : ""}`}
                    onClick={() => selectScene(scene.annotation_name)}
                  >
                    <span className="dataset-editor-scene-name">
                      <strong>{scene.image_name}</strong>
                      {deleted ? <span className="badge warning">к удалению</span> : null}
                      {changed ? <i className="dataset-editor-dirty-dot" aria-label="Есть неопубликованные изменения" /> : null}
                    </span>
                    <span className="dataset-editor-scene-counts" aria-hidden={deleted}>
                      {selectedDataset?.task === "multiclass"
                        ? objectTypeChoices.map((item) => (
                          <i key={item.slug} style={{ color: item.color }}>
                            {item.name}: <strong>{classCounts[item.slug] || 0}</strong>
                          </i>
                        ))
                        : (
                          <i style={{ color: POSITIVE_COLOR }}>
                            Разметка: <strong>{counts.positive}</strong>
                          </i>
                        )}
                      <i style={{ color: HARD_NEGATIVE_COLOR }}>
                        Hard negative: <strong>{counts.hardNegative}</strong>
                      </i>
                    </span>
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
                    <small className={`dataset-editor-draft-status ${activeDraftSaveStatus || "saved"}`}>
                      {activeDraftSaveStatus === "saving"
                        ? "Черновик сохраняется…"
                        : activeDraftSaveStatus === "unsaved"
                          ? "Есть несохранённые изменения"
                          : activeDraftSaveStatus === "error"
                            ? "Ошибка сохранения черновика"
                            : activeDraft.hasServerDraft
                              ? "Черновик сохранён на сервере"
                              : "Опубликованная разметка"}
                    </small>
                    {activeDraft.serverDraftStale ? (
                      <small className="dataset-editor-draft-status error">
                        MLMarkup изменился после создания черновика. Перед публикацией потребуется обновить правки.
                      </small>
                    ) : null}
                    {activeDraft.current.deleted ? (
                      <small className="dataset-editor-draft-status error">
                        Снимок помечен на удаление. Он исчезнет из MLMarkup только после публикации.
                      </small>
                    ) : null}
                  </span>
                  <div className="dataset-editor-map-actions">
                    <div className="dataset-editor-mode-toggle" role="group" aria-label="Режим редактирования">
                      <button
                        className={`${editMode === "select" ? "primary" : "secondary"} icon-button dataset-editor-icon-button`}
                        type="button"
                        disabled={Boolean(activeDraft.current.deleted)}
                        aria-label="Выбор и правка полигонов"
                        aria-pressed={editMode === "select"}
                        title="Выбор и правка: протяните рамку левой кнопкой, затем удалите выбранные вершины клавишей Del"
                        onClick={() => setEditMode("select")}
                      >
                        <MousePointer2 size={17} />
                      </button>
                      <button
                        className={`${editMode === "draw" ? "primary" : "secondary"} icon-button dataset-editor-icon-button`}
                        type="button"
                        disabled={Boolean(activeDraft.current.deleted)}
                        aria-label="Нарисовать новый полигон"
                        aria-pressed={editMode === "draw"}
                        title="Новый полигон: ставьте вершины кликами, завершите двойным кликом"
                        onClick={() => setEditMode("draw")}
                      >
                        <PencilLine size={17} />
                      </button>
                    </div>
                    {selectedDataset?.task === "multiclass" ? (
                      <div className="dataset-editor-object-switch" role="group" aria-label="Тип объекта">
                        {objectTypeChoices.map((item) => (
                          <button
                            type="button"
                            key={item.slug}
                            className={role === item.slug ? "active" : ""}
                            aria-pressed={role === item.slug}
                            title={`Тип объекта: ${item.name}`}
                            style={{ "--object-color": item.color } as CSSProperties}
                            onClick={() => changeRole(item.slug)}
                          >
                            <span className="dataset-editor-color-dot" />{item.name}
                          </button>
                        ))}
                        <button
                          type="button"
                          className={role === "hard_negative" ? "active" : ""}
                          aria-pressed={role === "hard_negative"}
                          title="Общий hard negative / фон"
                          style={{ "--object-color": HARD_NEGATIVE_COLOR } as CSSProperties}
                          onClick={() => changeRole("hard_negative")}
                        >
                          <span className="dataset-editor-color-dot" />Hard negative
                        </button>
                      </div>
                    ) : (
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
                    )}
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
                      className={`${activeDraft.current.deleted ? "secondary" : "danger"} icon-button dataset-editor-icon-button`}
                      type="button"
                      disabled={busy || drawInProgress}
                      aria-label={activeDraft.current.deleted ? "Отменить удаление снимка" : "Удалить снимок из датасета"}
                      title={activeDraft.current.deleted ? "Отменить удаление снимка" : "Пометить снимок на удаление; действие сохраняется в черновике"}
                      onClick={() => void removeScene()}
                    >
                      {activeDraft.current.deleted ? <Undo2 size={17} /> : <Trash2 size={17} />}
                    </button>
                  </div>
                </div>
                <div className="dataset-editor-help">
                  <MousePointer2 size={14} /> Левая кнопка — рамка выбора вершин, Del — удалить выбранные; клик по ребру — новая вершина, зажатое колесо — перемещение, Ctrl+Z — отмена.
                </div>
                {selectedDataset?.task === "multiclass" ? (
                  <div className="dataset-editor-legend" aria-label="Легенда типов объектов">
                    {objectTypeChoices.map((item) => (
                      <span key={item.slug}>
                        <i style={{ backgroundColor: item.color }} />
                        {item.name}: <strong>{activeClassCounts[item.slug] || 0}</strong>
                      </span>
                    ))}
                    <span>
                      <i style={{ backgroundColor: HARD_NEGATIVE_COLOR }} />
                      Hard negative: <strong>{activeDraft ? featureCounts(activeDraft.current.geojson).hardNegative : 0}</strong>
                    </span>
                  </div>
                ) : null}
                <div className="dataset-editor-map-shell">
                  <div className="dataset-editor-map" ref={mapTargetRef} />
                  <div className="dataset-editor-map-controls">
                    <button
                      className={`${pseudoVisible ? "primary" : "secondary"} icon-button dataset-editor-map-control`}
                      type="button"
                      disabled={pseudoRequestPending}
                      aria-label={pseudoVisible ? "Скрыть псевдоразметку основной сети" : "Показать псевдоразметку основной сети"}
                      aria-pressed={pseudoVisible}
                      title={pseudoVisible
                        ? "Скрыть псевдоразметку текущей основной сети"
                        : "Показать псевдоразметку текущей основной сети; если её нет, запустить срочный инференс по снимку"}
                      onClick={togglePseudoMarkup}
                    >
                      <Layers size={17} />
                    </button>
                    <button
                      className={`${annotationsVisible ? "primary" : "secondary"} icon-button dataset-editor-map-control`}
                      type="button"
                      aria-label={annotationsVisible ? "Скрыть всю разметку" : "Показать всю разметку"}
                      aria-pressed={annotationsVisible}
                      title={annotationsVisible ? "Скрыть все полигоны и временно отключить их редактирование" : "Показать полигоны и снова включить их редактирование"}
                      onClick={toggleAnnotations}
                    >
                      {annotationsVisible ? <Eye size={17} /> : <EyeOff size={17} />}
                    </button>
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
                  {pseudoVisible && pseudoMarkup ? (
                    <div className={`dataset-editor-pseudo-status ${pseudoMarkup.status}`} role="status">
                      <span>
                        {pseudoMarkup.status === "ready"
                          ? `Псевдоразметка: ${formatObjectCount(pseudoMarkup.object_count)} · ${pseudoMarkup.source === "dataset" ? "готовый результат датасета" : "инференс снимка"}`
                          : pseudoMarkup.status === "queued"
                            ? "Псевдоразметка: срочное задание в очереди"
                            : pseudoMarkup.status === "running"
                              ? `Псевдоразметка: инференс${pseudoMarkup.progress_total ? ` ${pseudoMarkup.progress_current || 0}/${pseudoMarkup.progress_total}` : ""}`
                              : pseudoMarkup.message || "Псевдоразметка недоступна"}
                      </span>
                      {pseudoMarkup.can_retry ? (
                        <button
                          className="secondary"
                          type="button"
                          disabled={pseudoRequestPending}
                          onClick={() => void loadPseudoMarkup(true, true)}
                        >
                          <RefreshCw size={14} /> Повторить
                        </button>
                      ) : null}
                    </div>
                  ) : null}
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
                disabled={!browser.rasters.some((raster) => !addedAnnotationNames.has(raster.annotation_name.toLocaleLowerCase("ru"))) || busy}
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
            {browser.rasters.map((raster) => {
              const added = addedAnnotationNames.has(raster.annotation_name.toLocaleLowerCase("ru"));
              return (
                <label
                  className={`dataset-editor-file${added ? " already-added" : ""}`}
                  key={raster.path}
                  title={added ? `TIFF ${raster.name} уже добавлен` : `Добавить TIFF ${raster.name}`}
                >
                  {added ? null : (
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
                  )}
                  <span>
                    <strong>{raster.name}</strong>
                    <small>{added ? "Уже добавлен" : raster.annotation_name}</small>
                  </span>
                </label>
              );
            })}
          </div>
        </section>
      ) : null}

      {rebuildPreview ? (
        <div className="modal-backdrop" role="presentation">
          <section className="modal-card panel wide" role="dialog" aria-modal="true" aria-labelledby="dataset-rebuild-title">
            <header className="modal-header">
              <div>
                <h2 id="dataset-rebuild-title">Предпросмотр пересборки</h2>
                <p className="muted">{rebuildPreview.replacement_scene_count} снимков после пересборки</p>
              </div>
              <button className="ghost" type="button" disabled={busy} onClick={() => setRebuildPreview(null)}>Закрыть</button>
            </header>
            <div className="modal-body dataset-editor-rebuild-preview">
              <div className="dataset-editor-rebuild-counts">
                {objectTypeChoices.map((item) => (
                  <span key={item.slug}>
                    <i style={{ backgroundColor: item.color }} />
                    {item.name}: <strong>{rebuildPreview.replacement_class_counts[item.slug] || 0}</strong>
                  </span>
                ))}
                <span><i style={{ backgroundColor: HARD_NEGATIVE_COLOR }} />Hard negative: <strong>{rebuildPreview.replacement_hard_negative_count}</strong></span>
              </div>
              {rebuildPreview.source_changes.length ? (
                <RebuildChangeList title="Изменения источников" items={rebuildPreview.source_changes} />
              ) : null}
              {rebuildPreview.local_changes.length ? (
                <RebuildChangeList
                  title="Ручные изменения"
                  items={rebuildPreview.local_changes.map(formatRebuildChange)}
                />
              ) : null}
              {rebuildPreview.conflicts.length ? (
                <RebuildChangeList
                  title="Конфликты (в merge сохраняется ручная версия)"
                  items={rebuildPreview.conflicts.map(formatRebuildChange)}
                />
              ) : null}
              {rebuildPreview.warnings.length ? (
                <RebuildChangeList title="Предупреждения" items={rebuildPreview.warnings} />
              ) : null}
              {!rebuildPreview.local_changes.length ? (
                <p className="muted">Ручных отклонений от baseline нет.</p>
              ) : null}
            </div>
            <footer className="modal-footer">
              <button className="secondary" type="button" disabled={busy} onClick={() => void applyDatasetRebuild("merge")}>
                Merge · сохранить ручные правки
              </button>
              <button className="danger" type="button" disabled={busy} onClick={() => void applyDatasetRebuild("replace")}>
                Replace · заменить полностью
              </button>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
}

function formatObjectCount(count: number): string {
  const modulo100 = Math.abs(count) % 100;
  const modulo10 = modulo100 % 10;
  const noun = modulo100 >= 11 && modulo100 <= 14
    ? "объектов"
    : modulo10 === 1
      ? "объект"
      : modulo10 >= 2 && modulo10 <= 4
        ? "объекта"
        : "объектов";
  return `${count} ${noun}`;
}

function draftSummary(info: DraftInfo): DraftSummary {
  return {
    annotation_name: info.annotation_name,
    base_revision: info.base_revision,
    deleted: info.deleted,
    stale: info.stale,
    total_count: info.total_count,
    positive_count: info.positive_count,
    hard_negative_count: info.hard_negative_count,
    class_counts: info.class_counts,
    updated_at: info.updated_at,
  };
}

function draftNewFeatureIndexes(live: JsonObject, draft: JsonObject): number[] {
  const liveFeatures = Array.isArray(live.features) ? live.features : [];
  const knownIds = new Set(
    liveFeatures.flatMap((feature) => {
      if (!feature || typeof feature !== "object") return [];
      const id = (feature as JsonObject).id;
      return id === undefined || id === null ? [] : [JSON.stringify(id)];
    }),
  );
  const draftFeatures = Array.isArray(draft.features) ? draft.features : [];
  return draftFeatures.flatMap((feature, index) => {
    if (!feature || typeof feature !== "object") return [];
    const id = (feature as JsonObject).id;
    return id !== undefined && id !== null && !knownIds.has(JSON.stringify(id))
      ? [index]
      : [];
  });
}

function pseudoCacheKey(
  datasetKey: string,
  annotationName: string,
  trainingResultId: string,
): string {
  return JSON.stringify([datasetKey, annotationName, trainingResultId]);
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

function matchesObjectSelection(
  feature: Feature<Geometry>,
  selection: ObjectSelection,
): boolean {
  if (selection === "hard_negative") return feature.get(ROLE_PROPERTY) === "hard_negative";
  if (feature.get(ROLE_PROPERTY) === "hard_negative") return false;
  return selection === "positive" || feature.get(CLASS_PROPERTY) === selection;
}

function applyObjectSelection(
  feature: Feature<Geometry>,
  selection: ObjectSelection,
  dataset: EditorDataset | null,
): void {
  if (selection === "hard_negative") {
    feature.set(ROLE_PROPERTY, "hard_negative", true);
    feature.unset(CLASS_PROPERTY, true);
    return;
  }
  feature.set(ROLE_PROPERTY, "positive", true);
  if (dataset?.task === "multiclass") {
    const slug = dataset.object_types.some((item) => item.slug === selection)
      ? selection
      : dataset.object_types[0]?.slug;
    if (slug) feature.set(CLASS_PROPERTY, slug, true);
  } else {
    feature.unset(CLASS_PROPERTY, true);
  }
}

function formatRebuildChange(change: RebuildChange): string {
  const origin = change.origin_key ? ` · ${change.origin_key}` : "";
  const detail = change.detail ? ` · ${change.detail}` : "";
  return `${change.kind}: ${change.annotation_name}${origin}${detail}`;
}

function RebuildChangeList({ title, items }: { title: string; items: string[] }) {
  return (
    <section>
      <h3>{title}</h3>
      <ul>{items.map((item, index) => <li key={`${index}:${item}`}>{item}</li>)}</ul>
    </section>
  );
}

export function pseudoMarkupStyle(
  feature: Feature<Geometry>,
  objectTypes: EditorObjectType[],
): Style {
  const sourceColor = feature.get("object_type_color");
  const slug = feature.get("object_type_slug");
  const rawId = feature.get("object_type_id");
  const objectType = objectTypes.find((item) =>
    (typeof slug === "string" && item.slug === slug)
    || (rawId !== undefined && rawId !== null && item.id === Number(rawId))
  );
  const color = typeof sourceColor === "string" && /^#[0-9a-f]{6}$/i.test(sourceColor)
    ? sourceColor.toUpperCase()
    : objectType?.color || "#22D3EE";
  const key = color.toUpperCase();
  const cached = pseudoStyleCache.get(key);
  if (cached) return cached;
  const style = new Style({
    fill: new Fill({ color: hexToRgba(color, 0.18) }),
    stroke: new Stroke({ color, width: 2, lineDash: [7, 5] }),
  });
  pseudoStyleCache.set(key, style);
  return style;
}

function featureStyle(
  feature: Feature<Geometry>,
  selected: boolean,
  filled: boolean,
  newFeatures: WeakSet<Feature<Geometry>>,
  objectTypes: EditorObjectType[],
): Style {
  const role = feature.get(ROLE_PROPERTY) === "hard_negative" ? "hard_negative" : "positive";
  const classSlug = typeof feature.get(CLASS_PROPERTY) === "string" ? String(feature.get(CLASS_PROPERTY)) : "";
  const semanticColor = role === "hard_negative"
    ? HARD_NEGATIVE_COLOR
    : objectTypes.find((item) => item.slug === classSlug)?.color || POSITIVE_COLOR;
  const isNew = newFeatures.has(feature);
  const key = `${selected ? "selected" : role}:${semanticColor}:${isNew ? "new" : "saved"}:${filled ? "filled" : "outline"}`;
  const cached = styleCache.get(key);
  if (cached) return cached;
  const strokeColor = selected ? "#38BDF8" : semanticColor;
  const fillColor = hexToRgba(semanticColor, selected ? 0.28 : 0.22);
  const style = new Style({
    stroke: new Stroke({ color: strokeColor, width: selected ? 4 : 3, lineDash: isNew ? [8, 5] : undefined }),
    fill: filled ? new Fill({ color: fillColor }) : undefined,
  });
  styleCache.set(key, style);
  return style;
}

function selectedFeatureStyles(
  feature: Feature<Geometry>,
  filled: boolean,
  newFeatures: WeakSet<Feature<Geometry>>,
  objectTypes: EditorObjectType[],
  selectedVertices: VertexSelection[],
): Style[] {
  const styles = [
    featureStyle(feature, true, filled, newFeatures, objectTypes),
    EDITABLE_VERTICES_STYLE,
  ];
  const coordinates = selectedVertices
    .filter((selection) => selection.feature === feature)
    .map((selection) => selection.vertex.coordinate);
  if (coordinates.length) {
    styles.push(new Style({
      geometry: new MultiPoint(coordinates),
      image: SELECTED_VERTEX_IMAGE,
    }));
  }
  return styles;
}

function hexToRgba(color: string, alpha: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(color);
  if (!match) return `rgba(243, 198, 35, ${alpha})`;
  const value = Number.parseInt(match[1], 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
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
    contrast: RASTER_CONTRAST,
  };
}

async function rasterViewWithOverzoom(source: GeoTIFF): Promise<ViewOptions> {
  const view = await source.getView();
  const viewResolutions = view.resolutions;
  const nativeResolution = source.getTileGrid()?.getResolutions().at(-1);
  if (!viewResolutions || nativeResolution === undefined) return view;
  return {
    ...view,
    resolutions: extendRasterResolutions(viewResolutions, nativeResolution),
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

function coordinatesEqual(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
