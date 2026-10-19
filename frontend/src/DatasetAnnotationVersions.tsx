import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Check, Plus, Trash2 } from "lucide-react";
import { createPortal } from "react-dom";
import type { AnnotationVersionCatalog, AnnotationVersionDetail, AnnotationVersionDraft } from "./api/types";
import { cloneSnapshot, snapshotsEqual, type DraftSnapshot, type JsonObject } from "./utils/datasetEditor";
import { formatDateTime } from "./utils/format";
import "./styles/annotation-versions.css";

export function retainVersionEdit(snapshot: DraftSnapshot, catalog: AnnotationVersionCatalog, baseline?: JsonObject): AnnotationVersionDraft {
  const state = snapshot.annotationVersions;
  if (!state) throw new Error("Каталог версий ещё не загружен");
  const existing = (state.edits || []).find((edit) => edit.id === state.selected_id);
  const published = catalog.versions.find((version) => version.id === state.selected_id);
  if (!existing && !published) throw new Error("Выбранная версия отсутствует");
  if (published && baseline && snapshotsEqual({ geojson: snapshot.geojson, newFeatureIndexes: [] }, { geojson: baseline, newFeatureIndexes: [] })) {
    return { ...state, edits: (state.edits || []).filter((edit) => edit.id !== state.selected_id) };
  }
  return {
    ...state,
    edits: [...(state.edits || []).filter((edit) => edit.id !== state.selected_id), {
      ...(existing || { id: state.selected_id, name: published!.name }),
      geojson: snapshot.geojson,
    }],
  };
}

type Props = {
  catalog: AnnotationVersionCatalog;
  snapshot: DraftSnapshot;
  publishedGeojson: JsonObject;
  publishedVersions?: Map<string, JsonObject>;
  disabled: boolean;
  loadVersion: (id: string) => Promise<AnnotationVersionDetail | undefined>;
  onChange: (snapshot: DraftSnapshot, previous: DraftSnapshot) => void;
  anchor?: HTMLElement;
  onClose?: () => void;
};

export function DatasetAnnotationVersions({ catalog, snapshot, publishedGeojson, publishedVersions, disabled, loadVersion, onChange, anchor, onClose }: Props) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const payloads = useRef(publishedVersions || new Map<string, JsonObject>(catalog.selected_id ? [[catalog.selected_id, publishedGeojson]] : []));
  const [pending, setPending] = useState(false);
  const [dialog, setDialog] = useState<"create" | "delete" | null>(null);
  const [name, setName] = useState("");
  const [empty, setEmpty] = useState(false);
  const [replacement, setReplacement] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [error, setError] = useState("");
  useLayoutEffect(() => {
    if (!anchor) return;
    const place = () => {
      const bounds = anchor.getBoundingClientRect();
      const below = window.innerHeight - bounds.bottom - 14;
      const height = Math.min(menuRef.current?.scrollHeight || 440, Math.max(bounds.top - 14, below));
      const top = below < height && bounds.top > below ? bounds.top - height - 6 : bounds.bottom + 6;
      setPosition({ top: Math.max(8, top),
        left: Math.max(8, Math.min(bounds.left, window.innerWidth - 360)) });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => { window.removeEventListener("resize", place); window.removeEventListener("scroll", place, true); };
  }, [anchor]);
  useEffect(() => {
    const outside = (event: PointerEvent) => {
      if (!dialog && !menuRef.current?.contains(event.target as Node) && !anchor?.contains(event.target as Node)) onClose?.();
    };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") { event.stopPropagation(); dialog ? setDialog(null) : onClose?.(); } };
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", escape, true);
    return () => { document.removeEventListener("pointerdown", outside, true); document.removeEventListener("keydown", escape, true); };
  }, [anchor, dialog, onClose]);
  const state = snapshot.annotationVersions;
  if (!state) return null;
  const entries = new Map<string, Omit<AnnotationVersionCatalog["versions"][number], "updated_at" | "created_at"> & { updated_at?: string; created_at?: string }>(catalog.versions.map((version) => [version.id, version]));
  for (const edit of state.edits || []) {
    if (!entries.has(edit.id)) entries.set(edit.id, { id: edit.id, name: edit.name, revision: 1, usages: [] });
  }
  const versions = [...entries.values()].filter((version) => !(version.id in (state.replacements || {})));
  const selected = entries.get(state.selected_id);
  const deleting = entries.get(deletingId);
  const usages = deleting?.usages || [];
  const unavailable = disabled || pending;

  const payloadFor = async (id: string): Promise<JsonObject | undefined> => {
    const edit = (state.edits || []).find((edit) => edit.id === id);
    if (edit) return edit.geojson as JsonObject;
    const response = await loadVersion(id);
    if (response && response.catalog.revision !== state.catalog_revision) {
      setError("Состав версий изменился. Сохраните текущую работу и заново откройте снимок.");
      return undefined;
    }
    if (response) payloads.current.set(id, response.geojson as JsonObject);
    return response?.geojson as JsonObject | undefined;
  };

  const select = async (id: string, removedId?: string) => {
    if (unavailable || (!removedId && id === state.selected_id)) return;
    setPending(true);
    setError("");
    try {
      const nextId = removedId && state.selected_id !== removedId ? state.selected_id : id;
      const payload = nextId === state.selected_id ? snapshot.geojson : await payloadFor(nextId);
      if (!payload) return;
      const next = cloneSnapshot(snapshot);
      next.annotationVersions = retainVersionEdit(next, catalog, payloads.current.get(state.selected_id));
      next.annotationVersions.selected_id = nextId;
      if (removedId) next.annotationVersions.replacements = {
        ...Object.fromEntries(Object.entries(next.annotationVersions.replacements || {}).map(([from, to]) => [from, to === removedId ? id : to])),
        [removedId]: id,
      };
      next.geojson = payload;
      next.newFeatureIndexes = nextId === state.selected_id ? [...snapshot.newFeatureIndexes] : [];
      onChange(next, snapshot);
      setDialog(null);
      onClose?.();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось переключить версию.");
    } finally {
      setPending(false);
    }
  };

  const create = () => {
    if (!name.trim() || unavailable) return;
    const next = cloneSnapshot(snapshot);
    next.annotationVersions = retainVersionEdit(next, catalog, payloads.current.get(state.selected_id));
    const id = crypto.randomUUID();
    const geojson = empty ? { ...next.geojson, features: [] } : next.geojson;
    next.annotationVersions.edits = [...(next.annotationVersions.edits || []), {
      id, name: name.trim(), source_id: empty ? null : state.selected_id, geojson,
    }];
    next.annotationVersions.selected_id = id;
    next.geojson = geojson;
    next.newFeatureIndexes = [];
    onChange(next, snapshot);
    setDialog(null);
    onClose?.();
  };

  const menu = <div ref={menuRef} className="annotation-versions-menu" role="dialog" aria-label="Версии разметки снимка"
    style={anchor ? { position: "fixed", ...position, maxHeight: `calc(100dvh - ${position.top + 8}px)` } : undefined}>
    <header><strong>Версии разметки · {versions.length}</strong><small>Выбор и удаления применяются после публикации.</small></header>
    <div className="annotation-version-cards">
      {versions.map((version) => <div key={version.id} className={`annotation-version-card${version.id === state.selected_id ? " selected" : ""}`}>
        <button type="button" className="annotation-version-choice" disabled={unavailable} aria-pressed={version.id === state.selected_id}
          onClick={() => version.id === state.selected_id ? onClose?.() : void select(version.id)}>
          <strong>{version.id === state.selected_id ? <Check size={14} /> : null}{version.name}</strong>
          <small title="Для прежних файлов это дата начала учёта версии, а не исходного создания разметки.">Создана: {version.created_at ? formatDateTime(version.created_at) : "ещё не опубликована"}</small>
          {version.updated_at ? <small>Изменена: {formatDateTime(version.updated_at)}</small> : null}
          <small>{version.usages?.length ? `Используется: ${version.usages.map((usage) => usage.dataset_name + (usage.indirect ? " (через источник)" : "")).join(", ")}` : "Пока не используется в опубликованных датасетах"}</small>
        </button>
        <button type="button" className="annotation-version-delete secondary icon-button" disabled={unavailable || versions.length < 2}
          aria-label={`Удалить версию «${version.name}»`} title={versions.length < 2 ? "Сначала создайте другую версию для замены" : "Удалить с заменой во всех использующих датасетах"}
          onClick={() => { setDeletingId(version.id); setReplacement(versions.find((candidate) => candidate.id !== version.id)?.id || ""); setDialog("delete"); }}><Trash2 size={14} /></button>
      </div>)}
    </div>
    <button type="button" className="secondary annotation-version-create" disabled={unavailable} onClick={() => {
      setName(`Версия ${entries.size + 1}`); setEmpty(false); setDialog("create");
    }}><Plus size={15} />Создать версию</button>
    {selected ? <small>Выбрана: {selected.name}</small> : null}
    {error ? <small role="alert">{error}</small> : null}
    {dialog ? createPortal(<div className="modal-backdrop" role="presentation">
      <section className="modal-card panel" role="dialog" aria-modal="true" aria-labelledby="annotation-version-dialog-title">
        <header className="modal-header"><h2 id="annotation-version-dialog-title">{dialog === "create" ? "Новая версия разметки" : `Удалить «${deleting?.name}»`}</h2></header>
        <div className="modal-body">
          {dialog === "create" ? <>
            <label>Название<input autoFocus maxLength={240} value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label><input type="checkbox" checked={empty} onChange={(event) => setEmpty(event.target.checked)} />Начать с пустой разметки</label>
            <p>По умолчанию копируется текущая разметка вместе с вашими правками. Новая версия станет независимой и будет выбрана в этом датасете после публикации.</p>
            <p>Уже сделанные правки исходной версии остаются в черновике: создание копии их не отменяет. Чтобы менять только копию, создавайте её до редактирования.</p>
          </> : <>
            <p>Версия будет удалена из доступных вариантов. В использующих её датасетах она заменится выбранной ниже.</p>
            <ul>{usages.map((usage) => <li key={usage.dataset_key}>{usage.dataset_name}{usage.indirect ? " — через исходный датасет" : ""}</li>)}</ul>
            {!usages.length ? <p>Опубликованных использований нет.</p> : null}
            <label>Заменить на<select value={replacement} onChange={(event) => setReplacement(event.target.value)}>
              {versions.filter((version) => version.id !== deletingId).map((version) => <option key={version.id} value={version.id}>{version.name} · {version.updated_at ? formatDateTime(version.updated_at) : "ещё не опубликована"}</option>)}
            </select></label>
            <p>Сохранённые снимки данных прежних обучений и их результаты остаются прежними.</p>
          </>}
          <p>Это действие сохраняется в черновик. Его можно отменить до общей публикации.</p>
        </div>
        <footer className="modal-footer">
          <button type="button" className="secondary" disabled={pending} onClick={() => setDialog(null)}>Отмена</button>
          <button type="button" className="primary" disabled={unavailable || (dialog === "create" ? !name.trim() : !replacement)}
            onClick={() => dialog === "create" ? create() : void select(replacement, deletingId)}>{dialog === "create" ? "Создать в черновике" : "Удалить и заменить при публикации"}</button>
        </footer>
      </section>
    </div>, document.fullscreenElement || document.body) : null}
  </div>;
  return anchor ? createPortal(menu, document.fullscreenElement || document.body) : menu;
}
