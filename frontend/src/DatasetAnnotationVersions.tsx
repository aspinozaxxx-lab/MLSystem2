import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { AnnotationVersionCatalog, AnnotationVersionDetail, AnnotationVersionDraft } from "./api/types";
import { cloneSnapshot, snapshotsEqual, type DraftSnapshot, type JsonObject } from "./utils/datasetEditor";
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
  disabled: boolean;
  loadVersion: (id: string) => Promise<AnnotationVersionDetail | undefined>;
  onChange: (snapshot: DraftSnapshot, previous: DraftSnapshot) => void;
};

export function DatasetAnnotationVersions({ catalog, snapshot, publishedGeojson, disabled, loadVersion, onChange }: Props) {
  const payloads = useRef(new Map<string, JsonObject>(catalog.selected_id ? [[catalog.selected_id, publishedGeojson]] : []));
  const [pending, setPending] = useState(false);
  const [dialog, setDialog] = useState<"create" | "delete" | null>(null);
  const [name, setName] = useState("");
  const [empty, setEmpty] = useState(false);
  const [replacement, setReplacement] = useState("");
  const [error, setError] = useState("");
  const state = snapshot.annotationVersions;
  if (!state) return null;
  const entries = new Map(catalog.versions.map((version) => [version.id, version]));
  for (const edit of state.edits || []) {
    if (!entries.has(edit.id)) entries.set(edit.id, { id: edit.id, name: edit.name, revision: 1, usages: [] });
  }
  const versions = [...entries.values()].filter((version) => !(version.id in (state.replacements || {})));
  const selected = entries.get(state.selected_id);
  const usages = selected?.usages || [];
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

  const select = async (id: string, deleting = false) => {
    if (unavailable || (!deleting && id === state.selected_id)) return;
    setPending(true);
    setError("");
    try {
      const payload = await payloadFor(id);
      if (!payload) return;
      const next = cloneSnapshot(snapshot);
      next.annotationVersions = retainVersionEdit(next, catalog, payloads.current.get(state.selected_id));
      next.annotationVersions.selected_id = id;
      if (deleting) next.annotationVersions.replacements = {
        ...Object.fromEntries(Object.entries(next.annotationVersions.replacements || {}).map(([from, to]) => [from, to === state.selected_id ? id : to])),
        [state.selected_id]: id,
      };
      next.geojson = payload;
      next.newFeatureIndexes = [];
      onChange(next, snapshot);
      setDialog(null);
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
  };

  return <div className="annotation-versions">
    <label>Версия разметки · {versions.length}
      <select aria-label="Версия разметки снимка" value={state.selected_id} disabled={unavailable}
        onChange={(event) => void select(event.target.value)}>
        {versions.map((version) => <option key={version.id} value={version.id}>{version.name}</option>)}
      </select>
    </label>
    <button type="button" className="secondary" disabled={unavailable} onClick={() => {
      setName(`Версия ${entries.size + 1}`); setEmpty(false); setDialog("create");
    }}>Новая версия</button>
    <button type="button" className="secondary" disabled={unavailable || versions.length < 2}
      title={versions.length < 2 ? "Сначала создайте другую версию для замены" : "Удалить выбранную версию с заменой в датасетах"}
      onClick={() => { setReplacement(versions.find((version) => version.id !== state.selected_id)?.id || ""); setDialog("delete"); }}>
      Удалить версию
    </button>
    <small className="annotation-version-usage">
      {usages.length ? <>Используется: {usages.map((usage, index) => <span key={usage.dataset_key}>
        {index ? ", " : ""}<a href={`#/dataset-editor/${encodeURIComponent(usage.dataset_key)}`}>{usage.dataset_name}</a>{usage.indirect ? " (через источник)" : ""}
      </span>)}. Правки этой версии при публикации обновят все эти датасеты.</> : catalog.versions.some((version) => version.id === state.selected_id)
        ? "Не используется в опубликованных датасетах. После публикации будет выбрана в этом датасете."
        : "Новая версия: пока существует только в вашем черновике."}
      {Object.keys(state.replacements || {}).length > 0 ? " Удаление и замена версий ожидают публикации." : ""}
    </small>
    {error ? <small role="alert">{error}</small> : null}
    {dialog ? createPortal(<div className="modal-backdrop" role="presentation">
      <section className="modal-card panel" role="dialog" aria-modal="true" aria-labelledby="annotation-version-dialog-title">
        <header className="modal-header"><h2 id="annotation-version-dialog-title">{dialog === "create" ? "Новая версия разметки" : `Удалить «${selected?.name}»`}</h2></header>
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
              {versions.filter((version) => version.id !== state.selected_id).map((version) => <option key={version.id} value={version.id}>{version.name}</option>)}
            </select></label>
            <p>Сохранённые снимки данных прежних обучений и их результаты остаются прежними.</p>
          </>}
          <p>Это действие сохраняется в черновик. Его можно отменить до общей публикации.</p>
        </div>
        <footer className="modal-footer">
          <button type="button" className="secondary" disabled={pending} onClick={() => setDialog(null)}>Отмена</button>
          <button type="button" className="primary" disabled={unavailable || (dialog === "create" ? !name.trim() : !replacement)}
            onClick={() => dialog === "create" ? create() : void select(replacement, true)}>{dialog === "create" ? "Создать в черновике" : "Удалить и заменить при публикации"}</button>
        </footer>
      </section>
    </div>, document.fullscreenElement || document.body) : null}
  </div>;
}
