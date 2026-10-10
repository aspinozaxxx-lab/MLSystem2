import { CircleAlert, ChevronDown, ChevronUp, Layers3, LoaderCircle, Play, Square, Star, Trash2 } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

import { apiJson } from "./api/client";
import type { JobDetail, TestSampleBatchCreate, TestSampleBatchInfo, TestSampleBatchOptionsResponse, TestSampleCreationSettings } from "./api/types";
import { formatDateTime, shortVersion } from "./utils/format";
import { estimateTestMarkupImageVolume, formatTestMarkupImageVolume } from "./utils/testMarkups";
import { useTestMarkupClasses } from "./useTestMarkupClasses";

export const TEST_SAMPLE_TILE_SIZES = [512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192] as const;
type Settings = Required<TestSampleCreationSettings>;
type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;
const defaults: Settings = { tile_size: 1536, min_image_count: 5, image_count: 10, min_object_count: 150, min_object_area_m2: 0, exclude_boundary_objects: false, use_optimization: true };
const statusLabels = { queued: "В очереди", running: "Создаётся", ok: "Готово", partial: "Частично", error: "Ошибка", cancelled: "Отменено" };
const pseudoLabels = { ready: "Псевдоразметка готова", queued: "Псевдоразметка в очереди", running: "Создаётся псевдоразметка", unavailable: "Псевдоразметки нет", error: "Ошибка псевдоразметки" };

export function TestMarkupCreatePage({ run }: { run: Runner }) {
  const [options, setOptions] = useState<TestSampleBatchOptionsResponse | null>(null);
  const { index, classKey, setClassKey, loadClasses } = useTestMarkupClasses(run, true);
  const [datasetKey, setDatasetKey] = useState("");
  const [drafts, setDrafts] = useState<Record<string, Settings>>({});
  const [queue, setQueue] = useState<TestSampleBatchInfo[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [actionId, setActionId] = useState<string | null>(null);
  const [showNetworkWarning, setShowNetworkWarning] = useState(false);
  const saves = useRef(Promise.resolve());
  const queueRequest = useRef(0);
  const completedQueueRevision = useRef<string | null>(null);
  const optionsRequest = useRef(0);
  const optionsCache = useRef(new Map<string, TestSampleBatchOptionsResponse>());

  const classesLoaded = Boolean(index);
  const loadOptions = useCallback(async () => {
    if (!classKey || !classesLoaded) return;
    const revision = ++optionsRequest.current;
    const payload = await run(() => apiJson<TestSampleBatchOptionsResponse>(`/test-sample-batches/options?class_key=${encodeURIComponent(classKey)}`));
    if (!payload || revision !== optionsRequest.current) return;
    optionsCache.current.set(classKey, payload);
    setOptions(payload);
    setDrafts((current) => {
      const updated = { ...current };
      for (const group of payload.classes || []) for (const source of group.datasets || []) {
        updated[source.dataset_key] ??= { ...defaults, ...source.creation_settings };
      }
      return updated;
    });
  }, [run, classKey, classesLoaded]);
  const loadQueue = useCallback(async () => {
    const revision = ++queueRequest.current;
    const payload = await run(() => apiJson<TestSampleBatchInfo[]>("/test-sample-batches"));
    if (payload && revision === queueRequest.current) {
      setQueue(payload);
      const completed = payload.filter((job) => job.status === "ok" || job.status === "partial").map((job) => job.id).join("|");
      if (completedQueueRevision.current !== null && completedQueueRevision.current !== completed) void loadClasses();
      completedQueueRevision.current = completed;
    }
  }, [run, loadClasses]);
  useEffect(() => { void loadQueue(); }, [loadQueue]);
  useEffect(() => {
    setOptions(optionsCache.current.get(classKey) || null);
    void loadOptions();
    return () => { optionsRequest.current += 1; };
  }, [classKey, loadOptions]);
  const selectedClass = options?.classes?.find((group) => group.class_key === classKey);
  const datasets = selectedClass?.datasets || [];
  const dataset = datasets.find((source) => source.dataset_key === datasetKey);
  useEffect(() => {
    if (!datasets.some((source) => source.dataset_key === datasetKey)) setDatasetKey(datasets[0]?.dataset_key || "");
  }, [selectedClass, datasetKey]);
  const saved = drafts[datasetKey] || defaults;
  const settings: Settings = { ...saved, use_optimization: Boolean(dataset?.training_result_id && saved.use_optimization), exclude_boundary_objects: Boolean(dataset?.quality_metric === "objects" && saved.exclude_boundary_objects) };
  const valid = [settings.min_image_count, settings.image_count, settings.min_object_count].every((value) => Number.isInteger(value) && value > 0) && settings.min_image_count <= settings.image_count && Number.isFinite(settings.min_object_area_m2) && settings.min_object_area_m2 >= 0;
  const imageVolume = estimateTestMarkupImageVolume(dataset?.imagery_type, settings.tile_size, settings.min_image_count, settings.image_count);
  const active = queue.filter((job) => job.status === "queued" || job.status === "running");
  const pending = active.filter((job) => job.status === "queued");
  const pseudoActive = options?.classes?.some((group) => group.datasets?.some((source) => source.pseudo_status === "queued" || source.pseudo_status === "running"));
  useEffect(() => {
    const timer = window.setInterval(() => { void loadQueue(); }, active.length ? 2_000 : 10_000);
    return () => window.clearInterval(timer);
  }, [active.length, loadQueue]);
  useEffect(() => {
    if (!pseudoActive) return;
    const timer = window.setInterval(() => void loadOptions(), 2_000);
    return () => window.clearInterval(timer);
  }, [pseudoActive, loadOptions]);

  const updateSettings = (change: Partial<Settings>) => {
    const next = { ...settings, ...change };
    setDrafts((current) => ({ ...current, [datasetKey]: next }));
    if (![next.min_image_count, next.image_count, next.min_object_count].every((value) => Number.isInteger(value) && value > 0) || next.min_image_count > next.image_count || !Number.isFinite(next.min_object_area_m2) || next.min_object_area_m2 < 0) return;
    const key = datasetKey;
    saves.current = saves.current.then(async () => {
      await run(() => apiJson<TestSampleCreationSettings>(`/test-sample-batches/options/${encodeURIComponent(key)}/settings`, { method: "PUT", body: next }));
    });
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!dataset || !valid || submitting) return;
    const request: TestSampleBatchCreate = {
      tile_size: settings.tile_size, min_image_count: settings.min_image_count, image_count: settings.image_count,
      items: [{ dataset_key: dataset.dataset_key, training_result_id: settings.use_optimization ? dataset.training_result_id : null, min_object_count: settings.min_object_count, min_object_area_m2: settings.min_object_area_m2, metric: dataset.quality_metric, exclude_boundary_objects: settings.exclude_boundary_objects, use_optimization: settings.use_optimization }],
    };
    setSubmitting(true);
    try {
      await saves.current;
      const result = await run(() => apiJson<TestSampleBatchInfo>("/test-sample-batches", { method: "POST", body: request }));
      if (result) await loadQueue();
    } finally { setSubmitting(false); }
  };
  const queueAction = async (id: string, action: "up" | "down" | "delete" | "cancel") => {
    setActionId(id);
    try {
      await run(() => apiJson(action === "delete" ? `/test-sample-batches/${id}` : `/test-sample-batches/${id}/${action === "cancel" ? "cancel" : "move"}`, { method: action === "delete" ? "DELETE" : "POST", ...(action === "up" || action === "down" ? { body: { direction: action } } : {}) }));
      await loadQueue();
    } finally { setActionId(null); }
  };
  const launchPseudo = async () => {
    const result = await run(() => apiJson<JobDetail>(`/test-sample-batches/options/${encodeURIComponent(datasetKey)}/pseudo-markup`, { method: "POST" }));
    if (result) await loadOptions();
  };
  const selectedIndex = index?.classes?.find((item) => item.key === classKey);

  return <div className="test-markup-create">
    <header className="page-header"><div><h1>Создание тестовой разметки</h1><p>Один датасет — одно задание</p></div></header>
    <form className={`panel creation-form${settings.use_optimization && dataset?.pseudo_status !== "ready" ? " needs-pseudo" : ""}`} onSubmit={submit}>
      <div className="creation-source-fields">
        <label className="field creation-class"><span>Класс <small>Разметок: {selectedIndex?.sample_count ?? (index ? 0 : "…")}{selectedIndex?.has_primary ? " · есть основная" : ""}</small></span><select aria-label="Класс" value={classKey} onChange={(event) => { setClassKey(event.target.value); setShowNetworkWarning(false); }} disabled={!index?.classes?.length}>
          {!index?.classes?.length ? <option>{index ? "Нет классов с датасетами" : "Загрузка…"}</option> : null}{index?.classes?.map((group) => <option key={group.key} value={group.key}>{group.name}</option>)}
        </select></label>
        <div className="creation-sources" role="radiogroup" aria-label="Датасет">
          {datasets.map((source) => <label className={`creation-source ${source.dataset_key === datasetKey ? "selected" : ""}`} key={source.dataset_key}>
            <input type="radio" name="creation-dataset" aria-label={source.dataset_name} checked={source.dataset_key === datasetKey} onChange={() => { setDatasetKey(source.dataset_key); setShowNetworkWarning(false); }} />
            <span><strong>{source.dataset_name}</strong><small className="creation-source-meta">{source.image_count} снимков · {shortVersion(source.dataset_version)} · {source.quality_metric === "objects" ? "объектовая F1" : "пиксельная F1"}</small>
              <small className="creation-network">Сеть: {source.training_model_name || "нет обученных сетей"}{source.training_is_primary ? <Star size={12} fill="currentColor" aria-label="Основная сеть" /> : null}{source.training_trained_at ? ` · ${formatDateTime(source.training_trained_at)}` : ""}</small>
              {source.training_result_id ? <small title={source.error || undefined}>{pseudoLabels[source.pseudo_status]}</small> : null}
            </span>
          </label>)}
          {index && !index.classes?.length ? <p className="muted">Нет доступных классов с датасетами.</p> : !options ? <p className="muted" role="status">Загрузка датасетов выбранного класса…</p> : !datasets.length ? <p className="muted">Нет готовых датасетов с размеченными снимками.</p> : null}
        </div>
      </div>
      <div className="creation-settings">
        {dataset ? <>
          <div className="creation-parameters">
            <label className="field"><span>Размер тайла, пикс.</span><select aria-label="Размер тайла" value={settings.tile_size} onChange={(event) => updateSettings({ tile_size: Number(event.target.value) as Settings["tile_size"] })}>{TEST_SAMPLE_TILE_SIZES.map((size) => <option key={size} value={size}>{size} × {size}</option>)}</select></label>
            <fieldset className="creation-range"><legend>Тайлов в итоге</legend><input type="number" inputMode="numeric" aria-label="Минимум тайлов" min="1" step="1" required value={settings.min_image_count || ""} onChange={(event) => updateSettings({ min_image_count: Number(event.target.value) })} /><span>—</span><input type="number" inputMode="numeric" aria-label="Максимум тайлов" min="1" step="1" required value={settings.image_count || ""} onChange={(event) => updateSettings({ image_count: Number(event.target.value) })} /></fieldset>
            <label className="field"><span>Минимум объектов</span><input type="number" inputMode="numeric" aria-label="Минимум объектов" min="1" step="1" required value={settings.min_object_count || ""} onChange={(event) => updateSettings({ min_object_count: Number(event.target.value) })} /></label>
            <label className="field" title="Площадь исходного объекта до нарезки. Объекты меньше этого значения исключаются; 0 — без фильтра."><span>Мин. площадь, м²</span><input type="number" inputMode="decimal" aria-label="Минимальная площадь объекта в м²" min="0" step="any" required value={settings.min_object_area_m2} onChange={(event) => updateSettings({ min_object_area_m2: Number(event.target.value) })} /></label>
            <div className="creation-optimization"><label><input type="checkbox" checked={settings.use_optimization} disabled={!dataset.training_result_id} onChange={(event) => updateSettings({ use_optimization: event.target.checked })} />Использовать оптимизацию</label>
              {!dataset.training_result_id ? <span className="creation-warning"><button type="button" aria-label="Почему оптимизация недоступна" title="Нет обученных сетей для выбранного датасета" onClick={() => setShowNetworkWarning((current) => !current)}><CircleAlert size={17} /></button><span role="tooltip" className={showNetworkWarning ? "visible" : ""}>Нет обученных сетей для выбранного датасета. Создайте разметку без оптимизации.</span></span> : null}
            </div>
          </div>
          {dataset.quality_metric === "objects" ? <label className="creation-boundary"><input type="checkbox" checked={settings.exclude_boundary_objects} onChange={(event) => updateSettings({ exclude_boundary_objects: event.target.checked })} />Не учитывать объекты, выходящие за тайл</label> : null}
          <div className="creation-hint">Настройки сохраняются · запас: до {settings.image_count * 3} тайлов · площадь 0 — без фильтра</div>
          <div className="creation-size-estimate" role="status" aria-live="polite" aria-atomic="true">
            <strong>Примерный объём снимков (TIFF)</strong>
            {imageVolume ? <>
              <div className="creation-size-range"><span>≈ {formatTestMarkupImageVolume(imageVolume.minBytes)} ({settings.min_image_count.toLocaleString("ru-RU")} шт.)</span><span>— {formatTestMarkupImageVolume(imageVolume.maxBytes)} ({settings.image_count.toLocaleString("ru-RU")} шт.)</span></div>
              <small>{dataset.imagery_type === "kanopus" ? "Канопус" : "Ортофото"}: ориентир с учётом сжатия и пирамид. Без превью, масок и запасных тайлов; фактический объём может отличаться.</small>
            </> : <small>{dataset.imagery_type ? "Укажите корректный размер и диапазон количества снимков для оценки." : "Тип снимков не указан — оценка объёма недоступна."}</small>}
          </div>
          {settings.use_optimization && dataset.pseudo_status !== "ready" ? <div className="creation-pseudo"><button className="secondary" type="button" title="Для оптимизации нужна полная псевдоразметка выбранной сети этого датасета" disabled={dataset.pseudo_status === "queued" || dataset.pseudo_status === "running"} onClick={() => void launchPseudo()}><Play size={14} />Создать псевдоразметку для оптимизации</button></div> : null}
          {!valid ? <small className="error-text">Укажите положительные целые количества и площадь от 0; минимум тайлов должен быть не больше максимума.</small> : null}
          <button className="primary creation-submit" type="submit" disabled={submitting || !valid || (settings.use_optimization && dataset.pseudo_status !== "ready")}>
            {submitting ? <LoaderCircle size={17} className="spin" /> : <Layers3 size={17} />} {submitting ? "Добавление…" : "Создать разметку"}
          </button>
        </> : null}
      </div>
    </form>
    <section className="panel creation-queue" aria-label="Очередь создания разметок">
      <div className="creation-queue-heading"><h2>Очередь создания разметок</h2><span className="muted">Ожидают: {pending.length}</span></div>
      <div className="creation-queue-list">
        {[...active, ...queue.filter((job) => !active.includes(job)).slice(0, Math.max(0, 5 - active.length))].map((job) => {
          const item = job.items?.[0];
          const position = pending.findIndex((entry) => entry.id === job.id);
          return <article className={`creation-job ${job.status}${active.includes(job) ? "" : " history"}`} key={job.id}>
            <div className="creation-job-info"><strong>{item?.class_name} · {item?.dataset_name.split("\\").at(-1)}</strong><small>{job.tile_size} пикс. · {job.min_image_count}–{job.image_count} тайлов · ≥ {item?.min_object_count} объектов{item?.min_object_area_m2 ? ` · от ${item.min_object_area_m2.toLocaleString("ru-RU")} м²` : ""} · {item?.use_optimization ? "с оптимизацией" : "случайный выбор"}</small>
              <span>{job.cancel_requested && job.status === "running" ? "Отмена…" : statusLabels[job.status]}{job.status === "running" ? ` · ${Math.floor(job.elapsed_seconds / 60)} мин.` : ""}{item?.sample_id ? <> · <a href={`#/test-markups/${item.sample_id}`}>Открыть разметку</a></> : null}</span>{item?.error ? <small className="error-text">{item.error}</small> : null}
            </div>
            {job.status === "queued" ? <div className="creation-job-actions"><button className="secondary icon-button" type="button" aria-label="Выше в очереди" disabled={actionId !== null || position === 0} onClick={() => void queueAction(job.id, "up")}><ChevronUp size={19} /></button><button className="secondary icon-button" type="button" aria-label="Ниже в очереди" disabled={actionId !== null || position === pending.length - 1} onClick={() => void queueAction(job.id, "down")}><ChevronDown size={19} /></button><button className="secondary icon-button" type="button" aria-label="Удалить из очереди" disabled={actionId !== null} onClick={() => void queueAction(job.id, "delete")}><Trash2 size={17} /></button></div> : job.status === "running" ? <button className="secondary icon-button" type="button" aria-label="Отменить создание" disabled={actionId !== null || job.cancel_requested} onClick={() => void queueAction(job.id, "cancel")}><Square size={17} /></button> : null}
          </article>;
        })}
        {!queue.length ? <div className="muted creation-empty">Заданий пока нет. Выберите параметры и создайте разметку.</div> : null}
      </div>
    </section>
  </div>;
}
