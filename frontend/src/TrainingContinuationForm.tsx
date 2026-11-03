import { LoaderCircle, Play } from "lucide-react";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { apiJson } from "./api/client";
import type { DatasetInfo, JobDetail, TestSampleCard, TrainingContinuationCreate, TrainingContinuationOptions, TrainingContinuationNormalizationAdvice, TrainingResultInfo } from "./api/types";
import { trainingPipelineLabel } from "./utils/trainingConfig";

export function TrainingContinuationForm({ result, options, datasets = [], testSamples = [], run, closeModal, reload, inferenceAvailable = true }: {
  result: TrainingResultInfo;
  options: TrainingContinuationOptions;
  datasets?: DatasetInfo[];
  testSamples?: TestSampleCard[];
  run: <T>(operation: () => Promise<T>) => Promise<T | undefined>;
  closeModal: () => void;
  reload: () => Promise<void>;
  inferenceAvailable?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [checkpoint, setCheckpoint] = useState<"best" | "last">("best");
  const [pipeline, setPipeline] = useState(result.pipeline_variant || "legacy");
  const pipelineChoices = options.pipeline_variants?.length ? options.pipeline_variants : [result.pipeline_variant || "legacy"];
  const [datasetSource, setDatasetSource] = useState<NonNullable<TrainingContinuationCreate["dataset_source"]>>("original");
  const classDatasets = datasets.filter(item => options.class_key && item.class_key === options.class_key && !item.is_custom);
  const classSamples = testSamples.filter(item => options.class_key && item.class_key === options.class_key);
  const [selectedDataset, setSelectedDataset] = useState(classDatasets.find(item => item.key !== result.dataset_key)?.key || classDatasets[0]?.key || "");
  const [selectedSample, setSelectedSample] = useState(classSamples[0]?.id || "");
  const [normalizationPolicy, setNormalizationPolicy] = useState<"auto" | "preserve" | "recompute">("auto");
  const [advice, setAdvice] = useState<TrainingContinuationNormalizationAdvice | null>(null);
  const [advising, setAdvising] = useState(false);
  const globalProfile = (variant: string) => variant === "next_gen4_global_norm" || variant.startsWith("next_gen4_augmentation_") || variant.startsWith("next_gen4_consistency_");
  const globalTarget = globalProfile(pipeline), globalSource = globalProfile(result.pipeline_variant || "legacy");
  useEffect(() => {
    setAdvice(null);
    if (!globalTarget) { setAdvising(false); return; }
    const controller = new AbortController();
    setAdvising(true);
    const timer = window.setTimeout(() => {
      void apiJson<TrainingContinuationNormalizationAdvice>(`/results/training/${result.id}/continue/normalization`, {
        method: "POST", signal: controller.signal, body: {
          checkpoint, pipeline_variant: pipeline, dataset_source: datasetSource,
          ...(datasetSource === "dataset" ? { dataset_key: selectedDataset } : {}),
          ...(datasetSource === "test_sample" ? { test_sample_id: selectedSample } : {}),
        },
      }).then(value => { if (!controller.signal.aborted) setAdvice(value); })
        .catch(() => { if (!controller.signal.aborted) setAdvice({status:"unavailable",recommendation:globalSource ? "preserve" : "recompute",reason:"Не удалось оценить снимки. Рекомендация не применяется автоматически.",scene_count:0,source_count:0,sampled_scene_count:0,sampled_valid_pixels:0,estimated_valid_pixels:0}); })
        .finally(() => { if (!controller.signal.aborted) setAdvising(false); });
    }, 300);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [result.id, checkpoint, pipeline, datasetSource, selectedDataset, selectedSample, globalTarget, globalSource]);
  const sending = useRef(false);
  const [requestId] = useState(() => crypto.randomUUID());
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (sending.current) return;
    const values = new FormData(event.currentTarget);
    const request: TrainingContinuationCreate = {
      additional_epochs: Number(values.get("epochs")),
      additional_time_sec: Math.round(Number(values.get("minutes")) * 60),
      early_stopping_patience: Number(values.get("patience")),
      checkpoint,
      pipeline_variant: String(values.get("pipeline_variant") || result.pipeline_variant || "legacy"),
      normalization_policy: (values.get("normalization_policy") || "auto") as typeof normalizationPolicy,
      dataset_source: (values.get("dataset_source") || "original") as typeof datasetSource,
      run_inference_after_training: values.get("run_inference_after_training") === "on",
      secondary_priority: values.get("secondary_priority") === "on",
      request_id: requestId,
    };
    if (request.dataset_source === "dataset") request.dataset_key = String(values.get("dataset_key") || "");
    if (request.dataset_source === "test_sample") request.test_sample_id = String(values.get("test_sample_id") || "");
    sending.current = true;
    setBusy(true);
    try {
      const created = await run(() => apiJson<JobDetail>(`/results/training/${result.id}/continue`, { method: "POST", body: request }));
      if (created) {
        closeModal();
        await reload();
      }
    } finally {
      sending.current = false;
      setBusy(false);
    }
  };
  return <form className="form-stack" onSubmit={submit}>
    <div className="continuation-source">
      <strong>{result.model_name}</strong>
      <span className="muted">{checkpoint === "best" ? `Лучшие веса${result.epoch != null ? ` · эпоха ${result.epoch}` : ""}` : "Последняя завершённая эпоха"}</span>
    </div>
    <fieldset className="continuation-checkpoints" disabled={busy}>
      <legend>От какого чекпойнта учить</legend>
      <label className="field checkbox-field"><input type="radio" name="checkpoint" value="best" checked={checkpoint === "best"} onChange={() => setCheckpoint("best")} />Лучший</label>
      <label className="field checkbox-field"><input type="radio" name="checkpoint" value="last" checked={checkpoint === "last"} disabled={!options.last_checkpoint_available} onChange={() => setCheckpoint("last")} />Последний</label>
      {!options.last_checkpoint_available ? <small className="muted">Последний чекпойнт не сохранён; доступны только лучшие веса.</small> : null}
    </fieldset>
    <label className="field"><span>Конвейер дообучения</span>
      <select name="pipeline_variant" value={pipeline} disabled={busy}
        onChange={event => { setPipeline(event.currentTarget.value as typeof pipeline); setNormalizationPolicy("auto"); }}>
        {pipelineChoices.map(variant => <option key={variant} value={variant}>{trainingPipelineLabel(variant)}{variant === result.pipeline_variant ? " · исходный" : ""}</option>)}
      </select>
      <small className="muted">{pipelineChoices.length === 1 ? "У этой сети доступен только исходный конвейер: другие требуют изменения структуры или смысла выходов." : "Показаны конвейеры с совместимыми весами, выходными каналами и нормализацией декодера."}</small>
    </label>
    {pipeline !== result.pipeline_variant ? <p className="info-box" role="note">Будет создан новый этап от всех весов выбранной сети. Правила разбиения, аугментаций, нормализации, loss и выбора лучших весов берутся из выбранного конвейера. Размер тайла сохраняется, если поддерживается; иначе используется 512 пикселей.</p> : null}
    {globalSource && !globalTarget ? <p className="info-box">Выбран переход с глобальной на оконную нормализацию. Меняется привычный сети масштаб входа; сравните результат на прежних и новых территориях.</p> : null}
    <fieldset className="continuation-checkpoints" disabled={busy}>
      <legend>На каких данных учить</legend>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="original" checked={datasetSource === "original"} onChange={() => setDatasetSource("original")} />Прежний состав</label>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="current" checked={datasetSource === "current"} onChange={() => setDatasetSource("current")} />Актуальный состав</label>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="dataset" checked={datasetSource === "dataset"} disabled={!classDatasets.length} onChange={() => setDatasetSource("dataset")} />Другой датасет класса</label>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="test_sample" checked={datasetSource === "test_sample"} disabled={!classSamples.length} onChange={() => setDatasetSource("test_sample")} />Тестовая разметка класса</label>
    </fieldset>
    {datasetSource === "dataset" ? <label className="field"><span>Датасет класса</span>
      <select name="dataset_key" required disabled={busy} value={selectedDataset} onChange={event => setSelectedDataset(event.currentTarget.value)}>
        {classDatasets.map(item => <option key={item.key} value={item.key}>{item.dataset_name || item.name}</option>)}
      </select></label> : null}
    {datasetSource === "test_sample" ? <><label className="field"><span>Тестовая разметка класса</span>
      <select name="test_sample_id" required disabled={busy} value={selectedSample} onChange={event => setSelectedSample(event.currentTarget.value)}>
        {classSamples.map(item => <option key={item.id} value={item.id}>{item.name} · снимков включено: {item.enabled_image_count}</option>)}
      </select></label>
      <p className="info-box" role="note">В обучение войдут включённые снимки и эталонные полигоны выбранной разметки. После такого дообучения F1 на ней не является независимой оценкой качества. В результатах появится соответствующая пометка.</p></> : null}
    <p className="muted">{datasetSource === "current" || datasetSource === "dataset"
      ? "При фактическом старте берётся опубликованный состав этого датасета: новые снимки и исправленные полигоны учитываются, удалённые снимки исключаются. Личные неопубликованные черновики не используются. После старта снимок разметки фиксируется."
      : datasetSource === "test_sample" ? "При старте фиксируются текущая ревизия тестовой разметки, включённые снимки и эталон. Дальнейшие изменения или удаление разметки не меняют данные уже запущенного обучения."
      : "Используется сохранённый состав и разметка выбранного обучения. Последующие изменения датасета не учитываются."}</p>
    <p className="muted">Выбирайте данные того же класса с совместимыми типами объектов и каналами снимков. При выборе другого датасета результат появится в нём; при выборе тестовой разметки — в датасете исходной сети.</p>
    {globalTarget ? <section aria-label="Нормализация дообучения" className="form-stack">
      {globalSource ? <fieldset className="continuation-checkpoints" disabled={busy}>
        <legend>Глобальная нормализация</legend>
        <label className="field checkbox-field"><input type="radio" name="normalization_policy" value={normalizationPolicy === "auto" ? "auto" : "preserve"}
          checked={normalizationPolicy !== "recompute"} onChange={() => setNormalizationPolicy("preserve")} />Сохранить исходную</label>
        <label className="field checkbox-field"><input type="radio" name="normalization_policy" value="recompute"
          checked={normalizationPolicy === "recompute"} onChange={() => setNormalizationPolicy("recompute")} />Пересчитать на новых train-данных</label>
      </fieldset> : <p className="info-box">У исходной сети нет общих min/max. Переход на глобальную нормализацию потребует новой калибровки по train и может изменить привычный сети масштаб входа.</p>}
      {advising ? <p className="muted" role="status"><LoaderCircle size={14} className="spin" /> Оцениваем объём, разнообразие и диапазоны снимков…</p> : advice ? <div className="info-box" role="status">
        <strong>{advice.status === "unavailable" ? "Предварительная оценка недоступна" : advice.recommendation === "preserve" ? "Рекомендация: сохранить исходную нормализацию" : "Рекомендация: сравнить этап с пересчётом"}</strong>
        <p>{advice.reason}</p>
        {advice.status === "available" ? <small>Снимков: {advice.scene_count ?? 0}; разных исходников: {advice.source_count ?? 0}; проверено снимков: {advice.sampled_scene_count ?? 0}. {globalSource ? `За прежними границами: до ${(100 * Math.max(0,...(advice.clipped_fractions || []))).toFixed(1)}% значений канала.` : ""}</small> : null}
        {advice.warnings?.map(warning => <p className="muted" key={warning}>{warning}</p>)}
        {globalSource && advice.status === "available" && (normalizationPolicy === "recompute" ? "recompute" : "preserve") !== advice.recommendation ? <button type="button" className="secondary" disabled={busy} onClick={() => setNormalizationPolicy(advice.recommendation)}>Применить рекомендацию</button> : null}
      </div> : null}
      {globalSource && normalizationPolicy !== "recompute" ? <small className="muted">Сохраняются точные min/max чекпойнта, даже на маленьком датасете и при смене совместимого глобального профиля.</small> : null}
      {normalizationPolicy === "recompute" ? <p className="info-box">Меняется масштаб входа. Чтобы оценить сохранение прежних знаний, сравните результат на независимых старых и новых территориях. Прежняя сеть останется доступна.</p> : null}
    </section> : null}
    <p className="muted">{pipeline === result.pipeline_variant ? "Параметры исходного конвейера сохраняются. " : "Применяется стандартный профиль выбранного конвейера. "}Новый этап начнётся от выбранных весов; счётчики и оптимизатор запускаются заново. Исходная сеть остаётся в результатах.</p>
    <div className="continuation-fields">
      <label className="field"><span>Ещё времени, мин</span>
        <input name="minutes" type="number" min={1 / 60} step="any" required defaultValue={options.additional_time_sec / 60} disabled={busy} />
      </label>
      <label className="field"><span>Ещё эпох, максимум</span>
        <input name="epochs" type="number" min="1" step="1" required defaultValue={options.additional_epochs} disabled={busy} />
      </label>
      <label className="field"><span>Ранняя остановка</span>
        <input name="patience" type="number" min="1" step="1" required defaultValue={options.early_stopping_patience} disabled={busy} />
        <small className="muted">{pipeline === "next_gen" ? "Проверок валидации" : "Эпох"} без улучшения</small>
      </label>
    </div>
    <p className="muted">Обучение остановится по первому достигнутому пределу. Время учитывает обучение и валидацию; текущая эпоха завершается полностью.</p>
    <div className="continuation-options" role="group" aria-label="Очередь и результат">
      <label className="training-launch-check"><input type="checkbox" name="run_inference_after_training" defaultChecked={options.run_inference_after_training && inferenceAvailable} disabled={busy || !inferenceAvailable} />
        <span>Псевдоразметка после обучения<small>{datasetSource === "test_sample" ? "Все снимки датасета исходной сети, после успешного завершения." : "Все снимки выбранного датасета, после успешного завершения."}</small></span>
      </label>
      {!inferenceAvailable ? <p className="inference-template-warning" role="alert">Классу не назначен шаблон инференса. Чтобы создать псевдоразметку после обучения, <a href="#/templates/inference" onClick={closeModal}>назначьте шаблон</a> в разделе «Инференс».</p> : null}
      <label className="training-launch-check"><input type="checkbox" name="secondary_priority" defaultChecked={options.secondary_priority} disabled={busy} />
        <span>Второстепенный приоритет<small>Обучение и псевдоразметка уступают ресурсы обычным заданиям.</small></span>
      </label>
    </div>
    <div className="button-row">
      <button className="secondary" type="button" onClick={closeModal} disabled={busy}>Отмена</button>
      <button className="primary" type="submit" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{busy ? "Постановка в очередь…" : "Продолжить обучение"}</button>
    </div>
  </form>;
}
