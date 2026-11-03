import { LoaderCircle, Play } from "lucide-react";
import { type FormEvent, useRef, useState } from "react";
import { apiJson } from "./api/client";
import type { DatasetInfo, JobDetail, TestSampleCard, TrainingContinuationCreate, TrainingContinuationOptions, TrainingResultInfo } from "./api/types";

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
  const [datasetSource, setDatasetSource] = useState<NonNullable<TrainingContinuationCreate["dataset_source"]>>("original");
  const classDatasets = datasets.filter(item => options.class_key && item.class_key === options.class_key && !item.is_custom);
  const classSamples = testSamples.filter(item => options.class_key && item.class_key === options.class_key);
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
    <fieldset className="continuation-checkpoints" disabled={busy}>
      <legend>На каких данных учить</legend>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="original" checked={datasetSource === "original"} onChange={() => setDatasetSource("original")} />Прежний состав</label>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="current" checked={datasetSource === "current"} onChange={() => setDatasetSource("current")} />Актуальный состав</label>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="dataset" checked={datasetSource === "dataset"} disabled={!classDatasets.length} onChange={() => setDatasetSource("dataset")} />Другой датасет класса</label>
      <label className="field checkbox-field"><input type="radio" name="dataset_source" value="test_sample" checked={datasetSource === "test_sample"} disabled={!classSamples.length} onChange={() => setDatasetSource("test_sample")} />Тестовая разметка класса</label>
    </fieldset>
    {datasetSource === "dataset" ? <label className="field"><span>Датасет класса</span>
      <select name="dataset_key" required disabled={busy} defaultValue={classDatasets.find(item => item.key !== result.dataset_key)?.key || classDatasets[0]?.key || ""}>
        {classDatasets.map(item => <option key={item.key} value={item.key}>{item.dataset_name || item.name}</option>)}
      </select></label> : null}
    {datasetSource === "test_sample" ? <><label className="field"><span>Тестовая разметка класса</span>
      <select name="test_sample_id" required disabled={busy} defaultValue={classSamples[0]?.id || ""}>
        {classSamples.map(item => <option key={item.id} value={item.id}>{item.name} · снимков включено: {item.enabled_image_count}</option>)}
      </select></label>
      <p className="info-box" role="note">В обучение войдут включённые снимки и эталонные полигоны выбранной разметки. После такого дообучения F1 на ней не является независимой оценкой качества. В результатах появится соответствующая пометка.</p></> : null}
    <p className="muted">{datasetSource === "current" || datasetSource === "dataset"
      ? "При фактическом старте берётся опубликованный состав этого датасета: новые снимки и исправленные полигоны учитываются, удалённые снимки исключаются. Личные неопубликованные черновики не используются. После старта снимок разметки фиксируется."
      : datasetSource === "test_sample" ? "При старте фиксируются текущая ревизия тестовой разметки, включённые снимки и эталон. Дальнейшие изменения или удаление разметки не меняют данные уже запущенного обучения."
      : "Используется сохранённый состав и разметка выбранного обучения. Последующие изменения датасета не учитываются."}</p>
    <p className="muted">Выбирайте данные того же класса с совместимыми типами объектов и каналами снимков. При выборе другого датасета результат появится в нём; при выборе тестовой разметки — в датасете исходной сети.</p>
    <p className="muted">Остальные параметры обучения сохраняются. Новый этап начнётся от выбранных весов; счётчики и оптимизатор запускаются заново. Исходная сеть остаётся в результатах.</p>
    <div className="continuation-fields">
      <label className="field"><span>Ещё времени, мин</span>
        <input name="minutes" type="number" min={1 / 60} step="any" required defaultValue={options.additional_time_sec / 60} disabled={busy} />
      </label>
      <label className="field"><span>Ещё эпох, максимум</span>
        <input name="epochs" type="number" min="1" step="1" required defaultValue={options.additional_epochs} disabled={busy} />
      </label>
      <label className="field"><span>Ранняя остановка</span>
        <input name="patience" type="number" min="1" step="1" required defaultValue={options.early_stopping_patience} disabled={busy} />
        <small className="muted">{result.pipeline_variant === "next_gen" ? "Проверок валидации" : "Эпох"} без улучшения</small>
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
