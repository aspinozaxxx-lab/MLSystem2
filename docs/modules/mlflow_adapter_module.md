# Модуль mlflow_adapter

## Назначение

`mlflow_adapter` изолирует работу с MLflow от остальных модулей: открывает запуск, пишет отчеты, метрики, артефакты и завершает запуск. Модуль не принимает решений о качестве модели и не управляет обучением.

## Публичный интерфейс

- `list_experiments(tracking_uri: str) -> list[MLflowExperiment]` - возвращает доступные experiments из указанного MLflow tracking URI.
- `create_experiment(request: MLflowExperimentRequest) -> MLflowExperiment` - создает experiment или возвращает существующий с тем же именем.
- `get_best_training_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint | None` - читает `val/quality_f1` для запуска с tag `quality_metric`, использует pixel fallback для старого запуска и возвращает эпоху, значение F1, threshold и ссылку на `checkpoints/best.pt`.
- `get_usable_training_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint | None` - дополнительно требует статус MLflow `FINISHED`, threshold и фактический артефакт `checkpoints/best.pt`; используется при выборе модели для нового инференса.
- `get_finished_run_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> MLflowRunArtifactInfo | None` - возвращает точный файловый артефакт только завершённого запуска без требования training-метрик.
- `get_training_epoch_progress(tracking_uri: str, run_id: str) -> MLflowTrainingProgress` - возвращает число завершённых эпох по истории `train/epoch_time_sec`.
- `download_run_artifact(tracking_uri: str, run_id: str, artifact_path: str, dst_dir: str | Path) -> MLflowDownloadedArtifact` - скачивает артефакт запуска в локальную рабочую папку вызывающего модуля.
- `start_run(request: MLflowStartRunRequest) -> MLflowRunRef` - создает или отключает MLflow run.
- `log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None` - пишет отчет подготовки датасета.
- `log_dataset_artifacts(run: MLflowRunRef, files: dict[str, str | Path]) -> None` - пишет исходные txt/geojson файлы датасета в папку `dataset` артефактов MLflow.
- `log_tile_preparation(run: MLflowRunRef, report: dict[str, object]) -> None` - пишет отчет подготовки тайлов.
- `log_run_config(run: MLflowRunRef, config_path: str | Path) -> None` - пишет YAML-конфиг запуска.
- `log_training_epoch(run: MLflowRunRef, metrics: EpochMetrics) -> None` - пишет перечисленные ниже loss, quality, pixel, optional object и threshold метрики одной эпохи и tag `quality_metric`.
- `log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None` - пишет итоговые `train/epochs_total`, `train/training_time_sec`, `train/best_quality_f1`, `train/best_threshold_pixel_f1`.
- `log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None` - пишет историю обучения и checkpoint-файлы.
- `log_timing_report(run: MLflowRunRef, report: TimingReport) -> None` - пишет отчет времени выполнения.
- `log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None` - пишет итоговый отчет конвейера.
- `end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None` - завершает MLflow run.
- `mark_run_killed(tracking_uri: str, run_id: str) -> None` - переводит активный запуск в `KILLED`.

## Публичные контракты

- `MLflowAdapterError` - ошибка адаптера MLflow.
- `MLflowExperiment` - поля `experiment_id`, `name`, `lifecycle_stage`.
- `MLflowExperimentRequest` - поля `tracking_uri`, `name`.
- `MLflowRunStatus` - статусы `FINISHED`, `FAILED`, `KILLED`.
- `MLflowStartRunRequest` - поля `enabled`, `tracking_uri`, `experiment_name`, `dataset`, `run_name`, `tags`.
- `MLflowRunRef` - поля `run_id`, `experiment_name`, `tracking_uri`, `active`.
- `MLflowArtifactRef` - ссылка на артефакт MLflow.
- `MLflowBestCheckpoint` - поля `tracking_uri`, `run_id`, `metric_name`, `f1_score`, `epoch`, `artifact_path`, `artifact_uri`, `threshold`.
- `MLflowDownloadedArtifact` - поля `run_id`, `artifact_path`, `local_path`.
- `MLflowRunArtifactInfo` - поля `tracking_uri`, `run_id`, `artifact_path`, `artifact_uri` произвольного файла завершённого запуска.
- `MLflowTrainingProgress` - поле `completed_epochs`.

## Список используемых данным модулем модулей и с какой целью

- `mlflow` - прочитать/создать experiments, создать run, записать metrics/artifacts, скачать artifact запуска и завершить run.
- `dataset_preparing.contracts` - тип отчета подготовки датасета.
- `train.contracts` - типы `EpochMetrics` и `TrainResult`.
- `train_pipeline.contracts` - типы итоговых отчетов и timing report.

### Публикуемые MLflow-метрики

Список MLflow-метрик является частью архитектурного контракта модуля. Любое добавление, удаление или переименование метрики требует предварительного изменения этого документа. Каждый публичный API, который пишет MLflow-метрики, обязан явно перечислять имена записываемых метрик в разделе публичного интерфейса.

`log_training_epoch` пишет только epoch-метрики с `step=metrics.epoch`:

- `train/loss`
- `val/loss`
- `val/best_threshold`
- `val/best_pixel_threshold`
- `val/quality_f1`
- `val/quality_precision`
- `val/quality_recall`
- `val/best_threshold_pixel_f1`
- `val/pixel_f1`
- `val/pixel_precision`
- `val/pixel_recall`
- `val/best_threshold_precision`
- `val/best_threshold_recall`
- `val/best_threshold_object_f1` — только при наличии object validation
- `val/best_threshold_object_precision` — только при наличии object validation
- `val/best_threshold_object_recall` — только при наличии object validation
- `val/object_f1` — только при наличии object validation
- `val/object_precision` — только при наличии object validation
- `val/object_recall` — только при наличии object validation
- `train/epoch_time_sec`

`log_training_metrics` пишет только итоговые run-level метрики без `step`:

- `train/epochs_total`
- `train/training_time_sec`
- `train/best_quality_f1`
- `train/best_threshold_pixel_f1`

## Алгоритм работы и его особенности

`start_run` подключается к `tracking_uri`, выбирает experiment и запускает run. Если `request.dataset` задан, адаптер сначала проверяет наличие одноименного MLflow dataset в experiment и создает его при отсутствии, затем добавляет MLflow tag `dataset` и логирует MLflow input dataset через `mlflow.log_input`; имя, source и tag dataset равны переданному значению. Адаптер не вычисляет имя датасета и не ходит в папки датасета; вызывающий модуль должен передать готовое имя без расширения `.geojson`. Если `request.run_name` задан, имя используется как есть. Если имя не задано и в tags есть `class`, адаптер строит имя вида `{class}_{DDMM}_{номер}`: например, `deforestation_2305_1`. Номер считается по уже существующим run за тот же день и класс. Если поиск run недоступен, используется номер `1`.

`log_run_config` сохраняет YAML как `config/train_config.yaml`. `log_dataset_artifacts` сохраняет исходные файлы разметки в `dataset/`. `log_tile_preparation` сохраняет отчет как `reports/tile_preparation.json`.

`log_training_epoch` вызывается из `train_pipeline` через progress sink на событии `epoch_finished`. Он логирует только метрики, перечисленные в разделе "Публикуемые MLflow-метрики". Это обеспечивает появление HPO-сигналов в MLflow во время долгого обучения без диагностического шума.

`log_training_metrics` не дублирует per-epoch метрики. `train/best_quality_f1` равна максимуму выбранной метрики, а совместимая `train/best_threshold_pixel_f1` — максимуму pixel F1. `log_training_artifacts` пишет полную историю обучения в JSON и сохраняет существующие best/final checkpoint-файлы.

`get_best_training_checkpoint` для запуска с tag `quality_metric=pixel|objects` читает `val/quality_f1`, потому что `best.pt` сохраняется train-модулем по этой же метрике. Запуск без tag считается старым и читается по `val/best_threshold_pixel_f1`; если новый metric history отсутствует, также применяется pixel fallback. При равном F1 выбирается более ранняя эпоха. `val/best_threshold` той же эпохи возвращается вместе с checkpoint summary.

`get_finished_run_artifact` проверяет статус запуска, точное имя и то, что найден именно файл. `download_run_artifact` оборачивает публичный MLflow client и нужен вызывающим модулям, которым требуется локальный файл артефакта без прямого импорта MLflow.
