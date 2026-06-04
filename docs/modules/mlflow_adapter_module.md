# Модуль mlflow_adapter

## Назначение

`mlflow_adapter` изолирует работу с MLflow от остальных модулей: открывает запуск, пишет отчеты, метрики, артефакты и завершает запуск. Модуль не принимает решений о качестве модели и не управляет обучением.

## Публичный интерфейс

- `list_experiments(tracking_uri: str) -> list[MLflowExperiment]` - возвращает доступные experiments из указанного MLflow tracking URI.
- `create_experiment(request: MLflowExperimentRequest) -> MLflowExperiment` - создает experiment или возвращает существующий с тем же именем.
- `get_best_training_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint | None` - читает history метрики `val/best_threshold_pixel_f1` и возвращает эпоху, значение F1, threshold и ссылку на `checkpoints/best.pt`.
- `download_run_artifact(tracking_uri: str, run_id: str, artifact_path: str, dst_dir: str | Path) -> MLflowDownloadedArtifact` - скачивает артефакт запуска в локальную рабочую папку вызывающего модуля.
- `start_run(request: MLflowStartRunRequest) -> MLflowRunRef` - создает или отключает MLflow run.
- `log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None` - пишет отчет подготовки датасета.
- `log_tile_preparation(run: MLflowRunRef, report: dict[str, object]) -> None` - пишет отчет подготовки тайлов.
- `log_run_config(run: MLflowRunRef, config_path: str | Path) -> None` - пишет YAML-конфиг запуска.
- `log_training_epoch(run: MLflowRunRef, metrics: EpochMetrics) -> None` - пишет метрики одной эпохи сразу после ее завершения.
- `log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None` - пишет итоговые метрики обучения.
- `log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None` - пишет историю обучения и checkpoint-файлы.
- `log_timing_report(run: MLflowRunRef, report: TimingReport) -> None` - пишет отчет времени выполнения.
- `log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None` - пишет итоговый отчет конвейера.
- `end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None` - завершает MLflow run.

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

## Список используемых данным модулем модулей и с какой целью

- `mlflow` - прочитать/создать experiments, создать run, записать metrics/artifacts, скачать artifact запуска и завершить run.
- `dataset_preparing.contracts` - тип отчета подготовки датасета.
- `train.contracts` - типы `EpochMetrics` и `TrainResult`.
- `train_pipeline.contracts` - типы итоговых отчетов и timing report.

## Алгоритм работы и его особенности

`start_run` подключается к `tracking_uri`, выбирает experiment и запускает run. Если `request.dataset` задан, адаптер сначала проверяет наличие одноименного MLflow dataset в experiment и создает его при отсутствии, затем добавляет MLflow tag `dataset` и логирует MLflow input dataset через `mlflow.log_input`; имя, source и tag dataset равны переданному значению. Адаптер не вычисляет имя датасета и не ходит в папки датасета; вызывающий модуль должен передать готовое имя без расширения `.geojson`. Если `request.run_name` задан, имя используется как есть. Если имя не задано и в tags есть `class`, адаптер строит имя вида `{class}_{DDMM}_{номер}`: например, `deforestation_2305_1`. Номер считается по уже существующим run за тот же день и класс. Если поиск run недоступен, используется номер `1`.

`log_run_config` сохраняет YAML как `config/train_config.yaml`. `log_tile_preparation` сохраняет отчет как `reports/tile_preparation.json`.

`log_training_epoch` вызывается из `train_pipeline` через progress sink на событии `epoch_finished`. Он логирует с `step=metrics.epoch`: `train/loss`, `train/loss_focal`, `train/loss_tversky`, `train/loss_bce`, `train/loss_dice`, `train/optimizer_steps`, `train/skipped_optimizer_steps`, `val/loss`, `val/pixel_precision`, `val/pixel_recall`, `val/pixel_f1`, `val/macro_f1`, `val/mean_iou`, `val/pixel_accuracy`, `val/positive_pixels`, `val/pred_positive_pixels`, `val/true_positive`, `val/false_positive`, `val/false_negative`, `val/best_threshold`, `val/best_threshold_pixel_f1`, `val/best_threshold_precision`, `val/best_threshold_recall`, `val/prob_mean`, `val/prob_min`, `val/prob_max`, `val/prob_p50`, `val/prob_p90`, `val/prob_p99`, `val/prob_p999`, `val/prob_positive_mean`, `val/prob_positive_p50`, `val/prob_positive_p90`, `val/prob_positive_p99`, `val/prob_negative_mean`, `val/prob_negative_p50`, `val/prob_negative_p90`, `val/prob_negative_p99`, `val/sweep_*_f1`, `val/sweep_*_precision`, `val/sweep_*_recall`, `val/<slug>/f1`, `val/<slug>/iou`, `val/<slug>/precision`, `val/<slug>/recall`, `val/<slug>/support_pixels`, `train/epoch_time_sec`. Неиспользуемые loss components и optional multiclass metrics не логируются. Это обеспечивает появление метрик в MLflow во время долгого обучения.

`log_training_metrics` не дублирует per-epoch метрики. Он пишет только итоговые значения: `train/epochs_total`, `train/training_time_sec`, `val/best_pixel_f1`, `val/final_pixel_f1`, а для multiclass дополнительно `val/best_macro_f1`, `val/final_macro_f1`. `log_training_artifacts` пишет полную историю обучения в JSON и сохраняет существующие best/final checkpoint-файлы.

`get_best_training_checkpoint` читает историю `val/best_threshold_pixel_f1`, потому что `best.pt` сохраняется train-модулем по этой же метрике. При равном F1 выбирается более ранняя эпоха, что соответствует сохранению checkpoint только при строгом улучшении. Если в MLflow есть `val/best_threshold` на той же эпохе, значение возвращается вместе с checkpoint summary.

`download_run_artifact` оборачивает публичный MLflow client и нужен вызывающим модулям, которым требуется локальный файл артефакта без прямого импорта MLflow.
