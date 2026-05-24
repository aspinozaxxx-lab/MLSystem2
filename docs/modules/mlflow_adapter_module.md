# Модуль mlflow_adapter

## Назначение

`mlflow_adapter` изолирует работу с MLflow от остальных модулей: открывает запуск, пишет отчеты, метрики, артефакты и завершает запуск. Модуль не принимает решений о качестве модели и не управляет обучением.

## Публичный интерфейс

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
- `MLflowRunStatus` - статусы `FINISHED`, `FAILED`, `KILLED`.
- `MLflowStartRunRequest` - поля `enabled`, `tracking_uri`, `experiment_name`, `run_name`, `tags`.
- `MLflowRunRef` - поля `run_id`, `experiment_name`, `tracking_uri`, `active`.
- `MLflowArtifactRef` - ссылка на артефакт MLflow.

## Список используемых данным модулем модулей и с какой целью

- `mlflow` - создать run, записать metrics/artifacts и завершить run.
- `dataset_preparing.contracts` - тип отчета подготовки датасета.
- `train.contracts` - типы `EpochMetrics` и `TrainResult`.
- `train_pipeline.contracts` - типы итоговых отчетов и timing report.

## Алгоритм работы и его особенности

`start_run` подключается к `tracking_uri`, выбирает experiment и запускает run. Если `request.run_name` задан, имя используется как есть. Если имя не задано и в tags есть `class`, адаптер строит имя вида `{class}_{DDMM}_{номер}`: например, `deforestation_2305_1`. Номер считается по уже существующим run за тот же день и класс. Если поиск run недоступен, используется номер `1`.

`log_run_config` сохраняет YAML как `config/train_config.yaml`. `log_tile_preparation` сохраняет отчет как `reports/tile_preparation.json`.

`log_training_epoch` вызывается из `train_pipeline` через progress sink на событии `epoch_finished`. Он логирует с `step=metrics.epoch`: `train/loss`, `train/optimizer_steps`, `train/skipped_optimizer_steps`, `val/loss`, `val/pixel_precision`, `val/pixel_recall`, `val/pixel_f1`, `val/positive_pixels`, `val/pred_positive_pixels`, `val/true_positive`, `val/false_positive`, `val/false_negative`, `train/epoch_time_sec`. Это обеспечивает появление метрик в MLflow во время долгого обучения.

`log_training_metrics` не дублирует per-epoch метрики. Он пишет только итоговые значения: `train/epochs_total`, `train/training_time_sec`, `val/best_pixel_f1`, `val/final_pixel_f1`. `log_training_artifacts` пишет полную историю обучения в JSON и сохраняет существующие best/final checkpoint-файлы.
