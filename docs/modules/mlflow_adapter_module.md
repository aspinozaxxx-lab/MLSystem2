# Модуль mlflow_adapter

## Назначение

`mlflow_adapter` — единственный модуль, которому разрешено импортировать MLflow. Он отвечает за
создание и закрытие запусков, логирование метрик, параметров, артефактов, артефактов модели или
чекпойнта, отчета подготовки датасета, отчета времени, итогового отчета конвейера и метрик
обучения по эпохам.

## Публичный интерфейс

- `start_run(request: MLflowStartRunRequest) -> MLflowRunRef` — открыть или отключить MLflow run.
- `log_dataset_preparation(run, report) -> None` — записать отчет подготовки датасета.
- `log_training_metrics(run, result) -> None` — записать train/val metrics по эпохам и итоговые scalar.
- `log_training_artifacts(run, result) -> None` — записать history JSON и best/final checkpoint, если файлы существуют.
- `log_timing_report(run, report) -> None` — записать timing report.
- `log_pipeline_report(run, report) -> None` — записать итоговый отчет.
- `end_run(run, status) -> None` — завершить run.

## Публичные контракты

- `MLflowAdapterError` — ошибка адаптера.
- `MLflowRunStatus` — `FINISHED`, `FAILED`, `KILLED`.
- `MLflowStartRunRequest` — поля `enabled`, `tracking_uri`, `experiment_name`, `run_name`, `tags`.
- `MLflowRunRef` — поля `run_id`, `experiment_name`, `tracking_uri`, `active`.
- `MLflowArtifactRef` — поля `uri`, `artifact_path`.

## Список используемых данным модулем модулей и с какой целью

- `dataset_preparing.contracts` — сериализация отчета подготовки.
- `train.contracts` — сериализация `TrainResult` и epoch metrics.
- `train_pipeline.contracts` — сериализация timing и pipeline reports.

## Алгоритм работы и его особенности

MLflow импортируется только внутри этого модуля. При disabled run функции становятся no-op. Метрики обучения пишутся по эпохам: `train/loss`, `val/loss`, `val/pixel_precision`, `val/pixel_recall`, `val/pixel_f1`, `train/epoch_time_sec`; итоговые scalar: `train/epochs_total`, `train/training_time_sec`, `val/best_pixel_f1`, `val/final_pixel_f1`. Артефакты обучения включают history JSON и существующие best/final checkpoint.
