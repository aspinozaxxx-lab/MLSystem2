# Модуль mlflow_adapter

## Назначение

`mlflow_adapter` — единственный модуль, которому разрешено импортировать MLflow. Он отвечает за
создание и закрытие запусков, логирование метрик, параметров, артефактов, артефактов модели или
чекпойнта, отчета подготовки датасета, отчета времени, итогового отчета конвейера и метрик
обучения по эпохам.

## Публичный интерфейс

- `start_run(request: MLflowStartRunRequest) -> MLflowRunRef`
- `log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None`
- `log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None`
- `log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None`
- `log_timing_report(run: MLflowRunRef, report: TimingReport) -> None`
- `log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None`
- `end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None`

Параметры:

- `request`: tracking URI, имя experiment, имя run, tags и флаг enabled.
- `run`: активная или отключенная ссылка на запуск MLflow.
- `report`: типизированный DTO отчета.
- `result`: DTO результата обучения.
- `status`: финальный статус запуска.

## Контракты

`MLflowStartRunRequest`, `MLflowRunRef`, `MLflowRunStatus`, `MLflowArtifactRef`,
`MLflowAdapterError`.

## Выходные артефакты

Обязательные метрики: `train/f1_pixel`, `train/epoch_time_sec`, `train/epochs_total`,
`train/training_time_sec`.

Обязательные артефакты: `reports/dataset_preparation.json`, `reports/pipeline_timings.json`,
`reports/pipeline_summary.json`, `reports/training_history_full.json`, а также артефакты обученной
модели или чекпойнта внутри `model/` или `checkpoints/`.

## Что модуль НЕ делает

Не обучает, не готовит датасет, не готовит тайлы и не управляет потоком выполнения конвейера.

## Разрешенные зависимости

Ленивый импорт MLflow внутри приватных функций клиента, публичные контракты модулей-производителей
и стандартные вспомогательные средства JSON/tempfile.

## Запрещенные пересечения

Не импортирует приватные файлы обучения, подготовки датасета, подготовки тайлов и модулей
конвейеров.

## MLflow

Владеет всем прямым взаимодействием с MLflow.

## Временное профилирование

Логирует отчеты времени, созданные модулями конвейеров.
