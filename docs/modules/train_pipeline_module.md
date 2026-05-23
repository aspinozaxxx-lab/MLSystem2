# Модуль train_pipeline

## Назначение

`train_pipeline` оркестрирует обучение: загрузку настроек, жизненный цикл запуска MLflow, подготовку
датасета, создание train/val DataLoader, обучение, замеры времени и сборку итогового отчета.

## Публичный интерфейс

- `run_train_pipeline(request: TrainPipelineRequest) -> TrainPipelineResult` — запускает полный конвейер обучения.

## Публичные контракты

- `TrainPipelineError` — невосстановимая ошибка конвейера обучения.
- `PipelineStatus` — статус результата: `succeeded` или `failed`.
- `ModuleTiming` — поля `module`, `elapsed_sec`, `details`.
- `TimingReport` — поля `total_pipeline_time_sec`, `modules`.
- `PipelineReport` — поля `status`, `message`, `dataset_status`, `errors`, `warnings`, `artifacts`.
- `TrainPipelineRequest` — поле `run_name`.
- `TrainPipelineResult` — поля `status`, `mlflow_run`, `timings`, `report`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — получить текущие настройки через `get_settings`.
- `mlflow_adapter.api` — управлять запуском MLflow и записывать результаты.
- `dataset_preparing.api` — подготовить split и получить `train_vrt_xml`, `val_vrt_xml`, `annotation_file`.
- `tile_preparation.api` — создать `train_loader` и `val_loader`.
- `models.api` — создать модель или загрузить checkpoint.
- `train.api` — обучить модель на готовых DataLoader.

## Алгоритм работы и его особенности

Получает settings через `get_settings`, открывает запуск MLflow, вызывает `dataset_preparing` по `settings.dataset`. После успешной подготовки вызывает `create_tile_dataloader` для train/val VRT с `annotation_file` и `train.batch_size`. Если `settings.train.initial_checkpoint_uri` задан, вызывает `models.load_checkpoint` с `ModelSpec`; иначе вызывает `models.create_model`. В `TrainConfig` передает train-гиперпараметры из settings, затем вызывает `train` с готовыми loader. Ошибки и итоговые отчеты фиксируются через `mlflow_adapter`.
