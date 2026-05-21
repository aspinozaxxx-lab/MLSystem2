# Модуль train_pipeline

## Назначение

`train_pipeline` оркестрирует обучение: загрузку настроек, жизненный цикл запуска MLflow, подготовку
датасета, создание train/val DataLoader, обучение, замеры времени и сборку итогового отчета.

## Публичный интерфейс

- `run_train_pipeline(request: TrainPipelineRequest) -> TrainPipelineResult` — запускает полный конвейер обучения по YAML-конфигу.

## Публичные контракты

- `TrainPipelineError` — невосстановимая ошибка конвейера обучения.
- `PipelineStatus` — статус результата: `succeeded` или `failed`.
- `ModuleTiming` — поля `module`, `elapsed_sec`, `details`.
- `TimingReport` — поля `total_pipeline_time_sec`, `modules`.
- `PipelineReport` — поля `status`, `message`, `dataset_status`, `errors`, `warnings`, `artifacts`.
- `TrainPipelineRequest` — поля `config_path`, `run_name`.
- `TrainPipelineResult` — поля `status`, `mlflow_run`, `timings`, `report`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — загрузить настройки.
- `mlflow_adapter.api` — управлять запуском MLflow и записывать результаты.
- `dataset_preparing.api` — подготовить split и получить `train_vrt_xml`, `val_vrt_xml`, `annotation_file`.
- `tile_preparation.api` — создать `train_loader` и `val_loader`.
- `models.api` — создать модель.
- `train.api` — обучить модель на готовых DataLoader.

## Алгоритм работы и его особенности

Загружает settings, открывает запуск MLflow, вызывает `dataset_preparing`. После успешной подготовки вызывает `create_tile_dataloader` два раза: для `train_vrt_xml` с `mode="train"` и для `val_vrt_xml` с `mode="val"`. В оба запроса передает `config_path`, `annotation_file` и `train.batch_size`. Затем создает модель и вызывает `train`, передавая готовые `train_loader` и `val_loader`. Ошибки и итоговые отчеты фиксируются через `mlflow_adapter`.
