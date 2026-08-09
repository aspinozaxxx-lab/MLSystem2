# Модуль train_pipeline

## Назначение

`train_pipeline` оркестрирует обучение: управляет MLflow run, готовит датасет, создаёт train/val DataLoader и модель, запускает обучение и сохраняет отчёты.

## Публичный интерфейс

- `run_train_pipeline(request: TrainPipelineRequest) -> TrainPipelineResult` — выполнить полный конвейер обучения.

## Публичные контракты

- `TrainPipelineError` — невосстановимая ошибка конвейера.
- `PipelineStatus` — `succeeded|failed`.
- `ModuleTiming` — `module`, `elapsed_sec`, `details`.
- `TimingReport` — `total_pipeline_time_sec`, `modules`.
- `PipelineReport` — `status`, `message`, `dataset_status`, `errors`, `warnings`, `artifacts`.
- `TrainPipelineRequest` — optional `run_name`.
- `TrainPipelineResult` — `status`, `mlflow_run`, `timings`, `report`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — текущие настройки и путь YAML.
- `mlflow_adapter.api` — run, метрики, отчёты и артефакты.
- `dataset_preparing.api` — получить проверенный `PreparedDataset` со сценами.
- `tile_preparation.api` — создать train/val loaders.
- `models.api` — создать модель либо загрузить checkpoint.
- `train.api` — выполнить обучение.

## Алгоритм работы и его особенности

Конвейер передаёт в подготовку ожидаемые каналы и `uint8`, затем преобразует каждый `PreparedScene` в `TileSceneSource`. Оба loader получают один `TileSplitRequest`; split выполняется по окнам независимых TIFF. Legacy binary передаёт общие positive/hard-negative GeoJSON, per-image — локальный GeoJSON каждой сцены, multiclass — `class_annotations`; binary val также получает instance masks. В MLflow каталог `dataset/` попадают TXT/GeoJSON legacy либо все GeoJSON из `annotations_dir`. Счётчики loader фиксируют сцены, valid-footprint, sampling и cache. Затем создаётся/загружается модель, `train_model` получает настройки и progress sink, а конвейер пишет epoch metrics, checkpoints, tile/timing/pipeline reports и корректно завершает MLflow run.
