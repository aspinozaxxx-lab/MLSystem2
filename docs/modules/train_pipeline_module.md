# Модуль train_pipeline

## Назначение

`train_pipeline` оркестрирует обучение: получает настройки, управляет MLflow run, готовит датасет, создает train/val DataLoader, создает или загружает модель, запускает `train_model`, пишет отчеты и артефакты.

## Публичный интерфейс

- `run_train_pipeline(request: TrainPipelineRequest) -> TrainPipelineResult` - запускает полный конвейер обучения.

## Публичные контракты

- `TrainPipelineError` - невосстановимая ошибка конвейера обучения.
- `PipelineStatus` - статусы `succeeded` и `failed`.
- `ModuleTiming` - поля `module`, `elapsed_sec`, `details`.
- `TimingReport` - поля `total_pipeline_time_sec`, `modules`.
- `PipelineReport` - поля `status`, `message`, `dataset_status`, `errors`, `warnings`, `artifacts`.
- `TrainPipelineRequest` - поле `run_name`.
- `TrainPipelineResult` - поля `status`, `mlflow_run`, `timings`, `report`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` - получить текущие настройки через `get_settings` и путь YAML через `get_settings_path`.
- `mlflow_adapter.api` - открыть run, писать config, отчеты, live epoch metrics и итоговые артефакты.
- `dataset_preparing.api` - подготовить split и получить `train_vrt_xml`, `val_vrt_xml`, `annotation_file`.
- `tile_preparation.api` - создать `train_loader` и `val_loader`.
- `models.api` - создать модель или загрузить checkpoint.
- `train.api` - обучить модель на готовых DataLoader.

## Алгоритм работы и его особенности

`run_train_pipeline` получает settings, открывает MLflow run, пишет YAML-конфиг запуска и вызывает `dataset_preparing`. После отчета подготовки датасета создает train/val DataLoader и оборачивает их внутренним счетчиком tile batches, augmented/positive tiles, диагностик VRT source rects и valid-footprint filter. Если `settings.train.initial_checkpoint_uri` задан, вызывается `models.load_checkpoint` с `LoadCheckpointRequest`; иначе вызывается `models.create_model`.

В `TrainConfig` передаются train-гиперпараметры из settings, включая диагностические batch limits. В `train_model` передается progress sink. На событии `epoch_finished` sink вызывает `mlflow_adapter.log_training_epoch`, чтобы MLflow обновлялся сразу после каждой эпохи. Время live MLflow logging учитывается в timing как `mlflow_logging`.

Если `TrainPipelineRequest.run_name` не задан, в MLflow tags передается `class=Path(settings.dataset.annotation_file).stem`, чтобы `mlflow_adapter` мог сгенерировать имя вида `{class}_{DDMM}_{номер}`. После завершения обучения конвейер пишет итоговые metrics, training artifacts, `reports/tile_preparation.json`, timing report и pipeline report. Tile report содержит `smart_tiling_enabled`, `positive_factor`, source-rect diagnostics, valid-footprint diagnostics, estimated positive/negative tiles и observed counters.
