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
- `dataset_preparing.api` - подготовить общий tile pool и получить `train_vrt_xml`, `val_vrt_xml`, `pool_vrt_xml`, `annotation_file`, optional `hard_negative_annotation_file` или `class_annotations`.
- `tile_preparation.api` - создать `train_loader` и `val_loader`.
- `models.api` - создать модель или загрузить checkpoint.
- `train.api` - обучить модель на готовых DataLoader.

## Алгоритм работы и его особенности

`run_train_pipeline` получает settings, открывает MLflow run, пишет YAML-конфиг запуска и вызывает `dataset_preparing`. В запрос всегда передаются `expected_band_count=settings.train.input_channels` и `expected_dtype=uint8`. Если `settings.dataset.classes` непустой, передается multiclass список `DatasetClassRequest` с optional class-level `hard_negative_annotation_file`; иначе — binary `scenes_file`, `annotation_file` и optional `hard_negative_annotation_file`. После успешного отчета конвейер сохраняет исходные txt/geojson файлы разметки в MLflow artifacts `dataset/`, включая hard-negative GeoJSON при наличии. После этого создаются train/val DataLoader: для multiclass в `TileDataloaderRequest` передаются `class_annotations`, для binary — `annotation_file` и optional `hard_negative_annotation_file`. Binary val request дополнительно получает `include_object_instances=true`. Оба loader получают общий `pool_vrt_xml` и одинаковый `TileSplitRequest`. Train request получает `settings.train.max_train_batches_per_epoch`; val request получает `settings.train.max_val_batches_per_epoch`, чтобы ограничить balanced subset до оценки RAM и материализации.

DataLoader оборачивается внутренним счетчиком tile batches, category counters, augmented counters, диагностик valid-footprint filter и cache metadata. Mask из loader является единой supervision mask: `-1` hard negative, `0` background, `1..N` positive; отдельная hard-negative pixel mask в batch meta не создается. Если `settings.train.initial_checkpoint_uri` задан, вызывается `models.load_checkpoint` с `LoadCheckpointRequest`; иначе вызывается `models.create_model`.

В `ModelSpec` передаются `settings.train.input_channels` и `settings.train.output_channels`; для multiclass число выходов равно `len(settings.dataset.classes)+1`. В `TrainConfig` передаются train-гиперпараметры из settings, включая `task`, `quality_metric`, `hard_negative_weight`, диагностические batch limits и `class_slugs`. В `train_model` передается progress sink. На событии `epoch_finished` sink вызывает `mlflow_adapter.log_training_epoch`, чтобы MLflow обновлялся сразу после каждой эпохи. Время live MLflow logging учитывается в timing как `mlflow_logging`.

Если `TrainPipelineRequest.run_name` не задан, MLflow-имя формируется по датасету. После завершения обучения конвейер пишет metrics, training artifacts, `reports/tile_preparation.json`, timing report и pipeline report. Tile report на верхнем уровне содержит `input_channels`, `input_dtype`, заданные tile factors, worker/prefetch параметры и pool/split diagnostics. Для каждого split фиксируются sampling, cache mode, выбранные batch/tiles, оценка RAM, fallback reason и observed category counters. Для val `sampling_mode` равен `cached_balanced` или `lazy_balanced`; train factors к нему не применяются.
