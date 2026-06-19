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

`run_train_pipeline` получает settings, открывает MLflow run, пишет YAML-конфиг запуска и вызывает `dataset_preparing`. Если `settings.dataset.classes` непустой, в `DatasetPreparationRequest` передается multiclass список `DatasetClassRequest` с optional class-level `hard_negative_annotation_file`; иначе передаются binary `scenes_file`, `annotation_file` и optional `hard_negative_annotation_file`. После успешного отчета подготовки датасета конвейер сохраняет исходные txt/geojson файлы разметки в MLflow artifacts `dataset/`, включая hard-negative GeoJSON при наличии. После этого создаются train/val DataLoader: для multiclass в `TileDataloaderRequest` передаются `class_annotations`, для binary - `annotation_file` и optional `hard_negative_annotation_file`. Оба loader получают общий `pool_vrt_xml` и одинаковый `TileSplitRequest`. Для train request дополнительно передается `settings.train.max_train_batches_per_epoch`, чтобы расчет prefetch учитывал фактическую длину train-эпохи; для val этот лимит не передается, потому что val loader работает из RAM cache.

DataLoader оборачивается внутренним счетчиком tile batches, category counters, augmented counters, диагностик valid-footprint filter и cache metadata. Если `settings.train.initial_checkpoint_uri` задан, вызывается `models.load_checkpoint` с `LoadCheckpointRequest`; иначе вызывается `models.create_model`.

В `ModelSpec.output_channels` передается `settings.train.output_channels`; для multiclass это `len(settings.dataset.classes)+1`. В `TrainConfig` передаются train-гиперпараметры из settings, включая `task`, диагностические batch limits и `class_slugs`. В `train_model` передается progress sink. На событии `epoch_finished` sink вызывает `mlflow_adapter.log_training_epoch`, чтобы MLflow обновлялся сразу после каждой эпохи. Время live MLflow logging учитывается в timing как `mlflow_logging`.

Если `TrainPipelineRequest.run_name` не задан, в MLflow tags передается `class=Path(settings.dataset.annotation_file).stem` для двоичного режима или `class=multiclass` для многоклассового режима, чтобы `mlflow_adapter` мог сгенерировать имя вида `{class}_{DDMM}_{номер}`. В `MLflowStartRunRequest.dataset` конвейер передает имя датасета без расширения `.geojson`: для двоичного режима это stem `settings.dataset.annotation_file`, для многоклассового режима - stem файлов positive-разметки из `settings.dataset.classes` в порядке конфига, соединенные через `+`. После завершения обучения конвейер пишет итоговые metrics, training artifacts, `reports/tile_preparation.json`, timing report и pipeline report. Dataset artifacts для binary сохраняются под исходными именами файлов; для multiclass имена получают префикс class slug, чтобы файлы разных классов не перетирались. Tile report на верхнем уровне содержит заданные пользователем `positive_factor`, `hard_negative_factor`, `background_factor`, а также `val_positive_factor`, `class_balance`, `num_workers`, `prefetch_epochs`, valid-footprint diagnostics и pool/split window counts. Для каждого split фиксируются `sampling_mode`, effective `positive_factor_used`, `hard_negative_factor_used`, `background_factor_used`, `cache_mode`, `cached_batches`, `cached_tiles`, `estimated_positive_tiles`, `estimated_hard_negative_tiles`, `estimated_background_tiles`, observed counters и ratios для positive/hard_negative/background, `observed_augmented_tiles`, `observed_augmented_positive_tiles`, `observed_augmented_hard_negative_tiles`. В train split effective factors учитывают перенос hard-negative budget в positive при отсутствии или малом числе hard-negative tiles; такие переносы попадают в warnings отчета. Ratio abs error считается только там, где target factors реально применяются, то есть для train weighted sampler; для val штатный `sampling_mode` - `cached_balanced`, train factors не применяются, а `prefetch_epochs` относится только к train prefetch.
