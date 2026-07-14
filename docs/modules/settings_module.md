# Модуль settings

## Назначение

`settings` загружает YAML-конфиг и возвращает типизированные настройки для остальных модулей. Модуль является единственным местом валидации конфигурации процесса.

## Публичный интерфейс

- `load_settings(path: str | Path, run_path: str | Path | None = None) -> SystemSettings` - читает YAML, валидирует `SystemSettings`, сохраняет его как текущие настройки процесса и возвращает. Если передан `run_path`, сначала читается стабильный `settings.yml`, затем поверх него накладывается `run.yml` задания запуска.
- `get_settings() -> SystemSettings` - возвращает текущие настройки процесса. Если `load_settings` еще не вызывался, бросает `SettingsError`.
- `get_settings_path() -> Path` - возвращает путь к текущему YAML-конфигу. Если `load_settings` еще не вызывался, бросает `SettingsError`.

## Публичные контракты

- `SettingsError` - ошибка загрузки или валидации.
- `RuntimeSettings` - поля `project_root`, `scratch_root`, `logs_root`, `cleanup_scratch_after_mlflow_log`.
- `DatasetClassSettings` - поля `slug`, `name`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `DatasetSettings` - поля `images_dir`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `classes`, `val_fraction`; свойство `is_multiclass`.
- `TilePreparationSettings` - поля `tile_size`, `stride`, `num_workers`, `prefetch_epochs`, `seed`, `augmentation_level`, `positive_factor`, `hard_negative_factor`, `background_factor`, `val_positive_factor`, `class_balance`.
- `TrainSettings` - поля `task`, `quality_metric`, `model_name`, `input_channels`, `output_channels`, `pretrained`, `initial_checkpoint_uri`, `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `hard_negative_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`, `max_training_time_sec`.
- `InferenceSettings`, `MLflowSettings` - настройки соответствующих модулей конвейера.
- `SystemSettings` - корневой DTO настроек.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. YAML читается через `PyYAML`, валидация выполняется через Pydantic.

## Алгоритм работы и его особенности

`load_settings` проверяет, что путь настроек существует и является файлом, читает YAML, ожидает корневой словарь и валидирует его через `SystemSettings`. В режиме `settings.yml + run.yml` словари объединяются рекурсивно: `run.yml` переопределяет только поля конкретного запуска, а стабильные параметры приложения остаются в `settings.yml`. Результат и абсолютный путь YAML сохраняются в module-level current object; если передан `run_path`, `get_settings_path` возвращает именно путь к `run.yml`. Лишние секции и поля отклоняются после объединения.

`settings.yml` хранит параметры приложения, которые не должны меняться между запусками обычным оператором: `runtime.project_root`, базовые директории, `dataset.images_dir`, `tile_preparation.num_workers`, `prefetch_epochs`, `seed`, `val_positive_factor`, `class_balance`, `train.task`, `input_channels`, `output_channels`, `pretrained`, `device`, а также `mlflow.enabled` и `mlflow.tracking_uri`.

`run.yml` хранит задание конкретного обучения: пути positive-разметки, optional hard-negative разметки, `dataset.val_fraction`, `tile_size`, `stride`, аугментации, train sampling factors, модель, `quality_metric`, гиперпараметры обучения, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`, `max_training_time_sec` и имя MLflow experiment. CLI использует `quality_metric=pixel` по умолчанию; объектовая метрика допустима только для binary. Параметры inference задаются отдельно при создании задания псевдоразметки.

Основные train-поля использовались в tuning runs или необходимы реальному SegFormer train loop. Background имеет фиксированный вес `1`, `pos_weight` усиливает positive pixels в binary loss, а `hard_negative_weight` усиливает штраф за ложноположительные pixels внутри hard-negative тайлов в binary и multiclass train. Optimizer фиксирован как AdamW, scheduler фиксирован как cosine и не выносится в settings, пока нет необходимости менять их как гиперпараметры.

Category-aware tile sampling является штатным train-режимом. `positive_factor`, `hard_negative_factor` и `background_factor` задают публичную тройку долей полного train sampler budget. Каждый фактор находится в диапазоне `0..1`, сумма после defaults должна быть равна `1.0`, нулевое значение отключает категорию. Для legacy config без `background_factor` он вычисляется как `1 - positive_factor - hard_negative_factor`; server default равен `0.5 / 0.0 / 0.5`, UI default равен `0.8 / 0.0 / 0.2`. В train sampler `positive_factor + hard_negative_factor` трактуются как общий marked-бюджет, а `background_factor` остается отдельной долей обычного background. Effective hard-negative budget ограничен заданным `hard_negative_factor` и долей hard-negative tiles внутри marked-пула; если hard-negative разметки нет, hard-negative tiles не найдены или их мало относительно positive+hard_negative tiles, недостающий hard-negative budget переносится в positive. Поэтому `hard_negative_factor > 0` допустим без `hard_negative_annotation_file`, если есть positive tiles для marked-бюджета. Val выборка остается фиксированной и кэшированной в RAM: train factors не применяются к val loader, а `val_positive_factor` остается совместимым параметром и серверным default `0.5`. В multiclass режиме `class_balance=true` распределяет только effective positive-бюджет между классами с найденными positive windows; hard_negative и background бюджеты остаются отдельными. Аугментация применяется к positive и hard_negative train tiles, background train tiles не аугментируются. Hard negative не меняет masks и labels: в target mask это всегда background `0`.

`prefetch_epochs` задает целевой запас PyTorch DataLoader prefetch в эпохах только для train loader. `tile_preparation` вычисляет effective train `prefetch_factor` как `ceil(effective_batches_per_epoch * prefetch_epochs / num_workers)`, где `effective_batches_per_epoch = min(ceil(dataset_size / batch_size), max_train_batches_per_epoch)`, если batch limit задан, иначе используется полный train split. Val loader не использует PyTorch worker prefetch: его batch-и один раз собираются в CPU RAM cache до старта обучения и переиспользуются на каждой эпохе.

`max_train_batches_per_epoch` и `max_val_batches_per_epoch` добавлены только для диагностических коротких запусков. В полном обучении они могут оставаться `null`. `max_training_time_sec` - optional wall-clock лимит train loop; он проверяется после завершения эпохи и завершает обучение штатно, чтобы сохранить final checkpoint.

Проверяется: `stride <= tile_size`, `augmentation_level` в диапазоне `0..3`, positive train-размеры, `learning_rate > 0`, `weight_decay >= 0`, threshold/focal диапазоны, tversky/pos_weight/hard_negative_weight > 0, batch limits либо `null`, либо больше `0`.

`dataset` поддерживает два взаимоисключающих режима разметки: binary через `scenes_file` + `annotation_file` и multiclass через `classes`. Разбиение train/val всегда выполняется по тайлам: `dataset_preparing` строит общий пул найденных снимков, а `tile_preparation` делит уже тайлы.

Binary mode:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/
  scenes_file: /data/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/MLMarkup/Вырубки/deforestation.geojson
  hard_negative_annotation_file: /data/MLMarkup/Вырубки/hard_negative.geojson
  val_fraction: 0.2
```

Multiclass mode:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/
  val_fraction: 0.2
  classes:
    - slug: abrasion
      name: Абразия
      scenes_file: /data/MLMarkup/Абразия/abrasion.txt
      annotation_file: /data/MLMarkup/Абразия/abrasion.geojson
      hard_negative_annotation_file: /data/MLMarkup/Абразия/hard_negative.geojson
      priority: 0
```

Валидация `DatasetSettings`: либо заданы `classes`, либо заданы `scenes_file` + `annotation_file`; смешивать режимы нельзя. `hard_negative_annotation_file` в верхнем уровне относится только к binary режиму, а в multiclass задается внутри конкретного `DatasetClassSettings`. `classes` не должен быть пустым в multiclass режиме. `slug` и `name` должны быть уникальны. Class id назначается порядком в config: `background=0`, первый класс `1`. `priority` используется только при пересечении multiclass positive-разметки: больший приоритет перекрывает меньший, при равном приоритете используется порядок `class_id`.

Валидация `SystemSettings`: `dataset.classes` требует `train.task=multiclass`, `train.quality_metric=pixel`, `train.loss=cross_entropy` или `train.loss=cross_entropy_dice` и `train.output_channels=len(dataset.classes)+1`. Binary dataset требует `train.task=binary`; multiclass loss в binary режиме запрещен.
