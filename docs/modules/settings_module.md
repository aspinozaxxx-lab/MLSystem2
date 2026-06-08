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
- `DatasetClassSettings` - поля `slug`, `name`, `scenes_file`, `annotation_file`, `priority`.
- `DatasetSettings` - поля `images_dir`, `scenes_file`, `annotation_file`, `classes`, `val_fraction`; свойство `is_multiclass`.
- `TilePreparationSettings` - поля `tile_size`, `stride`, `num_workers`, `prefetch_epochs`, `seed`, `augmentation_level`, `positive_factor`, `val_positive_factor`, `class_balance`.
- `TrainSettings` - поля `task`, `model_name`, `input_channels`, `output_channels`, `pretrained`, `initial_checkpoint_uri`, `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`, `max_training_time_sec`.
- `InferenceSettings`, `MLflowSettings` - настройки соответствующих модулей конвейера.
- `SystemSettings` - корневой DTO настроек.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. YAML читается через `PyYAML`, валидация выполняется через Pydantic.

## Алгоритм работы и его особенности

`load_settings` проверяет, что путь настроек существует и является файлом, читает YAML, ожидает корневой словарь и валидирует его через `SystemSettings`. В режиме `settings.yml + run.yml` словари объединяются рекурсивно: `run.yml` переопределяет только поля конкретного запуска, а стабильные параметры приложения остаются в `settings.yml`. Результат и абсолютный путь YAML сохраняются в module-level current object; если передан `run_path`, `get_settings_path` возвращает именно путь к `run.yml`. Лишние секции и поля отклоняются после объединения.

`settings.yml` хранит параметры приложения, которые не должны меняться между запусками обычным оператором: `runtime.project_root`, базовые директории, `dataset.images_dir`, `tile_preparation.num_workers`, `prefetch_epochs`, `seed`, `val_positive_factor`, `class_balance`, `train.task`, `input_channels`, `output_channels`, `pretrained`, `device`, а также `mlflow.enabled` и `mlflow.tracking_uri`.

`run.yml` хранит задание конкретного обучения: пути разметки, `dataset.val_fraction`, `tile_size`, `stride`, аугментации, positive sampling, модель, гиперпараметры обучения, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`, `max_training_time_sec` и имя MLflow experiment. Параметры inference задаются отдельно при создании задания псевдоразметки.

Основные train-поля использовались в tuning runs или необходимы реальному SegFormer train loop. Optimizer фиксирован как AdamW, scheduler фиксирован как cosine и не выносится в settings, пока нет необходимости менять их как гиперпараметры.

Positive-aware tile sampling является единственным штатным режимом. `positive_factor` используется для `mode=train`: значение `0.8` означает примерно 80% positive и 20% negative samples в training epoch. `val_positive_factor` используется для `mode=val`; серверный default `0.5` дает примерно равные positive и negative validation samples. В multiclass режиме `positive_factor` остается балансом foreground/background. Если дополнительно задано `class_balance=true`, positive-доля распределяется между классами с найденными positive windows примерно равномерно; классы без positive windows попадают в warnings. Аугментация применяется только к positive train tiles. Эти поля не меняют masks и labels.

`prefetch_epochs` задает целевой запас PyTorch DataLoader prefetch в эпохах. `tile_preparation` вычисляет effective `prefetch_factor` как `ceil(ceil(dataset_size / batch_size) * prefetch_epochs / num_workers)`. Это заставляет DataLoader стремиться держать в worker queues запас уже прочитанных и rasterized batch-ей на указанное число эпох, но не сохраняет tiles на диск и не меняет ленивый `Dataset.__getitem__`.

`max_train_batches_per_epoch` и `max_val_batches_per_epoch` добавлены только для диагностических коротких запусков. В полном обучении они могут оставаться `null`. `max_training_time_sec` - optional wall-clock лимит train loop; он проверяется после завершения эпохи и завершает обучение штатно, чтобы сохранить final checkpoint.

Проверяется: `stride <= tile_size`, `augmentation_level` в диапазоне `0..3`, positive train-размеры, `learning_rate > 0`, `weight_decay >= 0`, threshold/focal диапазоны, tversky/pos_weight > 0, batch limits либо `null`, либо больше `0`.

`dataset` поддерживает два взаимоисключающих режима разметки: binary через `scenes_file` + `annotation_file` и multiclass через `classes`. Разбиение train/val всегда выполняется по тайлам: `dataset_preparing` строит общий пул найденных снимков, а `tile_preparation` делит уже тайлы.

Binary mode:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/
  scenes_file: /data/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/MLMarkup/Вырубки/deforestation.geojson
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
      priority: 0
```

Валидация `DatasetSettings`: либо заданы `classes`, либо заданы `scenes_file` + `annotation_file`; смешивать режимы нельзя. `classes` не должен быть пустым в multiclass режиме. `slug` и `name` должны быть уникальны. Class id назначается порядком в config: `background=0`, первый класс `1`. `priority` используется только при пересечении multiclass разметки: больший приоритет перекрывает меньший, при равном приоритете используется порядок `class_id`.

Валидация `SystemSettings`: `dataset.classes` требует `train.task=multiclass`, `train.loss=cross_entropy` или `train.loss=cross_entropy_dice` и `train.output_channels=len(dataset.classes)+1`. Binary dataset требует `train.task=binary`; multiclass loss в binary режиме запрещен.
