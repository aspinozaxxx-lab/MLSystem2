# Модуль settings

## Назначение

`settings` загружает YAML-конфиг и возвращает типизированные настройки для остальных модулей. Модуль является единственным местом валидации конфигурации процесса.

## Публичный интерфейс

- `load_settings(path: str | Path) -> SystemSettings` - читает YAML, валидирует `SystemSettings`, сохраняет его как текущие настройки процесса и возвращает.
- `get_settings() -> SystemSettings` - возвращает текущие настройки процесса. Если `load_settings` еще не вызывался, бросает `SettingsError`.
- `get_settings_path() -> Path` - возвращает путь к текущему YAML-конфигу. Если `load_settings` еще не вызывался, бросает `SettingsError`.

## Публичные контракты

- `SettingsError` - ошибка загрузки или валидации.
- `RuntimeSettings` - поля `project_root`, `scratch_root`, `logs_root`, `cleanup_scratch_after_mlflow_log`.
- `DatasetSettings` - поля `images_dir`, `scenes_file`, `annotation_file`, `val_fraction`.
- `TilePreparationSettings` - поля `tile_size`, `stride`, `num_workers`, `prefetch_factor`, `seed`, `augmentation_level`, `smart_tiling`.
- `TrainSettings` - поля `model_name`, `input_channels`, `output_channels`, `pretrained`, `initial_checkpoint_uri`, `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`.
- `InferenceSettings`, `MLflowSettings` - настройки соответствующих модулей конвейера.
- `SystemSettings` - корневой DTO настроек.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. YAML читается через `PyYAML`, валидация выполняется через Pydantic.

## Алгоритм работы и его особенности

`load_settings` проверяет, что путь настроек существует и является файлом, читает YAML, ожидает корневой словарь и валидирует его через `SystemSettings`. Результат и абсолютный путь YAML сохраняются в module-level current object; `get_settings` и `get_settings_path` отдают их остальным модулям. Лишние секции и поля отклоняются.

Основные train-поля использовались в tuning runs или необходимы реальному SegFormer train loop. Optimizer фиксирован как AdamW, scheduler фиксирован как cosine и не выносится в settings, пока нет необходимости менять их как гиперпараметры.

`smart_tiling=false` оставляет обычную регулярную сетку и стандартный DataLoader. `smart_tiling=true` включает positive-aware train sampling и запрещает аугментацию negative/background tiles; val loader остается deterministic и без augmentation. Дополнительные ratio/max-negative настройки не добавляются.

`max_train_batches_per_epoch` и `max_val_batches_per_epoch` добавлены только для диагностических коротких запусков. В полном обучении они остаются `null`. Лимит wall-clock задается внешним `timeout`, поэтому отдельное поле времени запуска не добавляется.

Проверяется: `stride <= tile_size`, positive train-размеры, `learning_rate > 0`, `weight_decay >= 0`, threshold/focal диапазоны, tversky/pos_weight > 0, batch limits либо `null`, либо больше `0`.
