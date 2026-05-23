# Модуль settings

## Назначение

`settings` загружает YAML-конфиг и возвращает типизированные настройки для модулей.

## Публичный интерфейс

- `load_settings(path: str | Path) -> SystemSettings` — читает YAML-файл, валидирует `SystemSettings`, сохраняет его как текущие настройки процесса и возвращает. Повторный вызов заменяет текущие настройки.
- `get_settings() -> SystemSettings` — возвращает текущие настройки процесса. Если `load_settings` еще не вызывался, бросает `SettingsError`.

## Публичные контракты

- `SettingsError` — ошибка загрузки или валидации.
- `DatasetSettings` — поля `images_dir`, `scenes_file`, `annotation_file`, `val_fraction`.
- `RuntimeSettings` — поля `project_root`, `scratch_root`, `logs_root`, `cleanup_scratch_after_mlflow_log`.
- `TilePreparationSettings` — поля `tile_size`, `stride`, `num_workers = 16`, `prefetch_factor = 2`, `seed = 42`, `augmentation_level = 0`.
- `TrainSettings` — настройки обучения SegFormer B2: `model_name`, `input_channels`, `output_channels`, `pretrained = false`, `initial_checkpoint_uri = null`, `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss: Literal["bce_dice", "focal_dice", "focal_tversky"]`, `focal_alpha = 0.6`, `pos_weight = 1.0`, `tversky_alpha = 0.4`, `tversky_beta = 0.6`, `threshold = 0.5`, `early_stopping_patience`. Эти поля использовались в tuning runs или необходимы для train loop. Optimizer фиксирован как AdamW, scheduler фиксирован как cosine и не выносятся в settings, пока нет необходимости менять их как гиперпараметры.
- `InferenceSettings`, `MLflowSettings` — настройки соответствующих модулей конвейера.
- `SystemSettings` — корневой DTO настроек.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей.

## Алгоритм работы и его особенности

Модуль проверяет, что путь настроек существует и является файлом, читает YAML через `PyYAML`, ожидает корневой словарь и валидирует его через Pydantic `SystemSettings`. `load_settings` сохраняет результат в module-level current object, `get_settings` отдает этот объект остальным модулям. CLI вызывает `load_settings` один раз в начале запуска. Лишние секции отклоняются. Проверяются `stride <= tile_size`, positive train-размеры, `learning_rate > 0`, `weight_decay >= 0`, threshold/focal диапазоны и tversky/pos_weight > 0.
