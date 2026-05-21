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
- `TrainSettings` — настройки обучения; содержит `batch_size`, который `train_pipeline` передает в `TileDataloaderRequest`, потому что DataLoader создается под обучение. Настройки числа рабочих процессов DataLoader в `TrainSettings` нет.
- `InferenceSettings`, `MLflowSettings` — настройки соответствующих модулей конвейера.
- `SystemSettings` — корневой DTO настроек.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей.

## Алгоритм работы и его особенности

Модуль проверяет, что путь настроек существует и является файлом, читает YAML через `PyYAML`, ожидает корневой словарь и валидирует его через Pydantic `SystemSettings`. `load_settings` сохраняет результат в module-level current object, `get_settings` отдает этот объект остальным модулям. CLI вызывает `load_settings` один раз в начале запуска. Секции `storage` в настройках нет; лишняя секция отклоняется как неизвестная. `stride <= tile_size`, `num_workers >= 0`, `prefetch_factor > 0`, `augmentation_level` от 0 до 3.
