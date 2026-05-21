# Модуль settings

## Назначение

`settings` загружает YAML-конфиг и возвращает типизированные настройки для модулей.

## Публичный интерфейс

- `load_settings(path: str | Path) -> SystemSettings` — читает YAML-файл и валидирует его по публичным контрактам настроек.

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

Модуль проверяет, что путь настроек существует и является файлом, читает YAML через `PyYAML`, ожидает корневой словарь и валидирует его через Pydantic `SystemSettings`. Настройки подготовки датасета описывают только локальные входы текущего шага подготовки. Секции `storage` в настройках нет; лишняя секция отклоняется как неизвестная.
