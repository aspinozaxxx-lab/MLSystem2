# Модуль models

## Назначение

`models` создает model handles, загружает чекпойнты, сохраняет чекпойнты и описывает
поддерживаемые архитектуры.

## Публичный интерфейс

- `list_supported_models() -> list[ModelSpec]`
- `create_model(spec: ModelSpec) -> ModelHandle`
- `load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint`
- `save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact`

Параметры:

- `spec`: запрошенная архитектура модели и конфигурация каналов.
- `request`: DTO запроса на загрузку или сохранение чекпойнта.

## Контракты

`ModelSpec`, `ModelHandle`, `LoadCheckpointRequest`, `SaveCheckpointRequest`,
`LoadedCheckpoint`, `CheckpointArtifact`, `ModelsError`.

## Выходные артефакты

Ссылки на артефакты чекпойнтов после операций сохранения.

## Что модуль НЕ делает

Не читает датасет, не пишет в MLflow, не запускает обучение и не оркестрирует конвейер инференса.

## Разрешенные зависимости

Необязательный `torch` внутри приватной реализации модели и чекпойнтов, публичный API хранилища для
доступа к checkpoint URI.

## Запрещенные пересечения

Не импортирует внутренности датасета, обучения, конвейера инференса и адаптера MLflow.

## MLflow

Не пишет в MLflow. Артефакты чекпойнтов логирует `mlflow_adapter`.

## Временное профилирование

Время не замеряет.
