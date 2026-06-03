# Модуль training_ui_api

## Назначение

`training_ui_api` — отдельный FastAPI-сервис сайта MLSystem2. Он отдает публичный HTTP API для frontend, хранит UI-данные обучения в Postgres БД/схеме, сканирует `/data/MLMarkup` для актуальных датасетов и классов, управляет очередями training/inference и не выполняет прямой доступ frontend к БД.

## Публичный интерфейс

- `create_app() -> Any` - создает FastAPI-приложение.
- `get_openapi_schema() -> dict[str, Any]` - возвращает OpenAPI-схему сервиса.
- `main() -> None` - запускает `uvicorn` для сервиса.

## Публичные контракты

- `TrainingUIAPIError` - ошибка сервиса.
- `TemplateSource`, `JobType`, `JobStatus`, `ResultStatus`, `StoredFileKind` - enum-значения API.
- `AppLink`, `AppLinksResponse` - ссылки Grafana/MLflow/MinIO.
- `MLflowExperimentInfo`, `MLflowExperimentCreate` - experiments MLflow.
- `DatasetInfo`, `DatasetListResponse`, `ClassInfo`, `ClassListResponse` - датасеты и классы MLMarkup.
- `ModelInfo`, `ModelListResponse` - публичные модели из `models`.
- `ConfigField`, `ConfigSchema`, `TrainingTemplate`, `TrainingTemplateListResponse`, `TrainingTemplateUpdate` - шаблоны обучения.
- `StoredFileInfo`, `CustomDatasetInfo` - загруженные файлы и custom datasets.
- `TrainingJobCreate`, `QueueEnabledUpdate`, `QueueControlInfo`, `JobSummary`, `QueueSnapshot`, `JobDetail` - задания и очереди.
- `TrainingResultInfo`, `PseudoMarkupResultInfo`, `ClassResultsResponse` - результаты обучения и псевдоразметки.

## Список используемых данным модулем модулей и с какой целью

- `models.api` - получить публичный список поддерживаемых архитектур.
- `mlflow_adapter.api` - получить и создать MLflow experiments.
- `mlflow_adapter.contracts` - передать публичный DTO создания experiment.

## Алгоритм работы и его особенности

Сервис читает настройки только из env vars. Frontend авторизуется через cookie-session с пользователем и паролем из тех же env vars, что старый сайт: `MLSYSTEM_FRONTEND_USER`/`MLSYSTEM_FRONTEND_PASSWORD`, либо новые `MLSYSTEM2_TRAINING_UI_USER`/`MLSYSTEM2_TRAINING_UI_PASSWORD`. Postgres доступен только из FastAPI; frontend работает через `/api/v1/*` и может отдаваться тем же сервисом из `MLSYSTEM2_TRAINING_UI_FRONTEND_DIST`. MLMarkup не кэшируется: `/datasets`, `/classes` и `/results/classes` сканируют каталог при каждом запросе, поэтому новая папка появляется без релиза. Шаблоны хранятся в `training_templates`; ручное изменение ставит `source=manual` и увеличивает `version`, reset возвращает baseline. Очереди показывают только `queued` и `running`; completed/cancelled/failed не попадают в таблицы. При pause running job получает SIGTERM, tmp-папка удаляется, job возвращается в `queued`, а новые jobs не стартуют до resume.
