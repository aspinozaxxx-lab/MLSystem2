# Training UI API

`training_ui_api` — отдельный FastAPI-сервис сайта MLSystem2.

## Env vars

- `MLSYSTEM2_TRAINING_UI_API_HOST` — host uvicorn, default `0.0.0.0`.
- `MLSYSTEM2_TRAINING_UI_API_PORT` — port uvicorn, default `8091`.
- `MLSYSTEM2_TRAINING_UI_DATABASE_URL` — Postgres URL, секреты задаются только через env.
- `MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA` — отдельная схема Postgres, default `training_ui`.
- `MLSYSTEM2_MLMARKUP_ROOT` — путь к MLMarkup, default `/data/MLMarkup`.
- `MLSYSTEM2_IMAGES_ROOT` — путь к подготовленным снимкам, default `/data/mlsystem2/prepared_images`.
- `MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT` — корень загруженных txt/geojson.
- `MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT` — временные файлы jobs.
- `MLSYSTEM2_TRAINING_UI_FRONTEND_DIST` — каталог собранного frontend, default `/opt/mlsystem2/frontend`.
- `MLSYSTEM2_MLFLOW_TRACKING_URI` или `MLFLOW_TRACKING_URI` — internal MLflow tracking URI.
- `MLSYSTEM2_TRAINING_UI_USER` или `MLSYSTEM_FRONTEND_USER` — пользователь входа.
- `MLSYSTEM2_TRAINING_UI_PASSWORD` или `MLSYSTEM_FRONTEND_PASSWORD` — пароль входа.
- `MLSYSTEM2_TRAINING_UI_SESSION_SECRET` или `MLSYSTEM_FRONTEND_SESSION_SECRET` — секрет подписи cookie.
- `MLSYSTEM2_GRAFANA_URL`, `MLSYSTEM2_MLFLOW_UI_URL`, `MLSYSTEM2_MINIO_UI_URL` — ссылки на UI сервисов.

## Endpoints

- `GET /api/v1/health`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/logout`
- `GET /api/v1/auth/me`
- `GET /api/v1/app-links`
- `GET /api/v1/mlflow/experiments`
- `POST /api/v1/mlflow/experiments`
- `GET /api/v1/datasets`
- `GET /api/v1/classes`
- `POST /api/v1/custom-datasets`
- `GET /api/v1/models`
- `GET /api/v1/training-templates`
- `GET /api/v1/training-templates/{architecture}`
- `PUT /api/v1/training-templates/{architecture}`
- `POST /api/v1/training-jobs`
- `GET /api/v1/queues`
- `PUT /api/v1/queues/training/enabled`
- `PUT /api/v1/queues/inference/enabled`
- `GET /api/v1/jobs/{job_id}`
- `DELETE /api/v1/jobs/{job_id}`
- `POST /api/v1/jobs/{job_id}/move-up`
- `POST /api/v1/jobs/{job_id}/move-down`
- `GET /api/v1/results/classes`
- `GET /api/v1/results/classes/{class_key}`
- `POST /api/v1/results/classes/{class_key}/pseudo-markup`
- `GET /api/v1/files/{file_id}/download`

OpenAPI доступен стандартно по `/openapi.json`.

Для reverse proxy дополнительно есть совместимый endpoint `GET /auth/proxy-check`: он не входит в OpenAPI,
возвращает `204` и `X-Remote-User` для авторизованной cookie-сессии, иначе `401`.

## Модели данных

Основные таблицы Postgres:

- `training_templates`
- `stored_files`
- `custom_datasets`
- `jobs`
- `queue_controls`
- `training_results`
- `pseudo_markup_results`

Шаблон хранит `config_schema` и `default_config` в JSONB. `default_config` использует ключи вида
`train.learning_rate`, `tile_preparation.tile_size`; это имена параметров YAML config-файла.

## Границы

Frontend не обращается к Postgres. Сервис не импортирует приватные модули MLSystem2 и берет список
моделей только из `models.api`, а MLflow experiments только из `mlflow_adapter.api`.
