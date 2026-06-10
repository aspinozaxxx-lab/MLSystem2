# Training UI API

`training_ui_api` — отдельный FastAPI-сервис сайта MLSystem2.

## Env vars

- `MLSYSTEM2_TRAINING_UI_API_HOST` — host uvicorn, default `0.0.0.0`.
- `MLSYSTEM2_TRAINING_UI_API_PORT` — port uvicorn, default `8091`.
- `MLSYSTEM2_PROJECT_ROOT` — корень установленного репозитория MLSystem2 для запуска CLI, default равен `cwd` сервиса.
- `MLSYSTEM2_TRAINING_UI_DATABASE_URL` — Postgres URL, секреты задаются только через env.
- `MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA` — отдельная схема Postgres, default `training_ui`.
- `MLSYSTEM2_MLMARKUP_ROOT` — путь к MLMarkup, default `/data/MLMarkup`.
- `MLSYSTEM2_IMAGES_ROOT` — путь к подготовленным снимкам, default `/data/mlsystem2/prepared_images`.
- `MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT` — корень загруженных txt/geojson.
- `MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT` — временные файлы jobs.
- `MLSYSTEM2_TRAINING_UI_FRONTEND_DIST` — каталог собранного frontend, default `/opt/mlsystem2/frontend`.
- `MLSYSTEM2_TRAINING_SETTINGS_PATH` — путь к стабильному `settings.yml`, default `configs/settings.server.yaml`
  относительно `MLSYSTEM2_PROJECT_ROOT`.
- `MLSYSTEM2_MLFLOW_TRACKING_URI` или `MLFLOW_TRACKING_URI` — internal MLflow tracking URI.
- `MLSYSTEM2_TRAINING_UI_USER` или `MLSYSTEM_FRONTEND_USER` — пользователь входа.
- `MLSYSTEM2_TRAINING_UI_PASSWORD` или `MLSYSTEM_FRONTEND_PASSWORD` — пароль входа.
- `MLSYSTEM2_TRAINING_UI_SESSION_SECRET` или `MLSYSTEM_FRONTEND_SESSION_SECRET` — секрет подписи cookie.
- `MLSYSTEM2_GRAFANA_URL`, `MLSYSTEM2_MLFLOW_UI_URL`, `MLSYSTEM2_MINIO_UI_URL` — ссылки на UI сервисов.
- `MLSYSTEM2_TRAINING_UI_WORKER_ENABLED` — включает фоновый исполнитель очереди, default `true`.
- `MLSYSTEM2_TRAINING_UI_WORKER_INTERVAL_SECONDS` — период проверки очереди, default `5`.

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
- `GET /api/v1/automation`
- `PUT /api/v1/automation/enabled`
- `PUT /api/v1/automation/rules`
- `GET /api/v1/training-templates`
- `POST /api/v1/training-templates`
- `PUT /api/v1/training-templates/by-id/{template_id}`
- `DELETE /api/v1/training-templates/by-id/{template_id}`
- `PUT /api/v1/training-templates/by-id/{template_id}/apply-field-to-all`
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
- `GET /api/v1/results/changes`
- `GET /api/v1/results/classes/{class_key}`
- `POST /api/v1/results/classes/{class_key}/pseudo-markup`
- `GET /api/v1/files/{file_id}/download`

OpenAPI доступен стандартно по `/openapi.json`.

`GET /api/v1/datasets` возвращает плоский список вариантов датасетов MLMarkup с ключами и именами вида
`Класс\вариант`, например `Вырубки\main`. `GET /api/v1/classes` и `GET /api/v1/results/classes`
возвращают классы с вложенным списком `variants`; frontend не выбирает класс целиком, а открывает конкретный
вариант. `updated_at` у варианта заполняется по последнему git-коммиту, затронувшему папку варианта в
`MLSYSTEM2_MLMARKUP_ROOT`. Если `MLSYSTEM2_MLMARKUP_ROOT` не является git checkout, используется filesystem
mtime как fallback. `version` у варианта равен `git:{commit_sha}` или `fs:{mtime_ns}` и используется автоматизацией
для дедупликации jobs по конкретной версии датасета.

`GET /api/v1/automation` возвращает глобальный выключатель, непустые MLMarkup-датасеты без `Custom`, UI-модели и
матрицу правил `(dataset_key, architecture)`. `PUT /api/v1/automation/enabled` включает автоматику или полностью
отключает ее: активные automatic jobs отменяются, queued automatic jobs очищаются из очередей, running training
process получает SIGTERM, а известный MLflow run помечается как `KILLED`. При повторном включении jobs создаются
заново по текущим правилам и `dataset_version`. `PUT /api/v1/automation/rules` сохраняет две галочки правила: `training_enabled` и
`pseudo_markup_enabled`.

Для reverse proxy дополнительно есть совместимый endpoint `GET /auth/proxy-check`: он не входит в OpenAPI,
возвращает `204` и `X-Remote-User` для авторизованной cookie-сессии, иначе `401`.

## Модели данных

Основные таблицы Postgres:

- `training_templates`
- `stored_files`
- `custom_datasets`
- `jobs`
- `queue_controls`
- `automation_controls`
- `automation_rules`
- `training_results`
- `pseudo_markup_results`

Шаблон хранит `config_schema` и `default_config` в JSONB. `default_config` использует ключи вида
`train.learning_rate`, `tile_preparation.tile_size`; это только параметры, которые оператор меняет на сайте.
В этот набор входит `train.max_training_time_sec`: пустое значение означает обучение без wall-clock лимита.
Инфраструктурные defaults DataLoader, `train.device=cuda`, binary task и каналы модели задаются модулем
`settings` и не сохраняются в UI-шаблонах.

`training_templates.dataset_key` nullable. Строка с `dataset_key=null` является базовым шаблоном сети.
Строка с заполненным `dataset_key` является переопределением для конкретного варианта MLMarkup, например
`Вырубки\test`, и создается через `POST /api/v1/training-templates` копированием текущих defaults базового
шаблона сети. При ручном запуске и автоматизации сервис сначала ищет активный шаблон `(architecture,
dataset_key)`, а если его нет, использует базовый шаблон `(architecture, null)`. Базовый шаблон удалить нельзя;
датасетный шаблон удаляется через `DELETE /api/v1/training-templates/by-id/{template_id}`. Endpoint
`PUT /api/v1/training-templates/by-id/{template_id}/apply-field-to-all` устанавливает одно поле во всех
существующих шаблонах и помечает их `source=manual`.

`jobs`, `training_results` и `pseudo_markup_results` имеют `source=manual|automation`, `dataset_key` и
`dataset_version`. Для auto rows дополнительно заполнен `automation_rule_id`. Auto jobs нельзя удалить или двигать
через endpoints очереди; они отменяются снятием соответствующей галочки в автоматизации или заменяются при новой
версии конкретного датасета. Глобальное отключение автоматизации отменяет все active auto jobs независимо от правила.

## Границы

Frontend не обращается к Postgres. Сервис не импортирует приватные модули MLSystem2 и берет список
моделей только из `models.api`, а данные MLflow только из `mlflow_adapter.api`.
`training_ui_api` не открывает training runs и не пишет MLflow-метрики: запись метрик, отчетов и артефактов
запуска выполняет только `train_pipeline`.

Фоновый worker работает внутри FastAPI-сервиса. Он берет первый `queued` job из единой очереди и запускает
обучение или псевдоразметку отдельным процессом. Для training job worker формирует `run.yml` из сохраненных
параметров задания и запускает публичный CLI
`python -m mlsystem2.cli.train --settings configs/settings.server.yaml --run ...`.
Стабильные параметры приложения, такие как workers/prefetch/seed/device, берутся из `settings.yml`
и не записываются в `run.yml`. Секция `inference` в training `run.yml` не создается: checkpoint, threshold,
batch size и output GeoJSON задаются в отдельном `pseudo_config.yaml` при запуске псевдоразметки. Training-процесс сразу после создания MLflow run пишет
его id в временный файл `mlflow_run_id`; worker читает этот файл и обновляет `training_results.mlflow_run_id`
еще во время `running`. Pause/delete отправляют SIGTERM группе процесса, а `train_pipeline` штатно завершает
MLflow run со статусом `KILLED`.

Перед обработкой очередей worker синхронизирует автоматизацию. Если глобальный выключатель включен и для правила
нет результата или job по текущей `dataset_version`, он ставит auto training job в experiment `MLSystem2 Automation`.
После успешного auto training result с MLflow run id worker ставит auto pseudo-markup job по txt того же датасета.
Jobs запускаются из единой очереди с внутренним приоритетом: ручная псевдоразметка, ручное обучение, auto
псевдоразметка, auto обучение. При выключенной автоматизации новые auto jobs не создаются и queued auto jobs не
стартуют, потому что `PUT /api/v1/automation/enabled` с `enabled=false` сразу отменяет и очищает все active auto
jobs. Failed auto attempt не ретраится до новой версии датасета или снятия и повторной установки галочки.

`GET /api/v1/results/changes` возвращает последние 20 успешных изменений из `training_results` и
`pseudo_markup_results`, отсортированные по времени изменения. Каждая строка содержит `class_key`, имя модели,
имя датасета и действие: `обучена сеть` или `создана разметка`.

При успешном завершении training job worker читает через публичный `mlflow_adapter.api` историю метрики
`val/best_threshold_pixel_f1`, по которой train-модуль сохраняет `checkpoints/best.pt`, и записывает лучший
F1 и эпоху в `training_results.f1_score`/`training_results.epoch`. Pseudo-markup job, созданный от результата
обучения, получает в `jobs.config` `mlflow_run_id`, `checkpoint_artifact_path=checkpoints/best.pt` и, когда
MLflow доступен, полный `checkpoint_uri`.

Для job типа `pseudo-markup` worker берет txt список снимков из выбранного датасета или загруженного файла, скачивает `checkpoints/best.pt`
через публичный `mlflow_adapter.api.download_run_artifact`, загружает checkpoint через `models.api.load_checkpoint`,
строит GeoJSON псевдоразметки в `EPSG:4326` и сохраняет его в `stored_files` для скачивания через
`/api/v1/files/{file_id}/download`.

Перед инференсом раннер ищет реальные TIFF по строкам txt, удаляет повторные совпадения и выбирает профиль
постобработки по числу уникальных снимков. Для `<=5` снимков используется прежний режим без постобработки, для
`6..50` — `detail_v2` с легкой чисткой маски и `Simplify=10 м`, для `>=51` — `strong` с `binary_closing`,
порогами площади `10000 м²` и `Simplify=30 м`. Отчет псевдоразметки содержит выбранный профиль, число уникальных
снимков и параметры постобработки; HTTP API и схема БД при этом не меняются.
