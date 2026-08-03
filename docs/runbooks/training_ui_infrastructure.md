# Runbook ручного развертывания training UI

Инфраструктура поднимается вручную на сервере. В CI/CD не добавляется создание контейнеров,
Postgres, сетей или reverse proxy.

## 1. Postgres

На сервере создать env-файл вне git, например `/etc/mlsystem2/training-ui-postgres.env`:

```bash
POSTGRES_DB=mlsystem2_training_ui
POSTGRES_USER=mlsystem2_training_ui
POSTGRES_PASSWORD=<задать вручную>
```

Поднять контейнер вручную:

```bash
docker run -d \
  --name mlsystem2-training-ui-postgres \
  --restart unless-stopped \
  --env-file /etc/mlsystem2/training-ui-postgres.env \
  -v /data/mlsystem2/training-ui/postgres:/var/lib/postgresql/data \
  -p 127.0.0.1:55432:5432 \
  postgres:16
```

Создать схему, если миграция еще не запускалась:

```bash
psql "$MLSYSTEM2_TRAINING_UI_DATABASE_URL" -c 'CREATE SCHEMA IF NOT EXISTS training_ui;'
```

## 2. Env FastAPI

Создать `/etc/mlsystem2/training-ui-api.env`:

```bash
MLSYSTEM2_TRAINING_UI_DATABASE_URL=postgresql+psycopg://mlsystem2_training_ui:<пароль>@127.0.0.1:55432/mlsystem2_training_ui
MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA=training_ui
MLSYSTEM2_PROJECT_ROOT=/opt/mlsystem2/repo
MLSYSTEM2_MLMARKUP_ROOT=/data/MLMarkup
MLSYSTEM2_IMAGES_ROOT=/data/mlsystem2/prepared_images
MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT=/data/mlsystem2/training-ui/files
MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT=/data/mlsystem2/training-ui/tmp
MLSYSTEM2_TRAINING_UI_FRONTEND_DIST=/opt/mlsystem2/frontend
MLSYSTEM2_TRAINING_SETTINGS_PATH=/opt/mlsystem2/repo/configs/settings.server.yaml
MLSYSTEM2_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLSYSTEM2_MLFLOW_UI_URL=/mlflow/
MLSYSTEM2_GRAFANA_URL=/grafana/
MLSYSTEM2_MINIO_UI_URL=/minio/browser/mlsystems/images/
MLSYSTEM2_TRAINING_UI_USER=mlsystem
MLSYSTEM2_TRAINING_UI_PASSWORD=<тот же пароль, что в старом MLSystem>
MLSYSTEM2_TRAINING_UI_SESSION_SECRET=<случайная строка>
MLSYSTEM2_TRAINING_UI_WORKER_ENABLED=true
MLSYSTEM2_TRAINING_UI_WORKER_INTERVAL_SECONDS=5
MLSYSTEM2_PSEUDOLABEL_API_TOKEN=
MLSYSTEM2_PSEUDOLABEL_MAX_AOI_AREA_M2=0
MLSYSTEM2_PSEUDOLABEL_MAX_VERTICES=10000
MLSYSTEM2_PSEUDOLABEL_JOB_TIMEOUT_SECONDS=3600
```

`MLSYSTEM2_PSEUDOLABEL_API_TOKEN` нужен только отдельным неинтерактивным клиентам. QGIS-плагин
входит по `MLSYSTEM2_TRAINING_UI_USER` и `MLSYSTEM2_TRAINING_UI_PASSWORD`, поэтому для него токен
оставляют пустым. Нулевой `MLSYSTEM2_PSEUDOLABEL_MAX_AOI_AREA_M2` разрешает AOI любой площади;
положительное значение включает операторский лимит в квадратных метрах.

Секреты не коммитить.

## 3. Миграции

После доставки кода:

```bash
cd /opt/mlsystem2/repo
python -m alembic upgrade head
```

Миграция создает таблицы:

- `training_templates`
- `stored_files`
- `custom_datasets`
- `jobs`
- `queue_controls`
- `training_results`
- `pseudo_markup_results`

## 4. FastAPI service

Пример systemd unit `/etc/systemd/system/mlsystem2-training-ui-api.service`:

```ini
[Unit]
Description=MLSystem2 Training UI API
After=network.target

[Service]
WorkingDirectory=/opt/mlsystem2/repo
EnvironmentFile=/etc/mlsystem2/training-ui-api.env
ExecStart=/opt/mlsystem2/venv/bin/mlsystem2-training-ui-api
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Запуск:

```bash
systemctl daemon-reload
systemctl enable --now mlsystem2-training-ui-api
curl --fail http://127.0.0.1:8091/api/v1/health
```

## 5. Frontend static

CI/CD копирует `frontend/dist` в путь из `MLSYSTEM2_FRONTEND_DIST_PATH`, по умолчанию
`/opt/mlsystem2/frontend`.

Reverse proxy должен:

- проксировать `/` и `/api/v1/` на `http://127.0.0.1:8091`;
- проксировать `/mlflow/`, `/grafana/`, `/minio/` как в старом gateway.

## 6. Отключение старого сайта

Старый репозиторий `aspinozaxxx-lab/MLSystem` не менять. На сервере отключить только старый runtime
frontend-сервис или контейнер, например:

```bash
systemctl disable --now <старый-frontend-service>
```

Если старый сайт был контейнером:

```bash
docker stop <старый-frontend-container>
docker update --restart=no <старый-frontend-container>
```

После этого reverse proxy должен вести на новый `frontend/dist` MLSystem2.
