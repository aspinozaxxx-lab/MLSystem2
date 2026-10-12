# Runbook ручного развертывания training UI

Инфраструктура поднимается вручную на сервере. В CI/CD не добавляется создание контейнеров,
Postgres, сетей или reverse proxy.

## Источник опубликованных разметок

Каталог, обучение и автоматизация используют `/data/mlsystem2/mlmarkup-editor`.
`MLSYSTEM2_MLMARKUP_ROOT` и `MLSYSTEM2_MLMARKUP_EDITOR_ROOT` должны указывать на этот клон.
Клон принадлежит `MLMarkup_deploy_user`; отдельный write deploy key и commit/push в GitHub сохраняются.
Редактор и читатели используют один `.mlmarkup-editor.lock` рядом с клоном. Не изменяйте Git-дерево
в обход этого lock во время работы сервиса. Серверные черновики остаются в Postgres.

Перед переходом проверьте чистоту рабочего дерева, совпадение HEAD с удалённой веткой и версии датасетов
в рабочем и архивном корнях. Незавершённые изменения не сбрасывайте. Измените только пути источника
в серверном env-файле, затем перезапустите API и worker штатными сервисами; уже запущенные дочерние
обучения используют свои снимки данных и должны сохраниться. Проверьте каталог, публикацию и очереди.
Токены и пароли не выводите; env-файл загружайте только на сервере.

Обратная доставка MLMarkup отключена в GitHub Actions, workflow остаётся в репозитории:

```bash
gh workflow disable deploy-markup-to-gpu.yml --repo aspinozaxxx-lab/MLMarkup
gh workflow list --repo aspinozaxxx-lab/MLMarkup --all
```

Прежний `/data/MLMarkup` и каталоги `/data/mlmarkup-runtime/releases/` не удаляются.
Для будущего восстановления доставки используйте `gh workflow enable deploy-markup-to-gpu.yml
--repo aspinozaxxx-lab/MLMarkup` и отдельно согласуйте переключение `MLSYSTEM2_MLMARKUP_ROOT`
на архивный корень. Само включение workflow не меняет рабочий источник Гровики.

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
MLSYSTEM2_MLMARKUP_ROOT=/data/mlsystem2/mlmarkup-editor
MLSYSTEM2_MLMARKUP_EDITOR_ROOT=/data/mlsystem2/mlmarkup-editor
MLSYSTEM2_IMAGES_ROOT=/data/mlsystem2/prepared_images
MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT=/data/mlsystem2/training-ui/files
MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT=/data/mlsystem2/training-ui/tmp
MLSYSTEM2_TRAINING_UI_FRONTEND_DIST=/opt/mlsystem2/frontend
MLSYSTEM2_TRAINING_SETTINGS_PATH=/opt/mlsystem2/repo/configs/settings.server.yaml
MLSYSTEM2_MLFLOW_TRACKING_URI=http://127.0.0.1:5000/mlflow
MLSYSTEM2_MLFLOW_UI_URL=/mlflow/
MLSYSTEM2_GRAFANA_URL=/grafana/
MLSYSTEM2_IMAGES_UI_URL=/prepared-images/
MLSYSTEM2_TRAINING_UI_USERS_JSON='[{"username":"<администратор>","password":"<пароль>","role":"admin","aliases":["mlsystem"]},{"username":"<пользователь>","password":"<пароль>","role":"user"}]'
MLSYSTEM2_TRAINING_UI_SESSION_SECRET=<случайная строка>
MLSYSTEM2_FEEDBACK_API_TOKEN=<случайный_служебный_токен>
MLSYSTEM2_TRAINING_UI_WORKER_ENABLED=false
MLSYSTEM2_TRAINING_UI_WORKER_INTERVAL_SECONDS=5
MLSYSTEM2_AUTOMATION_SYNC_INTERVAL_SECONDS=30
MLSYSTEM2_TRAINING_TORCH_NUM_THREADS=4
MLSYSTEM2_TRAINING_TORCH_NUM_INTEROP_THREADS=2
MLSYSTEM2_TRAINING_PROCESS_NICE=10
MLSYSTEM2_TRAINING_PROCESS_IO_PRIORITY=7
MLSYSTEM2_PSEUDOLABEL_API_TOKEN=
MLSYSTEM2_PSEUDOLABEL_MAX_AOI_AREA_M2=0
MLSYSTEM2_PSEUDOLABEL_MAX_VERTICES=10000
MLSYSTEM2_PSEUDOLABEL_JOB_TIMEOUT_SECONDS=3600
MLSYSTEM2_PSEUDOLABEL_IMAGERY_PROVIDERS_PATH=/etc/mlsystem2/imagery-providers.yaml
MLSYSTEM2_PSEUDOLABEL_IMAGE_SCAN_WORKERS=8
MLSYSTEM2_PSEUDOLABEL_TILE_READ_WORKERS=4
MLSYSTEM2_PSEUDOLABEL_PREFETCH_BATCHES=2
MLSYSTEM2_PSEUDOLABEL_EXTERNAL_HTTP_WORKERS=8
```

`MLSYSTEM2_TRAINING_UI_USERS_JSON` задаёт канонических пользователей, индивидуальные пароли, роли и optional aliases.
Cookie и черновики используют каноническое имя даже при входе через alias; оно же становится реальным Git author
публикации. Роли `admin|user` уже возвращаются API, но ограничения функций по ним пока не применяются. Alias
`mlsystem` сохраняет совместимость QGIS-плагина. Если JSON не задан, временно действует прежний одиночный контракт
`MLSYSTEM2_TRAINING_UI_USER`, `MLSYSTEM2_TRAINING_UI_USER_ALIASES`, `MLSYSTEM2_TRAINING_UI_PASSWORD`.
`MLSYSTEM2_PSEUDOLABEL_API_TOKEN` нужен только отдельным неинтерактивным клиентам. QGIS-плагин
входит по логину и паролю, поэтому для него токен оставляют пустым. Нулевой
`MLSYSTEM2_PSEUDOLABEL_MAX_AOI_AREA_M2` разрешает AOI любой площади;
положительное значение включает операторский лимит в квадратных метрах.
Worker запускается отдельным systemd-сервисом, поэтому встроенный worker HTTP-процесса обязательно
отключён. Ограничения потоков относятся к новым обучениям. `tile_preparation.prefetch_epochs=1`
в `configs/settings.server.yaml` не уменьшается: при восьми DataLoader workers автоматически растёт
`prefetch_factor`, и в очереди остаётся тот же целевой объём тайлов.
Каталог внешних источников необязателен: без него доступны локальные Канопус, ортофото и встроенный
OpenAerialMap. За основу файла можно взять `configs/imagery-providers.example.yaml`; секреты в нём
не хранятся, указываются только имена переменных окружения.

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
- `dataset_editor_drafts`

## 4. FastAPI service

Пример systemd unit `/etc/systemd/system/mlsystem2-training-ui-api.service`:

```ini
[Unit]
Description=MLSystem2 — HTTP API интерфейса обучения
After=network.target

[Service]
WorkingDirectory=/opt/mlsystem2/repo
EnvironmentFile=/etc/mlsystem2/training-ui-api.env
ExecStart=/opt/mlsystem2/repo/.venv/bin/mlsystem2-training-ui-api
Restart=always
RestartSec=5
KillMode=process
CPUWeight=1000
IOWeight=1000

[Install]
WantedBy=multi-user.target
```

Запуск:

```bash
systemctl daemon-reload
systemctl enable --now mlsystem2-training-ui-api
curl --fail http://127.0.0.1:8091/api/v1/health
```

## 5. Worker очередей

Установить unit из `docs/runbooks/mlsystem2-training-ui-worker.service`. На сервере с 32 логическими
CPU он оставляет ядра `0-3` вне cgroup worker, чтобы HTTP API и системные службы отвечали под нагрузкой.
На сервере с другой топологией значение `AllowedCPUs` нужно скорректировать.

```bash
install -m 0644 \
  /opt/mlsystem2/repo/docs/runbooks/mlsystem2-training-ui-worker.service \
  /etc/systemd/system/mlsystem2-training-ui-worker.service
systemctl daemon-reload
systemctl enable --now mlsystem2-training-ui-worker
systemctl status mlsystem2-training-ui-api mlsystem2-training-ui-worker
```

Обучение и его DataLoader-процессы наследуют низкие `CPUWeight/IOWeight` worker-cgroup, а стартовый
скрипт дополнительно применяет `nice` и `ionice`. Срочный поснимочный инференс не завершает обучение:
train loop и DataLoader workers ждут снятия `pause.request`, после чего продолжают тот же MLflow-run.

## 6. Frontend static

CI/CD доставляет `frontend/dist` через `frontend/deploy.py` в путь из `MLSYSTEM2_FRONTEND_DIST_PATH`,
по умолчанию `/opt/mlsystem2/frontend`. Для ручной доставки применяется тот же путь:

```bash
python3 frontend/deploy.py --source frontend/dist --target /opt/mlsystem2/frontend
```

Сначала копируются зависимости, затем атомарно заменяется `index.html`. Нельзя применять `rsync --delete`
к опубликованному каталогу `assets`: открытые вкладки продолжают запрашивать старые хешированные модули,
стили и декодеры TIFF. Скрипт сохраняет их до семи суток, а при превышении общего целевого размера
128 МиБ удаляет старейшие неактуальные файлы. Текущая сборка защищена от очистки. Публичные иллюстрации
новостей сохраняются. HTML не кэшируется; ошибки `/assets/*.js` с кодом 404 проверяются отдельно от TIFF.

Reverse proxy должен:

- проксировать `/` и `/api/v1/` на `http://127.0.0.1:8091`;
- проксировать `/mlflow/` и `/grafana/` как в старом gateway;
- публиковать `/data/mlsystem2/prepared_images` через подключение только для чтения и защищённый путь
  `/prepared-images/`; MinIO используется только для артефактов MLflow и не хранит вторую копию снимков.

Для контейнера обратного прокси добавить подключение:

```yaml
volumes:
  - /data/mlsystem2/prepared_images:/srv/prepared-images:ro
```

В HTTPS-сервер добавить путь с той же проверкой сессии, что используется остальным сайтом:

```nginx
location /prepared-images/ {
    auth_request /auth/proxy-check;
    alias /srv/prepared-images/;
    autoindex on;
    autoindex_exact_size off;
    autoindex_localtime on;
    charset utf-8;
}
```

## 7. Отключение старого сайта

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

## 8. Обновление MLflow

Клиент приложения закреплён в `pyproject.toml`: `mlflow==3.16.1`.
Сервер использует `ghcr.io/mlflow/mlflow:v3.16.1`. Его контейнер
`mlsystem-gpu-mlflow` управляется файлом `/data/mlsystem/platform/docker-compose.yml`,
а переменная `MLFLOW_IMAGE` находится в `/etc/mlsystem/gpu-platform.env`.
При обновлении нужно согласованно менять клиент, эту переменную и версию образа по умолчанию
в Compose. CI/CD обновляет клиент; сервер MLflow обновляется вручную.

Порядок обновления:

1. Проверить совместимость адаптера: `python -m pytest tests/test_mlflow_adapter.py`.
   Проверка с настоящим MLflow использует отдельную SQLite-базу и временные артефакты.
2. Сохранить Compose, закрытый env-файл и `pg_dump -Fc` базы `mlflow` контейнера
   `mlsystem-gpu-mlflow-postgres`. Резервные копии хранить с доступом только владельцу
   в `/data/mlsystem/mlflow/backups/`. Не выводить секреты в журналы.
3. Восстановить копию в отдельную проверочную базу, выполнить новым образом
   `mlflow db upgrade <URI проверочной базы>` и проверить запуск сервера.
4. Дождаться пустой очереди и отсутствия работающих обучений. На время миграции остановить
   `mlsystem2-training-ui-worker`, затем только контейнер MLflow; обновить резервную копию базы.
5. Выполнить `mlflow db upgrade <URI рабочей базы>`. Пересоздать только сервис `mlflow`
   через `docker compose --env-file /etc/mlsystem/gpu-platform.env
   -f /data/mlsystem/platform/docker-compose.yml up -d --no-deps mlflow`.
6. Проверить `/mlflow/version` и `/mlflow/health` на `http://127.0.0.1:5000`, затем
   авторизованный доступ через `https://grovika.ru/mlflow/`. В MLflow 3.16.1 `--static-prefix`
   применяется также к REST API: tracking URI клиента должен быть
   `http://127.0.0.1:5000/mlflow`, а внутри Docker-сети — `http://mlflow:5000/mlflow`.
   Обновить `MLSYSTEM2_MLFLOW_TRACKING_URI` в env приложения и `MLFLOW_TRACKING_URI` в env
   платформы до возобновления worker.
7. Проверить чтение существующего запуска и контрольную запись метрик, датасета, параметров,
   загрузку и скачивание артефакта через адаптер и MinIO. После проверки возобновить worker
   и удалить только созданную для проверки базу и временный контейнер.

Перед возвратом старого образа восстановить согласованную резервную копию базы:
миграции MLflow не гарантируют обратимость. Объекты существующих запусков в MinIO
при обновлении не изменяются. Резервная копия перехода с сервера 3.11.1 на 3.16.1:
`/data/mlsystem/mlflow/backups/20260923T060621Z-before-3.16.1`.

## 9. Проверка CUDA после обновления драйвера

Обучение проверяет выбранное CUDA-устройство до подготовки датасета и тайлов. Ошибка этой
проверки записывается в отчёт конвейера и завершает MLflow run; автоматического перехода на CPU нет.

Если `nvidia-smi` сообщает `Driver/library version mismatch`, а Torch — CUDA 804, сравнить
загруженный модуль, установленный модуль и библиотеку драйвера:

```bash
nvidia-smi
cat /proc/driver/nvidia/version
modinfo -F version nvidia
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
```

После обновления пакетов старый модуль может оставаться загруженным до перезагрузки.
Переустановка Torch не устраняет это несовпадение. Перед перезагрузкой общего GPU-сервера
нужно дождаться завершения работающих заданий, приостановить запуск новых через существующие
переключатели очередей и проверить остальные GPU-сервисы и их автоматический запуск.
Состояния переключателей сохранить и вернуть после проверки; ожидающие задания и результаты не удалять.

На сервере без графической сессии вместо полной перезагрузки можно остановить только потребителей
GPU и перезагрузить модули NVIDIA. Сначала проверить открытые `/dev/nvidia*` во всех процессах:
останавливать контейнеры с обучениями или интерактивной работой нельзя. После завершения запросов
Triton остановить его контейнеры и GPU-exporter, выполнить
`modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia`, затем `modprobe nvidia` и
`modprobe nvidia_uvm`. Если модуль остаётся занят, не применять принудительную выгрузку.
Перед возвратом контейнеров выполнить `systemctl restart nvidia-cdi-refresh.service`:
NVIDIA Container Toolkit должен пересобрать `/var/run/cdi/nvidia.yaml` под загруженную версию
драйвера. Старый CDI после обновления пакетов ссылается на удалённые библиотеки и не даёт
запустить GPU-контейнеры. Вернуть ранее работавшие контейнеры, проверить готовность Triton
и CUDA, затем вернуть очереди.

После перезагрузки проверить `nvidia-smi`, совпадение версий, API и CUDA в окружении приложения:

```bash
cd /opt/mlsystem2/repo
.venv/bin/python -c 'import torch; print((torch.zeros(1, device="cuda") + 1).item())'
curl --fail http://127.0.0.1:8091/api/v1/health
```

Если обучение упало до первой эпохи, повторить задание с прежними параметрами после восстановления GPU.
Исходную ошибку и историю результата сохранить. Не повторять уже выполняющееся или завершённое задание.
