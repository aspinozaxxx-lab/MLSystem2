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
- `MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT` — корень загруженных txt/geojson и постоянных тестовых разметок.
- `MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT` — временные файлы jobs.
- `MLSYSTEM2_TRAINING_UI_FRONTEND_DIST` — каталог собранного frontend, default `/opt/mlsystem2/frontend`.
- `MLSYSTEM2_TRAINING_SETTINGS_PATH` — путь к стабильному `settings.yml`, default `configs/settings.server.yaml`
  относительно `MLSYSTEM2_PROJECT_ROOT`.
- `MLSYSTEM2_MLFLOW_TRACKING_URI` или `MLFLOW_TRACKING_URI` — internal MLflow tracking URI.
- `MLSYSTEM2_TRAINING_UI_USER` или `MLSYSTEM_FRONTEND_USER` — пользователь входа.
- `MLSYSTEM2_TRAINING_UI_PASSWORD` или `MLSYSTEM_FRONTEND_PASSWORD` — пароль входа.
- `MLSYSTEM2_TRAINING_UI_SESSION_SECRET` или `MLSYSTEM_FRONTEND_SESSION_SECRET` — секрет подписи cookie.
- `MLSYSTEM2_GRAFANA_URL`, `MLSYSTEM2_MLFLOW_UI_URL`, `MLSYSTEM2_MINIO_UI_URL` — ссылки на UI сервисов.
- `MLSYSTEM2_OPEN_WEBUI_URL` — ссылка на Open WebUI LLM-стека, default `/open-webui/`.
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
- `GET /api/v1/image-folders`
- `GET /api/v1/classes`
- `GET /api/v1/dataset-catalog`
- `POST /api/v1/dataset-catalog/sync`
- `POST /api/v1/dataset-classes`
- `PATCH /api/v1/dataset-classes/{class_key}`
- `PUT /api/v1/dataset-classes/{class_key}/primary-dataset`
- `POST /api/v1/managed-datasets`
- `PATCH /api/v1/managed-datasets/{dataset_key}`
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
- `GET /api/v1/inference-templates`
- `POST /api/v1/inference-templates`
- `PUT /api/v1/inference-templates/by-id/{template_id}`
- `DELETE /api/v1/inference-templates/by-id/{template_id}`
- `PUT /api/v1/inference-templates/by-id/{template_id}/apply-field-to-all`
- `GET /api/v1/inference-templates/{architecture}`
- `PUT /api/v1/inference-templates/{architecture}`
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
- `GET /api/v1/results/datasets/{dataset_key}`
- `POST /api/v1/results/datasets/{dataset_key}/pseudo-markup`
- `POST /api/v1/results/datasets/{dataset_key}/test-f1`
- `POST /api/v1/results/training/{result_id}/triton-zip`
- `POST /api/v1/markup-export`
- `GET /api/v1/markup-export/{export_id}/tiles/{tile_index}/preview`
- `GET /api/v1/markup-export/{export_id}/download`
- `GET /api/v1/test-samples`
- `POST /api/v1/test-samples`
- `POST /api/v1/test-sample-batches`
- `GET /api/v1/test-sample-batches/latest`
- `GET /api/v1/test-sample-batches/{batch_id}`
- `GET /api/v1/test-samples/{sample_id}`
- `PATCH /api/v1/test-samples/{sample_id}`
- `DELETE /api/v1/test-samples/{sample_id}`
- `PATCH /api/v1/test-samples/{sample_id}/tiles/{tile_index}`
- `POST /api/v1/test-samples/{sample_id}/evaluate`
- `POST /api/v1/test-samples/{sample_id}/optimize`
- `PUT /api/v1/test-samples/{sample_id}/primary`
- `POST /api/v1/test-samples/download`
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview`
- `GET /api/v1/test-samples/{sample_id}/download`
- `POST /api/v1/test-samples/{sample_id}/download`
- `DELETE /api/v1/results/pseudo-markup/{result_id}`
- `GET /api/v1/files/{file_id}/download`

OpenAPI доступен стандартно по `/openapi.json`.

`GET /api/v1/datasets` возвращает плоский список датасетов MLMarkup, а `GET /api/v1/classes` и
`GET /api/v1/results/classes` — классы с вложенным списком `datasets`. Класс задаёт единый тип снимков:
`kanopus` использует четыре канала из `prepared_images/kanopus`, `ortho` — три RGB-канала из
`prepared_images/orto`. В папке датасета ожидается один TXT со списком сцен, один ordinary positive GeoJSON и optional
`hard_negative.geojson`. `hard_negative.geojson` возвращается как `hard_negative_annotation_file` и не выбирается
как positive `annotation_file`; несколько обычных GeoJSON дают diagnostics вместо случайного выбора.
`updated_at` датасета заполняется по последнему git-коммиту, затронувшему его папку в
`MLSYSTEM2_MLMARKUP_ROOT`. Если `MLSYSTEM2_MLMARKUP_ROOT` не является git checkout, используется filesystem
mtime как fallback. `version` равен `git:{commit_sha}` или `fs:{mtime_ns}` и используется автоматизацией
для дедупликации jobs по конкретной версии датасета. `image_count` считается по txt-списку сцен через
индекс снимков внутри корня типа класса: строки-папки разворачиваются в фактические TIFF, повторы
удаляются.

`POST /api/v1/markup-export` формирует самостоятельный набор тестовой разметки и не создает job или запись в БД.
Доступны только однозначные датасеты MLMarkup с TXT и одним GeoJSON положительной разметки; `Custom` и
`hard_negative.geojson` не участвуют. TIFF читаются только из `MLSYSTEM2_IMAGES_ROOT`, в рабочей конфигурации это
`/data/mlsystem2/prepared_images`. Окна обязаны целиком находиться внутри растра, иметь полностью валидную
`dataset_mask` и не содержать пикселей без данных или полностью чёрных пикселей. Выбор через `scipy.optimize.milp` сначала
максимизирует число территорий и исходных TIFF, затем минимизирует отклонение от целевого числа объектов.
Положения строятся и вдоль протяжённых геометрий: один GeoJSON-объект может войти в несколько непересекающихся
тайлов и считается отдельным объектом в каждом. Результат содержит по четыре файла на тайл:
`tile_001.tif`, `tile_001.geojson`, `tile_001_mask.png` и `tile_001_preview.png`. Геометрии GeoJSON приводятся к
единому типу `MultiPolygon`, поэтому файл открывается в QGIS одним слоем. Плоский ZIP получает имя вида
`вырубки_test_markup.zip` по русскому имени класса и вместе с превью хранится один час в
`scratch_root/markup-exports`, после истечения срока возвращается `404`.

`POST /api/v1/test-samples` использует ту же нарезку, но сохраняет готовые файлы без TTL в
`MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT/test-samples/{uuid}`, а описание и состояния тайлов — в Postgres.
Каталог группирует тестовые разметки как `класс → датасет`. Выключенные тайлы остаются доступными для возврата, но не входят
в скачиваемый ZIP и расчёт F1; полное удаление разметки удаляет запись и весь каталог файлов. Сервис считает
пиксельную и объектную F1 по последней успешной псевдоразметке с точным совпадением `class_key` и входного
`dataset_key`; порог объектного сопоставления равен `IoU ≥ 0,5`. После переключения прежние значения видны как
устаревшие до пересчёта, а новый подходящий результат псевдоразметки запускает автоматический пересчёт.
Редактор использует `evaluate-preview` и `optimize-preview` без записи; имя, основной статус и полный состав
применяются одним `PATCH /api/v1/test-samples/{sample_id}`.
Скачивание из редактора передаёт текущий состав и `include_previews` в
`POST /api/v1/test-samples/{sample_id}/download`, формирует временный ZIP выбранных тайлов и не сохраняет
черновик. Совместимый `GET` скачивает сохранённый состав с превью. Без превью ZIP содержит строго TIFF и
GeoJSON; полный состав дополнительно содержит PNG-маску и полноразмерные JPEG с жёлтым контуром и без него.
Тайлы последовательно нумеруются как `tile001..tileNNN`. Для RGB TIFF создаётся только `rgb`, для
четырёхканального TIFF — `rgb/nrg/ngb` с композициями `RED-GRN-BLU`, `NIR-RED-GRN` и `NIR-GRN-BLU`.
Превью формируются на лету, не сохраняются постоянно и кодируются с максимальным качеством, укладывающимся в
`300 KiB`. Временный `/markup-export` и PNG-превью редактора остаются совместимыми с прежним форматом.
`POST /api/v1/test-samples/download` принимает уникальный непустой список сохранённых разметок. Каждая
готовится отдельной задачей пула максимум из восьми потоков в собственном временном каталоге; SQLAlchemy-сессия
и `ZipFile` между потоками не разделяются. После успешной подготовки один поток собирает общий ZIP с отдельной
папкой каждой разметки, все записи имеют `ZIP_STORED`. При ошибке частичный архив и временные файлы удаляются.
`POST /api/v1/test-samples/{sample_id}/optimize` рассматривает включённые и выключенные тайлы, соблюдает минимум и
максимум тайлов и минимум объектов, затем атомарно применяет состав с максимальным агрегированным пиксельным либо
объектным F1. При равном F1 приоритетны территории, число объектов, исходные снимки и меньший состав.

`POST /api/v1/test-sample-batches` создаёт один последовательный групповой запуск. Для каждой строки сервис
строит непересекающийся пул до тройного максимума итоговых тайлов с целью тройного минимума объектов, проверяет
достижимость итоговых ограничений и по последней точной псевдоразметке включает состав внутри диапазона
`min_image_count..image_count` с максимальным выбранным F1. Без `min_image_count` запрос сохраняет прежний
режим точного числа тайлов. Геометрия восстанавливается через `make_valid` после смены CRS. Если вторичный этап
оптимизации пула достигает лимита времени, сохраняется последнее допустимое решение; невозможный минимум
возвращает достижимый максимум объектов с учётом конфликтов тайлов. Статусы группы и строк сохраняются в Postgres и восстанавливаются после перезапуска.
`GET /api/v1/test-sample-batches/latest` одновременно является источником последних значений формы: размера,
минимума и максимума тайлов, минимального числа объектов и метрики каждого участвовавшего датасета.

У каждого точного `dataset_key` может быть одна основная тестовая разметка. Совместимое назначение выполняется через
`PUT /api/v1/test-samples/{sample_id}/primary`. Для каждой успешной сети таблица
`training_result_test_metrics` хранит отдельные пиксельный и объектовый F1 на основной разметке. Задание
`purpose=test_sample_f1` использует общую inference-очередь, best checkpoint и активный inference-шаблон,
обрабатывает каждый включённый TIFF независимо и суммирует TP/FP/FN. Компоненты, касающиеся границы TIFF, не
удаляются фильтрами минимального размера, площади и компактности, так как это обрезанные фрагменты объектов;
внутренние компоненты проходят полную постобработку. Метрика устаревает при смене разметки, ревизии состава,
профиля, эффективного шаблона или версии алгоритма оценки; успешная новая сеть и изменение основной разметки
ставят оценки в очередь автоматически, старт сервиса восстанавливает отсутствующие и устаревшие расчёты,
а ручной повтор выполняет `POST /api/v1/results/datasets/{dataset_key}/test-f1`. В MLflow эти метрики не
записываются.

`GET /api/v1/image-folders` возвращает папки внутри `kanopus` и `orto`, в которых TIFF лежат напрямую. Ключ и
имя папки — относительный путь, например `orto/ryazan`; `imagery_type` сообщает тип, `image_count` — количество TIFF.

На странице запуска обучения frontend по умолчанию выбирает существующий MLflow experiment с максимальным числовым
`experiment_id`. Поле `Новое имя experiment` показывается только при выборе пункта `Новый experiment`. Ручное поле
`MLflow run name` не показывается и не отправляется: при пустом имени worker не передает `--run-name`, а имя run
формируется модулем `mlflow_adapter` по tag `class`.

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
- `inference_templates`
- `stored_files`
- `custom_datasets`
- `jobs`
- `queue_controls`
- `automation_controls`
- `automation_rules`
- `training_results`
- `pseudo_markup_results`
- `test_samples`
- `test_sample_tiles`

Шаблон хранит `config_schema` и `default_config` в JSONB. `default_config` использует ключи вида
`train.learning_rate`, `tile_preparation.tile_size`; это только параметры, которые оператор меняет на сайте.
В этот набор входят `tile_preparation.positive_factor`, `tile_preparation.hard_negative_factor`,
`tile_preparation.background_factor`, `train.pos_weight`, `train.hard_negative_weight` и
`train.max_training_time_sec`: пустое значение означает обучение без
wall-clock лимита. Схема каждого параметра хранит label, tooltip, допустимые границы и рекомендуемый диапазон
для одинаковых подсказок на страницах шаблонов, запуска и просмотра задания. Сумма трех tile factors должна быть равна `1`; если hard-negative разметки или tiles нет,
недостающая hard-negative доля используется как positive внутри общего marked-бюджета.
`train.hard_negative_weight` не меняет sampler и не взвешивает весь tile: он усиливает loss только на pixels,
которые в supervision mask пришли из `hard_negative_annotation_file`.
Инфраструктурные defaults DataLoader, `train.device=cuda`, binary task и каналы модели задаются модулем
`settings` и не сохраняются в UI-шаблонах.

`training_templates.dataset_key` nullable. Строка с `dataset_key=null` является базовым шаблоном сети.
Строка с заполненным `dataset_key` является переопределением для конкретного датасета MLMarkup, например
`Вырубки\test`, и создается через `POST /api/v1/training-templates` копированием текущих defaults базового
шаблона сети. При ручном запуске и автоматизации сервис сначала ищет активный шаблон `(architecture,
dataset_key)`, а если его нет, использует базовый шаблон `(architecture, null)`. Базовый шаблон удалить нельзя;
датасетный шаблон удаляется через `DELETE /api/v1/training-templates/by-id/{template_id}`. Endpoint
`PUT /api/v1/training-templates/by-id/{template_id}/apply-field-to-all` устанавливает одно поле во всех
существующих шаблонах и помечает их `source=manual`.

`inference_templates` устроены аналогично, но содержат только параметры Geoalert-совместимой
постобработки: `postprocess.mask_min_object_pixels`, `postprocess.mask_min_hole_pixels`,
`postprocess.binary_closing_radius`, `postprocess.min_area_m2`, `postprocess.min_hole_area_m2`,
`postprocess.simplify_m` и параметры `postprocess.filter_compact_objects.*`. При ручном и автоматическом
создании псевдоразметки сервис ищет активный шаблон инференса по датасету обученной модели
`(architecture, training_dataset_key)`, а если его нет, использует базовый `(architecture, null)`; выбранный
датасет, папка или загруженный TXT со снимками на шаблон не влияют. Для `Реки\main` и `smp_segformer_b2` начальный шаблон включает
профиль 18: `min_area=10000 м²`, `min_hole_area=5000 м²`, `Simplify=15 м` и фильтр компактных объектов
`min_isoperimetric_quotient=0.25`, `max_bbox_ratio=3.5`.

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
и не записываются в `run.yml`. Worker всегда записывает в `run.yml` нормализованные три tile factors,
`train.hard_negative_weight` и добавляет
`hard_negative_annotation_file`, если он найден у встроенного MLMarkup dataset. Секция `inference` в training `run.yml` не создается: checkpoint, threshold,
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

`GET /api/v1/results/changes` возвращает сначала active jobs со статусами `queued`/`running`, затем последние
20 успешных изменений из `training_results` и `pseudo_markup_results`, отсортированные по времени изменения.
Каждая строка содержит `item_type`, optional `job_id`, `type`, `dataset_key`, `class_key`, `class_name`, имя модели, имя датасета, status,
optional `mlflow_run_url` и действие: `запланировано обучение`, `идёт обучение`, `запланирована псевдоразметка`,
`идёт псевдоразметка`, `обучена сеть` или `создана разметка`.
`GET /api/v1/results/datasets/{dataset_key}` возвращает активные строки обучения и псевдоразметки прямо в основной
структуре датасета: `TrainingResultInfo` и `PseudoMarkupResultInfo` содержат необязательный `job_id`, а статус
`queued`/`running` берется из связанного задания, пока результат еще не завершен. `TrainingResultInfo.created_at`
содержит время создания строки результата, а `started_at` - время запуска связанного job, если job успел стартовать.

При успешном завершении training job worker читает через публичный `mlflow_adapter.api` историю метрики
`val/best_threshold_pixel_f1`, по которой train-модуль сохраняет `checkpoints/best.pt`, и записывает лучший
F1 и эпоху в `training_results.f1_score`/`training_results.epoch`. Pseudo-markup job, созданный от результата
обучения, получает в `jobs.config` `mlflow_run_id`, `checkpoint_artifact_path=checkpoints/best.pt`,
`inference_template_id`, `inference_template_config` и, когда MLflow доступен, полный `checkpoint_uri`.
Для успешного результата обучения `POST /api/v1/results/training/{result_id}/triton-zip` скачивает тот же
`checkpoints/best.pt` из MLflow, собирает временный архив Triton CPU тем же кодом, что endpoint checkpoint-экспорта,
и возвращает файл без записи в Postgres, MLflow, S3 или рабочий каталог инференса. `POST /api/v1/results/training/triton-zip`
принимает список успешных результатов и имен моделей, собирает каждую модель тем же кодом и возвращает общий zip с
`models-serving-service/`, `pipelines/`, `metadata/` и корневым `export_metadata.json`.

Для job типа `pseudo-markup` worker берет txt список снимков из выбранного датасета, выбранной папки снимков или
загруженного файла, пишет в `pseudo_config.yaml` `inference_backend=pytorch_one_off`, скачивает `checkpoints/best.pt`
через публичный `mlflow_adapter.api.download_run_artifact`, загружает checkpoint через `models.api.load_checkpoint`,
строит GeoJSON псевдоразметки в `EPSG:4326` и сохраняет его в `stored_files` для скачивания через
`/api/v1/files/{file_id}/download`. Этот путь не создает Triton model archive, Geoalert pipeline YAML или запись в
Triton model repository; после обработки или ошибки загрузки checkpoint раннер освобождает CUDA cache. Трёхканальный
checkpoint принимает RGB и RGBA GeoTIFF: у RGBA читаются только первые три канала. Все остальные несовпадения
числа каналов отклоняются; обучение и основной CLI-инференс остаются строгими. При выборе
папки сервис создает stored txt с одной строкой-относительным путем папки. `PseudoMarkupResultInfo.image_count`
содержит сохраненное в БД количество фактически найденных снимков по txt, включая загруженные custom txt и
строки-папки. `StoredFileInfo.size_bytes` и nullable `StoredFileInfo.object_count` берутся из БД и используются
frontend для отображения размера и количества объектов скачиваемого GeoJSON; страница результатов не читает
GeoJSON и не обходит корень снимков при открытии. Имя скачиваемого GeoJSON начинается с отображаемых названий
класса и датасета из результата обучения, а технический UUID датасета используется только как fallback для
неполных legacy-записей.
`GET /api/v1/jobs/{job_id}/log` сначала отдает `worker_error.txt`, `train.log`, `logs/train.log` или
`logs/pseudo_markup.log` из рабочей папки задания. Если локального лога нет или он указывает смотреть journalctl,
endpoint возвращает фрагмент `journalctl` по unit из `MLSYSTEM2_TRAINING_UI_JOURNAL_UNIT`.
`DELETE /api/v1/results/pseudo-markup/{result_id}` удаляет строку результата, связанный inference job и
принадлежащие UI-сервису stored files внутри `MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT`; исходные файлы MLMarkup не
удаляются.

Перед инференсом раннер ищет реальные TIFF по строкам txt, удаляет повторные совпадения и выбирает профиль
постобработки по числу уникальных снимков. Для `<=5` снимков используется прежний режим без постобработки, для
`6..50` — `detail_v2` с легкой чисткой маски и `Simplify=10 м`, для `>=51` — `strong` без `binary_closing`,
с чисткой маски `48 px`, `min_area=3000 м²`, `min_hole_area=5000 м²` и `Simplify=15 м`.
После выбора автоматического профиля раннер применяет непустые поля из `inference_template_config`, поэтому
шаблон может ужесточить или ослабить только нужные параметры. Включенный `filter_compact_objects` удаляет
компактные полигоны по isoperimetric quotient и отношению сторон minimum rotated rectangle; для рек это
используется для отсечения озер и прудов.
Перед записью итогового скачиваемого GeoJSON раннер сливает
пересекающиеся и касающиеся полигоны через `unary_union`; per-scene GeoJSON остаются диагностическими файлами
без глобального слияния. Отчет псевдоразметки содержит `inference_backend=pytorch_one_off`, `triton_model=null`,
выбранный профиль, число уникальных снимков, параметры постобработки, `feature_count_before_merge` и финальный
`feature_count`; HTTP API и схема БД при этом не меняются.
