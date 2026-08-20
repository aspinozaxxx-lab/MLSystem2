# Training UI API

`training_ui_api` — отдельный FastAPI-сервис сайта MLSystem2.

## Env vars

- `MLSYSTEM2_TRAINING_UI_API_HOST` — host uvicorn, default `0.0.0.0`.
- `MLSYSTEM2_TRAINING_UI_API_PORT` — port uvicorn, default `8091`.
- `MLSYSTEM2_PROJECT_ROOT` — корень установленного репозитория MLSystem2 для запуска CLI, default равен `cwd` сервиса.
- `MLSYSTEM2_TRAINING_UI_DATABASE_URL` — Postgres URL, секреты задаются только через env.
- `MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA` — отдельная схема Postgres, default `training_ui`.
- `MLSYSTEM2_MLMARKUP_ROOT` — путь к MLMarkup, default `/data/MLMarkup`.
- `MLSYSTEM2_MLMARKUP_EDITOR_ROOT` — отдельный Git-клон редактора, default `/data/mlsystem2/mlmarkup-editor`.
- `MLSYSTEM2_MLMARKUP_RELEASE_MARKER` — marker SHA опубликованного релиза, default `/data/MLMarkup/.mlsystem2-release`.
- `MLSYSTEM2_MLMARKUP_EDITOR_BRANCH` — ветка editor clone, default `main`.
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
- `MLSYSTEM2_GEOALERT_PYTHON_PATH` — Python окружения Workflow Engine, default `/opt/geoalert/inference/.venv/bin/python`.
- `MLSYSTEM2_GEOALERT_INFERENCE_ROOT` — рабочая установка Geoalert, default `/opt/geoalert/inference`.
- `MLSYSTEM2_GEOALERT_MODEL_REPOSITORY` — model repository explicit-control Triton, default `/opt/geoalert/triton_models`.
- `MLSYSTEM2_GEOALERT_PIPELINE_ROOT` — кеш сгенерированных pipeline, default `/opt/geoalert/pipelines/mlsystem2-runtime`.
- `MLSYSTEM2_GEOALERT_TRITON_HTTP_URL` — локальный HTTP API Triton, default `http://127.0.0.1:8000`.
- `MLSYSTEM2_GEOALERT_TRITON_PYTHON_SITE_PACKAGES` — read-only путь внутри контейнера Triton к
  `site-packages` окружения MLSystem2 для Python-backend внешней модели ЗУ500, default
  `/mlsystem2-venv/lib/python3.12/site-packages`. Сам контейнер запускается с `/dev/shm` не менее `1 GiB`.

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
- `GET /api/v1/dataset-editor/drafts`
- `GET /api/v1/dataset-editor/datasets`
- `DELETE /api/v1/dataset-editor/datasets/{dataset_key}`
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/scenes`
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}`
- `POST /api/v1/dataset-editor/datasets/{dataset_key}/scenes`
- `PUT /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}`
- `DELETE /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}`
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/rasters`
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/raster/{image_path}`
- `GET /api/v1/dataset-editor/publication/{commit}`
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
- `POST /api/v1/scene-list-export`
- `POST /api/v1/markup-export`
- `GET /api/v1/markup-export/{export_id}/tiles/{tile_index}/preview`
- `GET /api/v1/markup-export/{export_id}/download`
- `GET /api/v1/test-samples`
- `POST /api/v1/test-samples`
- `POST /api/v1/test-samples/reconcile`
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
`prepared_images/orto`. Папка с TXT считается legacy: в ней ожидаются один список сцен, один positive GeoJSON и
optional `hard_negative.geojson`. Папка без TXT считается per-image: каждый GeoJSON соответствует одному TIFF,
а `DatasetInfo` возвращает `format=per_image` и `annotations_dir`. Пустой управляемый per-image датасет виден
редактору, но не готов к обучению. В legacy несколько обычных GeoJSON дают diagnostics вместо случайного выбора.
Историческая основа имени модели из positive GeoJSON сохраняется в `datasets.model_name_stem` и возвращается в
`DatasetInfo`, поэтому переход на per-image не меняет предлагаемое имя выгрузки: `<имя>_kanopus` для Канопуса и
исторически совместимое `<имя>_orto` для ортофото.
`updated_at` датасета заполняется по последнему git-коммиту, затронувшему его папку в
`MLSYSTEM2_MLMARKUP_ROOT`. Атомарный runtime-релиз читает те же данные из
`.mlsystem2-release-metadata.json`; для произвольной папки без Git и release metadata используется filesystem
mtime как fallback. `version` равен `git:{commit_sha}` или `fs:{mtime_ns}` и используется автоматизацией
для дедупликации jobs по конкретной версии датасета. Для legacy `image_count` считается по TXT через индекс
снимков; для per-image — по однозначно сопоставленным GeoJSON и TIFF.

`POST /api/v1/managed-datasets/compose` создаёт внутри выбранного класса виртуальный multiclass-датасет из
нескольких обычных binary per-image источников. Запрос фиксирует их приоритеты; тип, slug и цвет объекта
соответствуют классу каждого источника. Целевая папка в MLMarkup не создаётся: чтение, обучение и автоматизация
используют детерминированную materialization в служебном cache, а её версия зависит от файлов всех источников и
явного списка сцен в Postgres. Публикация из управляемого редактора направляет positive в источник его типа,
hard negative — во все источники одним Git-коммитом. Пустой добавленный снимок остаётся только в составе
управляемого датасета; исходный GeoJSON возникает после первой опубликованной разметки. Удаление управляемого
датасета не удаляет источники, а источник нельзя удалить, пока на него есть активная ссылка.

Страница `#/dataset-editor` и endpoints `/api/v1/dataset-editor/*` работают только с per-image датасетами.
Редактор классов получает через `GET /api/v1/dataset-editor/drafts` только черновики вошедшего пользователя,
показывает их верхним блоком и открывает выбранный датасет для продолжения работы. В заголовке списка снимков
суммируются объекты каждого типа и hard negative по актуальному состоянию, включая серверные и локальные черновики;
снимки с черновой пометкой удаления в сумму не входят. Параметры виртуального управляемого датасета редактируются
той же формой состава: PATCH сохраняет ключ датасета, меняя название, источники, приоритеты и цвета, и отклоняет
изменение схемы при наличии неопубликованных черновиков любого пользователя.
Редактор читает и коммитит отдельный SSH-клон, никогда не пишет в live-каталог. Frontend автоматически сохраняет
изменения каждого снимка в пользовательский серверный черновик; публикация проверяет их все и создаёт один атомарный commit.
Совместимый одиночный save меняет один GeoJSON; добавление прямых TIFF папки обычного датасета создаёт один commit,
а пустые сцены управляемого — только строки состава в БД; delete остаётся отменяемым черновиком. Все Git-операции сериализованы,
перед мутацией выполняется fetch/fast-forward, blob revision защищает от потери конкурентных изменений и даёт
`409` без частичной batch-записи и автоматического слияния геометрий. Сервер строит footprint по фактической
`dataset_mask` TIFF, возвращает его как `valid_data_footprint` и обрезает им разметку для редактирования. При
автосохранении черновика полигональная топология восстанавливается, каждый объект пересекается с footprint, а
полностью внешний результат удаляется; нормализованный GeoJSON возвращается клиенту. Поэтому частичный выход за
границу снимка или в nodata не отменяет сохранение остальных объектов. После этого проверяются CRS снимка,
Polygon/MultiPolygon и валидность. OpenLayers отключает интерполяцию
и переходное размытие raster tiles, даёт nearest-neighbor масштаб до 1000%, немного повышает контраст и позволяет
полностью скрыть разметку отдельной кнопкой. Raster endpoint авторизован и поддерживает HTTP Range. Возвращённый commit имеет
статус `publishing`; он становится `published`, когда является предком SHA в live release marker. Если API
работает от другого системного пользователя, после каждой заблокированной операции UID/GID содержимого клона
восстанавливаются по владельцу его корневого каталога.

`POST /api/v1/scene-list-export` принимает multipart-поля `imagery_type=kanopus|ortho` и `geojson`.
Сервис рекурсивно читает TIFF только из `prepared_images/kanopus` или `prepared_images/orto`, преобразует
полигональную разметку в CRS каждого снимка и включает сцену при пересечении с положительной площадью валидных
пикселей по нативной `dataset_mask`. Снимок, где вся разметка лежит в `nodata`, не включается. Ответ —
UTF-8 TXT с отсортированными относительными путями внутри корня типа снимков без `.tif/.tiff`; пустой список
допустим. Одинаковые имена в разных папках различаются путём. Если совпадает сам относительный путь без
расширения, операция завершается ошибкой. Имя скачивания повторяет имя GeoJSON с расширением `.txt` и
сохраняет кириллицу.

`POST /api/v1/markup-export` формирует самостоятельный набор тестовой разметки и не создает job или запись в БД.
Доступны однозначные legacy и per-image датасеты MLMarkup; `Custom` и hard-negative объекты не участвуют. Для
per-image каждая сцена использует только positive-объекты собственного GeoJSON. TIFF читаются только из
`MLSYSTEM2_IMAGES_ROOT`, в рабочей конфигурации это
`/data/mlsystem2/prepared_images`. Окна обязаны целиком находиться внутри растра, иметь полностью валидную
`dataset_mask` и не содержать пикселей без данных или полностью чёрных пикселей. Выбор через `scipy.optimize.milp` сначала
максимизирует число территорий и исходных TIFF, затем минимизирует отклонение от целевого числа объектов.
Положения строятся и вдоль протяжённых геометрий: один GeoJSON-объект может войти в несколько непересекающихся
тайлов и считается отдельным объектом в каждом. Результат содержит по четыре файла на тайл:
`tile_001.tif`, `tile_001.geojson`, `tile_001_mask.png` и `tile_001_preview.png`. Геометрии GeoJSON приводятся к
единому типу `MultiPolygon`, поэтому файл открывается в QGIS одним слоем. Каждая обрезанная исходная геометрия
остаётся отдельной feature. PNG-маска является семантической маской классов, а контур превью строится по
instance-ID feature, поэтому соприкасающиеся объекты не сливаются визуально. Плоский ZIP получает имя вида
`вырубки_test_markup.zip` по русскому имени класса и вместе с превью хранится один час в
`scratch_root/markup-exports`, после истечения срока возвращается `404`.

`POST /api/v1/test-samples` использует ту же нарезку, но сохраняет готовые файлы без TTL в
`MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT/test-samples/{uuid}`, а описание и состояния тайлов — в Postgres.
Каталог группирует тестовые разметки как `класс → датасет`. Выключенные тайлы остаются доступными для возврата, но не входят
в скачиваемый ZIP и прямой расчёт F1; полное удаление разметки удаляет запись и весь каталог файлов. Для галереи
рядом с полным `tile_NNN_preview.png` хранится JPEG-миниатюра до `384×384`. Новые наборы получают её при
создании, исторические — лениво через endpoint `thumbnail`; оба изображения отдаются с приватным неизменяемым
браузерным кэшем. Версия renderer входит в URL, а старое превью лениво перестраивается при первом запросе.
Итоговые
pixel/object-метрики каждой сохранённой разметки рассчитываются текущей основной сетью класса через общую
inference-очередь независимо от её датасета. `POST /api/v1/test-samples/reconcile` идемпотентно ставит отсутствующие
и устаревшие оценки, а `POST /api/v1/test-samples/{sample_id}/evaluate` принудительно повторяет одну оценку.
Если сеть явно не отмечена звездой, все эти расчёты используют последнюю успешную сеть класса. Такой выбор не
записывается как основной и не меняет `TrainingResultInfo.is_primary`; новое успешное обучение только запускает
сверку зависящих от эффективной сети расчётов.
Повторное нажатие на заполненную звезду вызывает `DELETE /api/v1/results/training/{result_id}/primary`, очищает
явную ссылку класса и возвращает систему к этому неявному выбору без удаления результата обучения.
Псевдоразметка с точным совпадением датасета используется только для создания, `evaluate-preview`,
`optimize-preview` и поснимочного кэша оптимизатора; новый pseudo result не перезаписывает контрольные метрики.
Имя, основной статус и полный состав применяются одним `PATCH /api/v1/test-samples/{sample_id}`.
Скачивание из редактора передаёт текущий состав и `include_previews` в
`POST /api/v1/test-samples/{sample_id}/download`, формирует временный ZIP выбранных тайлов и не сохраняет
черновик. Совместимый `GET` скачивает сохранённый состав с превью. Без превью ZIP содержит строго TIFF и
GeoJSON; полный состав дополнительно содержит PNG-маску и полноразмерные JPEG с жёлтым контуром и без него.
GeoJSON копируется в ZIP байт-в-байт, а жёлтые контуры помеченных JPEG строятся по отдельным instance-ID его
feature. Объектовая метрика сопоставляет векторные feature один к одному и не использует connected components
семантической PNG-маски.
Тайлы последовательно нумеруются как `tile001..tileNNN`. Для RGB TIFF создаётся только `rgb`, для
четырёхканального TIFF — `rgb/nrg/ngb` с композициями `RED-GRN-BLU`, `NIR-RED-GRN` и `NIR-GRN-BLU`.
Превью формируются на лету, не сохраняются постоянно и кодируются с максимальным качеством, укладывающимся в
`300 KiB`. Временный `/markup-export` и полноразмерные PNG редактора остаются совместимыми с прежним форматом.
`POST /api/v1/test-samples/download` принимает уникальный непустой список сохранённых разметок, не более одной
для каждого `dataset_key`. Каждая готовится отдельной задачей пула максимум из восьми потоков в собственном
временном каталоге; SQLAlchemy-сессия и `ZipFile` между потоками не разделяются. После успешной подготовки один
поток собирает общий ZIP, все записи имеют `ZIP_STORED`, а папки называются `<класс>_<датасет>` без имени
разметки, даты и UUID. Повтор датасета и коллизия нормализованных имён папок отклоняются до сборки. При ошибке
частичный архив и временные файлы удаляются.
`POST /api/v1/test-samples/{sample_id}/optimize` рассматривает включённые и выключенные тайлы, соблюдает минимум и
максимум тайлов и минимум объектов, затем атомарно применяет состав с максимальным агрегированным пиксельным либо
объектным F1. При равном F1 приоритетны территории, число объектов, исходные снимки и меньший состав.
Оба вида F1 кэшируются и для каждого отдельного тайла; `TestSampleTileInfo.f1_score` и карточка снимка выбирают
пиксельное либо объектовое значение по `quality_metric` разметки. Старый тайл без кэша рассчитывается при первом
открытии подробной страницы без изменения состава и ревизии разметки.

`POST /api/v1/test-sample-batches` создаёт один последовательный групповой запуск. Для каждой строки сервис
строит непересекающийся пул до тройного максимума итоговых тайлов с целью тройного минимума объектов, проверяет
достижимость итоговых ограничений и по последней точной псевдоразметке включает состав внутри диапазона
`min_image_count..image_count` с максимальным выбранным F1. Без `min_image_count` запрос сохраняет прежний
режим точного числа тайлов. Геометрия восстанавливается через `make_valid` после смены CRS. Допустимый итоговый
состав сначала ищется несколькими детерминированными жадными вариантами, при необходимости — компактной MILP,
после чего без перекрытий расширяется до пула. Incumbent по тайм-ауту принимается только после проверки всех
ограничений; невозможный минимум возвращает достижимый максимум объектов с учётом конфликтов тайлов. Статусы
группы и строк сохраняются в Postgres и восстанавливаются после перезапуска.
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
dataset_key)`. Если ключ сменился при миграции, каталог перепривязывает осиротевший шаблон при однозначном
совпадении канонического имени `класс\датасет`; только если датасетного шаблона нет, используется базовый
`(architecture, null)`. Базовый шаблон удалить нельзя;
датасетный шаблон удаляется через `DELETE /api/v1/training-templates/by-id/{template_id}`. Endpoint
`PUT /api/v1/training-templates/by-id/{template_id}/apply-field-to-all` устанавливает одно поле во всех
существующих шаблонах и помечает их `source=manual`.

`inference_templates` устроены аналогично, но содержат только параметры Geoalert-совместимой
постобработки: `postprocess.mask_min_object_pixels`, `postprocess.mask_min_hole_pixels`,
`postprocess.binary_closing_radius`, `postprocess.min_area_m2`, `postprocess.min_hole_area_m2`,
`postprocess.smooth.enabled`, `postprocess.smooth.iterations`, `postprocess.smooth.offset`,
`postprocess.simplify_m` и параметры `postprocess.filter_compact_objects.*`. При ручном и автоматическом
создании псевдоразметки сервис ищет активный шаблон инференса по датасету обученной модели
`(architecture, training_dataset_key)`, а если его нет, использует базовый `(architecture, null)`; выбранный
датасет, папка или загруженный TXT со снимками на шаблон не влияют. Seed-обновление добавляет новые defaults и
заменяет прежние системные значения, но сохраняет явные пользовательские переопределения. Для `Реки\main` и
`smp_segformer_b2` начальный шаблон включает профиль 18: `min_area=10000 м²`, `min_hole_area=5000 м²`,
`FilterCompactObjects(0.25, 3.5)`, `Smooth(iterations=1, offset=0.125)` и `Simplify=1 м`.
Compact-фильтр не использует площадь объекта; граничные фрагменты в полосе одного пикселя сохраняются обоими
фильтрами и теряют служебный тег после завершения цепочки. В базовом, в том числе озёрном, шаблоне compact-фильтр
и Smooth выключены.

`jobs`, `training_results` и `pseudo_markup_results` имеют `source=manual|automation`, `dataset_key` и
`dataset_version`. Для auto rows дополнительно заполнен `automation_rule_id`. Auto jobs нельзя удалить или двигать
через endpoints очереди; они отменяются снятием соответствующей галочки в автоматизации или заменяются при новой
версии конкретного датасета. Глобальное отключение автоматизации отменяет все active auto jobs независимо от правила.
При неудачном обучении `jobs.error` сохраняет хвост журнала до очистки runtime-папки; результат обучения отдаёт
этот текст в поле `error`, чтобы интерфейс показывал причину при наведении на статус.

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
`hard_negative_annotation_file`, если он найден у legacy MLMarkup dataset. Для per-image worker атомарно копирует
все GeoJSON в snapshot задания и записывает `annotations_dir`; дальнейшая публикация MLMarkup не меняет уже
запущенное обучение. Секция `inference` в training `run.yml` не создается: checkpoint, threshold,
batch size и output GeoJSON задаются в отдельном `pseudo_config.yaml` при запуске псевдоразметки. Training-процесс сразу после создания MLflow run пишет
его id в временный файл `mlflow_run_id`; worker читает этот файл и обновляет `training_results.mlflow_run_id`
еще во время `running`. Pause/delete отправляют SIGTERM группе процесса, а `train_pipeline` штатно завершает
MLflow run со статусом `KILLED`.

Срочная поснимочная псевдоразметка редактора использует ту же inference-очередь с признаком `priority=urgent`.
Явно назначенная сеть класса имеет приоритет; без звезды редактор сначала использует последнюю успешную сеть
открытого датасета и только при её отсутствии — последнюю успешную сеть класса. Поэтому готовая полная
псевдоразметка датасета не теряется из-за более нового обучения другого датасета того же класса.
Если выполняется обучение, worker создаёт tokenized `pause.request`; train loop на границе batch переносит модель
и optimizer state в CPU, освобождает CUDA и подтверждает `paused`. После срочного инференса worker снимает запрос,
а тот же процесс и MLflow-run продолжают работу. Обычный running-инференс не прерывается. Перед созданием job
редактор ищет среди всех готовых результатов выбранной сети тот, чей `scenes_file` содержит выбранный TIFF;
dataset key результата не ограничивает повторное использование. При отсутствии покрытия одноэлементный список
сцен проходит через общий builder `pseudo_config.yaml` и выбранный по типу модели runner штатной псевдоразметки.

Перед обработкой очередей worker синхронизирует автоматизацию. Если глобальный выключатель включен и для правила
нет результата или job по текущей `dataset_version`, он ставит auto training job в experiment `MLSystem2 Automation`.
После успешного auto training result с MLflow run id worker ставит auto pseudo-markup job того же датасета.
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
`GET /api/v1/results/classes` выделяет основной датасет и повторяет F1 эффективной сети класса на карточке каждого
его активного датасета; метрика при этом рассчитывается один раз по основной тестовой разметке класса.

При успешном завершении training job worker читает через публичный `mlflow_adapter.api` историю метрики
`val/best_threshold_pixel_f1`, по которой train-модуль сохраняет `checkpoints/best.pt`, и записывает лучший
F1 и эпоху в `training_results.f1_score`/`training_results.epoch`. Pseudo-markup job, созданный от результата
обучения, получает в `jobs.config` `mlflow_run_id`, `checkpoint_artifact_path=checkpoints/best.pt`,
`inference_template_id`, `inference_template_config` и, когда MLflow доступен, полный `checkpoint_uri`.
Для успешного результата обучения `POST /api/v1/results/training/{result_id}/triton-zip` скачивает тот же
`checkpoints/best.pt` из MLflow, собирает временный архив Triton CPU тем же кодом, что endpoint checkpoint-экспорта,
и применяет актуальный effective inference-шаблон по `architecture + class_key`, включая модели, обученные до
появления новых параметров. Нормализованный postprocess-конфиг и его SHA-256 записываются в
`export_metadata.json`. Низкоуровневый checkpoint-экспорт не подбирает классовый шаблон автоматически и
возвращает файл без записи в Postgres, MLflow, S3 или рабочий каталог инференса. `POST /api/v1/results/training/triton-zip`
принимает список успешных результатов и имен моделей, собирает каждую модель тем же кодом и возвращает общий zip с
`models-serving-service/`, `pipelines/`, `metadata/` и корневым `export_metadata.json`.

Для job типа `pseudo-markup` worker берёт TXT выбранного legacy датасета, выбранной папки или загруженного файла;
для per-image он временно формирует TXT из сопоставленных TIFF. Затем пишет в `pseudo_config.yaml`
backend по типу модели и скачивает `checkpoints/best.pt` через публичный
`mlflow_adapter.api.download_run_artifact`. Для Канопуса `pytorch_one_off` загружает checkpoint через
`models.api.load_checkpoint`. Для ортофото `geoalert_workflow_engine` кеширует экспорт конкретного checkpoint в
GPU Triton repository, загружает модель через explicit model-control API и запускает pipeline штатным
`urban.Compose`; после задания модель выгружается. Один ортофото-контур используется для полной и поснимочной
псевдоразметки, AOI и тестового F1. Он не вызывает локальный predictor. Нативный pipeline использует
`SplitRaster`, `Segmentation`, optional `MaskMorphology`, `VectorizeMasks` и vector postprocess bricks; внешние
ЗУ500/ОКС500 сохраняют свои Geoalert pipeline. Оба runner разрешают точные имена сцен через
`dataset_preparing.api.resolve_scene_images` и строят GeoJSON псевдоразметки
в `EPSG:4326`. Для датасета в runner передаются его positive и optional hard-negative GeoJSON, поэтому одинаковые
имена в подпапках выбираются тем же геометрическим алгоритмом, что при обучении; `.SCNxx` не считается расширением
и не приводит к обработке соседних сцен. Полная псевдоразметка сохраняется в `stored_files` только при успешной
обработке всех выбранных снимков; `partial`, failed/missing или несовпадение processed с числом уникальных сцен
переводит job и результат в ошибку, а неполный GeoJSON не публикуется. Результат скачивается через
`/api/v1/files/{file_id}/download`. Канопус не создаёт Triton-артефакты и освобождает CUDA cache; ортофото
переиспользует кеш model repository, но после каждого задания выгружает модель из GPU. Трёхканальный
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
компактные полигоны независимо от площади по isoperimetric quotient и отношению сторон minimum rotated rectangle;
для рек это используется для отсечения озер и прудов. Полигоны в пределах одного пикселя от границы растра и
вырожденные геометрии сохраняются; невалидные не приводят к ошибке фильтрации. Затем речной профиль выполняет
Smooth перед Simplify.
Перед записью итогового скачиваемого GeoJSON `pytorch_one_off` сливает
пересекающиеся и касающиеся полигоны через `unary_union`; per-scene GeoJSON остаются диагностическими файлами
без глобального слияния. Для ортофото слияние и постобработка остаются внутри per-scene Geoalert pipeline, чтобы
не повторять bricks локальным Shapely-кодом. Отчет всегда содержит фактический `inference_backend`; для Geoalert
дополнительно записываются `triton_model` и список выполненных bricks. HTTP API и схема БД при этом не меняются.
