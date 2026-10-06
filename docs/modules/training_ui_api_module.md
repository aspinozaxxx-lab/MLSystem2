# Модуль training_ui_api

## Назначение

`training_ui_api` — FastAPI-сервис сайта MLSystem2. Он хранит UI-данные и управляемый каталог датасетов, синхронизирует опубликованный `/data/MLMarkup`, управляет очередями training/inference и предоставляет Git-backed редактор per-image разметки. Редактор пишет только в отдельный `mlmarkup-editor`, а обучение читает только атомарно опубликованный live-релиз.

Frontend — React + TypeScript + Vite SPA. TypeScript-типы генерируются из OpenAPI командой
`npm run generate:api --prefix frontend`, production static собирается в `frontend/dist`, а сервер Node.js
на проде не нужен.

Компактный интерфейс включается до 700 CSS-пикселей по ширине либо до 1000 × 500 при повороте
телефона. Страницы и HTTP-контракты общие с настольной версией. Меню показывает иконки с доступными
названиями; очередь, результаты и экспорт моделей — карточки; матрицы прокручиваются внутри панели.
На телефоне шаблоны выбираются списком, отдельно для обучения и инференса. Формы сохраняют действия
и шрифт ввода не меньше 16 px; учитываются safe area и динамическая высота, масштабирование разрешено.
Списки снимков можно свернуть. Каждый снимок редактора открывается на телефоне в режиме просмотра
без выбора, правки и рисования; «Править» включает инструменты, «Просмотр» отключает их. Карты
поддерживают перемещение одним пальцем и масштаб двумя, а полный экран — запасной режим внутри
страницы. Правила — `styles/mobile.css`, стили просмотрщика и общий `useCompactLayout` во frontend.

В результатах плашки датасетов внутри класса отсортированы по последнему успешному обучению, свежие первыми;
без успешных обучений — в конце, при равной дате — по названию и ключу. Сортировка использует историю независимо
от выбора основной сети, исторический результат без `trained_at` — его `created_at`. Строка «Обновлено»
под названием класса не показывается.

Редакторы шаблонов обучения и инференса сохраняют только выбранный шаблон, без кнопок «Ко всем».
Во время сохранения поля и действия отключены, кнопка показывает вращающийся индикатор; успешный
ответ подтверждается названием шаблона под кнопкой. Правка, сброс, новая попытка и смена шаблона
убирают прежнее подтверждение. Ошибка использует общее окно ошибок и не показывает успешное сохранение.

Доставка `frontend/deploy.py` сначала публикует зависимости и атомарно заменяет `index.html`.
Хешированные файлы предыдущих сборок в `assets` сохраняются до семи суток с общим целевым пределом
128 МиБ; старейшие неактуальные файлы вытесняются, файлы текущей сборки не удаляются. HTML отдаётся
с `Cache-Control: no-store`. При ошибке загрузки модуля очень старой вкладки только страница просмотра
псевдоразметки может автоматически перечитаться, не чаще раза в минуту; редактор и формы не перезагружаются.

Главная показывает новости из `frontend/src/News.tsx` с датой и адресом `#/news/<slug>`. Новая существенная
возможность описывается большой статьёй о пользе и использовании, с обязательной подходящей картинкой
из `frontend/public/news/` (скриншот либо обозначенная схема, альтернативный текст, без секретов и личных данных).
Мелкие доработки, улучшения и исправления — компактный список `updates`, одна строка на изменение; большая
статья и картинка не требуются. Новости выходят с функционалом, старые публикации и адреса сохраняются.
Внутренние технические изменения без пользовательского эффекта новости не требуют.
Таблица последних событий сохраняется только в результатах.

## Публичный интерфейс

- `create_app() -> Any` - создает FastAPI-приложение.
- `get_openapi_schema() -> dict[str, Any]` - возвращает OpenAPI-схему сервиса.
- `main() -> None` - запускает `uvicorn` для сервиса.
- `worker_main() -> None` - отдельно запускает исполнителей общей очереди и batch-задач тестовых разметок.
- `feedback_main() -> None` — служебная команда `mlsystem2-feedback`: читает обращения через HTTP API или передаёт JSON изменения из стандартного ввода; токен берётся из окружения и не выводится.

## Публичные контракты

`FeedbackKind=remark|improvement|feature`, `FeedbackStatus=waiting|preparing|implementing|implemented`.
`FeedbackCreate`: `submission_id` — UUID отправки; `kind`, `title`, `message` — тип, заголовок и текст;
`page_path`, `page_title` — внутренний hash-адрес без query и название страницы; `credit_name` — добровольная
подпись для благодарности в новости или `null`. `FeedbackInfo` добавляет номер `id`, серверные `author`,
`app_version`, `status`, `revision`, `preparation`, `progress`, `approved_at`, `approved_by`, `approval_note`,
`news_slug`, `commit_sha`, `created_at`, `updated_at`. `FeedbackListResponse`: `items`, `has_more`.
`FeedbackUpdate`: обязательная `expected_revision` и необязательные `status`, `preparation`, `progress`,
`approval_note`, `news_slug`, `commit_sha`. Подготовленное решение нельзя менять вместе с подтверждением
или после него; реализация требует решения и явного подтверждения, завершение — новость и полный SHA.
HTTP: `POST /api/v1/feedback` — отправка из пользовательской сессии с защитой от дублей;
`GET /api/v1/feedback` — общий список с `limit=1..100`, `before_id`, `active`;
`GET /api/v1/feedback/{id}` — карточка; `PATCH /api/v1/feedback/{id}` — изменение с проверкой ревизии.
Список и карточка доступны вошедшим пользователям и исполнителю, изменение — администратору или отдельному
bearer-токену `MLSYSTEM2_FEEDBACK_API_TOKEN`. Пропуск этапов и устаревшая ревизия дают `409`.
Одна таблица `feedback`, миграция `20261001_0028`; хранение и HTTP принадлежат этому модулю,
ежечасная подготовка и подтверждённая разработка выполняются запланированной задачей Codex.

`DatasetEditorImportRequest` содержит `scenes` (от 1 до 100 элементов `DatasetEditorImportSceneRequest`):
`annotation_name` — имя per-image GeoJSON, `geojson` — FeatureCollection. Общий предел — 50 МиБ.
`POST /api/v1/dataset-editor/datasets/{dataset_key}/drafts/import` проверяет весь пакет и однозначное
соответствие серверным TIFF; возвращает `DatasetEditorSceneListResponse`. Создаются только личные черновики,
без файлов разметки, Git-коммита и изменения состава обучения. Отсутствующая опубликованная сцена имеет
`revision="new"`, её черновик — `base_revision="new"`; она видна владельцу в списке и на карте, доступна
изменению, отмене и общей публикации. Уже добавленные снимки и собственные черновики не перезаписываются.
При публикации новые и изменённые сцены сохраняются вместе; появление файла после импорта даёт `409`.
Управляемый датасет получает членство новой сцены и правки исходных классов только при публикации.
Импорт принимает `clip_to_footprint=false`. При выходе объектов за валидные данные возвращается `400`
с `detail` и `code="annotation_outside_footprint"`, количеством объектов и предложением обрезки, без
записи черновиков. Подтверждение повторяет запрос с true: существующая обрезка учитывает отверстия и
nodata, исключает пустые объекты, сохраняет ID/атрибуты и полные зоны разметки. Остальные проверки строгие;
весь пакет повторно проверяется перед атомарным сохранением.
Ошибки роли и типа включают файл, номер объекта, поле, неверное значение и допустимые роли либо
типы текущего датасета. Названия ролей строго канонические, `negative` не заменяется на `hard_negative`.
Frontend показывает ошибку внутри окна импорта без сброса выбранных файлов и текущего снимка;
при `401` сохраняется общий переход к входу. В окне заранее перечисляются роли и multiclass-типы датасета.

`DatasetEditorSceneInfo`, `DatasetEditorDraftSummary` и унаследованный `DatasetEditorDraftInfo` содержат `annotation_zone_count=0`. В GeoJSON редактора поддержана служебная роль `annotation_zone`, optional `_mlsystem2_class` задаёт исходный класс управляемого датасета. Отсутствие класса означает все источники. Зоны проходят штатные черновики, публикацию, копирование и пересборку, отдельно отображаются и считаются; новые endpoints и таблицы не вводятся. Контуры зон сохраняются целиком, включая части за границами TIFF/valid footprint; проверки типа, валидности геометрии и CRS остаются обязательными. Ограничение зон снимком выполняет подготовка обучения. `dataset_preparing.api.annotation_regions` ограничивает создаваемые эталонные тайлы размеченной территорией; обычная псевдоразметка остаётся поснимочной.

`ConfigSchema.pipeline_descriptions` содержит подробные описания профилей.
`ConfigSchema.pipeline_defaults` содержит серверные наборы параметров при переключении конвейера;
по умолчанию это пустой словарь. `JobSummary`, `JobDetail` и `TrainingResultInfo.pipeline_variant`
принимают `legacy|next_gen|next_gen2`. Новая таблица БД не требуется.

- `TrainingUIAPIError` - ошибка сервиса.
- `TemplateSource`, `JobType`, `JobSource`, `JobStatus`, `ResultStatus`, `StoredFileKind` - enum-значения API.
- `AppLink`, `AppLinksResponse` - ссылки Grafana, MLflow и авторизованного браузера подготовленных снимков.
- `BootstrapInfo` - стартовый DTO для React frontend: links, datasets, image folders, classes, models и оба набора templates одним ответом.
- `MLflowExperimentInfo`, `MLflowExperimentCreate` - experiments MLflow.
- `ImageryType`, `ImageryTypeInfo`, `DatasetFormat`, `DatasetInfo`, `DatasetObjectType`, `DatasetListResponse`, `ClassInfo`, `ClassListResponse` - управляемый каталог; `ClassInfo.technical_name` задаёт редактируемую каноническую основу имени модели и semantic slug типа, а `DatasetInfo` содержит `format=legacy|per_image|per_image_multiclass`, `task`, `object_types`, `combined`, source status/changes, class counts, состояние фоновой материализации, совместимый исторический `model_name_stem`, legacy-файлы либо `annotations_dir` готовой текущей версии.
- `DatasetCatalogInfo`, `DatasetSourceInfo`, `DatasetClassCreate`, `DatasetClassUpdate`, `DatasetPrimaryDatasetUpdate`, `ManagedDatasetCreate`, `ManagedDatasetUpdate` - чтение и изменение активных классов и датасетов; мягко удалённые строки остаются в Postgres, но в каталог не входят.
- `ImageFolderInfo`, `ImageFolderListResponse` - папки подготовленных снимков из `MLSYSTEM2_IMAGES_ROOT` с количеством TIFF.
- `ModelInfo`, `ModelListResponse` - публичные модели из `models`.
- `ConfigField`, `ConfigSchema`, `TrainingTemplate`, `TrainingTemplateListResponse`, `TrainingTemplateCreate`, `TrainingTemplateUpdate`, `TrainingTemplateApplyField`, `InferenceTemplate`, `InferenceTemplateListResponse`, `InferenceTemplateCreate`, `InferenceTemplateUpdate`, `InferenceTemplateApplyField` - шаблоны обучения и инференса; `ConfigField` содержит `tooltip`, допустимые границы и optional `recommended_range` для UI-подсказок.
- Встроенный датасетный inference-шаблон хранит человекочитаемую цель, но при инициализации привязывается к ключу действующей строки каталога по паре `класс/имя`; это сохраняет специальные настройки после смены ключа или миграции legacy-датасета.
- `StoredFileInfo`, `CustomDatasetInfo` - загруженные файлы и custom datasets.
- `TrainingJobCreate`, `QueueEnabledUpdate`, `QueueControlInfo`, `QueueCountInfo`, `JobSummary`, `QueueSnapshot`, `JobDetail` - задания и очереди; Job DTO вычисляет `pipeline_variant` и `validation_fold` из JSON config без миграции БД. `JobDetail.mlflow_run_url` возвращает сохранённую ссылку связанного результата обучения либо `null`, пока запуск не создан.
- `PseudolabelJobCreate`, `PseudolabelClassInfo`, `PseudolabelClassListResponse`, `PseudolabelJobInfo`, `PseudolabelErrorInfo` - AOI, доступная зафиксированная модель, состояние и структурированная ошибка QGIS-контракта.
- `AutomationEnabledUpdate`, `AutomationRuleUpdate`, `AutomationRuleInfo`, `AutomationSnapshot` - глобальный выключатель и матрица автоматизации `датасет × модель`.
- `TrainingResultInfo`, `TrainingResultTestF1Info`, `PrimaryTestSampleInfo`, `PseudoMarkupResultInfo`, `DatasetResultsResponse`, `ResultClassInfo`, `ResultDatasetInfo`, `ResultClassListResponse`, `ResultChangeInfo`, `ResultChangesResponse` - результаты обучения, включая вычисляемые `pipeline_variant`/`validation_fold`, task/class schema, метрики, test F1 и псевдоразметку.
- `TestF1ScoreInfo`: `precision`, `recall`, `f1`, `true_positive`, `false_positive`, `false_negative` — фактическая оценка одного снимка. `TestF1SceneInfo` расширяет `PseudoMarkupSceneInfo`: `layers_url`, `raster_available`, `sample_name`, `sample_revision`, `target_class_id`, `class_schema`, `pixel`, `objects`, `metrics` — сохранённые слои, ревизия и поснимочные оценки. `TestF1ViewInfo`: `training_result_id`, `model_name`, `training_dataset_name`, `status=ready|missing|queued|running|error|unavailable`, `metric`, `scenes`, `warnings` — состояние и состав просмотра.
- `GET|POST /api/v1/results/training/{result_id}/test-f1/view` — авторизованный просмотр конкретной сети; GET без запуска ML, POST идемпотентно готовит отсутствующие маски через полный штатный F1-job с блокировкой строки сети. `GET .../test-f1/layers/{scene_id}?class_id=<id>` — GeoJSON в EPSG:3857 с `test_f1_layer=reference|predicted|tp|fp|fn`; свой класс managed-выборки обязателен, остальные типы проверяются по сохранённой схеме. `GET .../test-f1/raster/{scene_id}?v=<revision>` и `GET .../test-f1/footprint/{scene_id}?v=<revision>` — TIFF через общий Range и контур. Все маршруты требуют сессию, принадлежность расчёту сети и путь внутри test-samples; изменение TIFF возвращает 412.
- Тестовый F1 в результатах открывает `#/test-f1/<training_result_id>`. Карта общая с псевдоразметкой: поснимочный выбор, независимые слои эталона и прогноза, пиксельные TP/FP/FN зелёным/красным/жёлтым, отдельный объектовый F1 и переключение смыслового типа. Runner сохраняет именно использованные маски и поснимочные счётчики; worker переносит их в `stored_files_root/test-f1/<job_id>` до удаления scratch и записывает непрозрачные ссылки в `metrics.viewer_jobs`. Частичный managed-пересчёт сохраняет ссылки остальных классов. Старая ревизия показывается с предупреждением; исторический расчёт без масок готовится явной кнопкой, удалённый TIFF допускает слои без подложки. Список на телефоне заменяется селектором, карта поддерживает жесты и полный экран.
- `PseudoMarkupViewInfo` — ID результата/сети, названия сети, источника и обучающего датасета, дата, ссылка GeoJSON, счётчики объектов/снимков, список `scenes` и предупреждения. `PseudoMarkupSceneInfo` — ID снимка, относительное имя, авторизованные `raster_url` и `footprint_url`, bounds в EPSG:3857, признаки `has_alpha`, `has_nir` (по умолчанию false) и nodata. Просмотр предлагает RGB/NRG/NGB для снимков с NIR; RGB/RGBA сохраняют RGB и прозрачность. Сочетания каналов общие с редактором. Колесо работает при наведении без фокуса, кнопок «+» и «−» нет. Frontend сопоставляет bounds снимков с текущим кадром карты и подсвечивает соответствующие строки списка, включая частичное пересечение и полноэкранный режим; счётчик не зависит от поиска. Глаз строки независимо скрывает её снимок; подсветка и счётчик учитывают только включённые снимки. Наведение или фокус подсвечивает фактический контур данных без nodata, нажатие на имя приближает снимок и закрепляет контур; повторное нажатие сразу снимает выделение, сохраняя охват карты. Готовый кадр сбрасывается при изменении видимости. Временные ошибки загрузки повторяются ограниченно, отмена не создаёт предупреждения; невосстановленные ошибки можно повторить кнопкой.
- `GET /api/v1/results/pseudo-markup/{result_id}/footprint/{scene_id}` — GeoJSON Feature контура valid-data mask в EPSG:3857, лениво по наведению/выбору. Те же авторизация, состав результата и необязательная ревизия `v` (412 при изменении). Общий с редактором расчёт маски до 4096 пикселей по длинной стороне и кэш 64 контуров; клиент хранит 32 контура.
- `GET /api/v1/results/pseudo-markup/{result_id}/view` и `GET /api/v1/results/pseudo-markup/{result_id}/raster/{scene_id}` — авторизованный просмотр готового результата и его TIFF через общий HTTP Range. Только сохранённый состав обработанных снимков в `pseudo_processed_images`, отчёте или списке сцен; fallback текущего датасета требует точного совпадения версии. Чужие TIFF и выход за prepared_images запрещены. Страница `#/pseudo-markup/<id>` открывается иконкой рядом со скачиванием, без меню и запуска инференса. Необязательный `v` растрового маршрута проверяет файловую ревизию (412 при изменении), ETag совпадает с ней. `private, no-store` оставляет хранение фрагментов ограниченному кэшу frontend (IndexedDB, сутки, до 512 МиБ, вытеснение по давности использования, разделение пользователей и очистка при выходе).
- `PseudoMarkupViewInfo.class_name` — название класса сети для сравнения результатов разных классов; историческое значение может отсутствовать.
- `PseudoMarkupComparisonRequest`: уникальные `result_ids` (1–12 активных разметок), `scene_revisions` — ожидаемые файловые ревизии TIFF по ID снимка; nullable `scene_id` ограничивает счётчики одним общим снимком, nullable `viewport` ограничивает слои видимой областью. `PseudoMarkupComparisonViewport`: упорядоченные конечные `bounds` EPSG:3857, `width`, `height` карты (1–2048 экранных пикселей). `PseudoMarkupPixelComparison`: `intersection`, `only_first`, `only_second` — неотрицательные точные пиксельные счётчики в порядке двух переданных ID. `PseudoMarkupComparisonCounts`: `result_ids`, `scenes` (счётчики запрошенных общих TIFF по ID), `total`, `warnings`. `PseudoMarkupComparisonLayers`: `scene_id`, `available_result_ids`, `geojson` в EPSG:3857 с `comparison_kind=layer|intersection|difference` и `comparison_result_id`, nullable `counts` только для общей пары без `viewport`. Обзорные слои уточняются при изменении видимой области и масштаба; точные счётчики считаются отдельно на нативной сетке.
- `POST /api/v1/results/pseudo-markup/compare/counts` — счётчики ровно двух активных готовых результатов по общим исходным TIFF и их сумма; `scene_id` ограничивает расчёт одним снимком. `POST .../compare/{scene_id}/layers` — слои активных разметок данного снимка; с `viewport` только видимая область в масштабе карты. Отсутствие снимка у части результатов возвращает только существующие слои без парных счётчиков. Оба маршрута авторизованы, проверяют состав каждой разметки, ограничение prepared_images и ревизии (412); не ставят ML-заданий и не меняют БД. Точные счётчики обрабатывают TIFF окнами до 512×512, полигоны объединяются в foreground на нативной сетке без nodata/alpha; маски обзора ограничены размером карты. Для ≥3 слоёв пересечение означает покрытие ≥2 слоёв, различие — покрытие только одного. Кэши ограничены и учитывают файловые ревизии; готовность и доступ проверяются перед чтением кэша. Разрыв соединения отменяет расчёт без сохранения частичных значений.
- Frontend хранит список до 12 выбранных результатов в авторизованной сессии независимо от открытого класса/датасета. Адрес `#/pseudo-markup/compare/<id1,id2,...>` использует общий `PseudoMap`: слои, легенда и подсветки раздельно управляются, снимок выбирается отдельно, парные счётчики видны в списке и компактной шапке; прежние каналы, телефон и полный экран сохранены. Снимки сопоставляются по ID исходного TIFF, без склейки совпадающих имён или географических областей; сравнение не является F1.
- `TrainingResultExportItem`, `TrainingResultBatchExportRequest` - JSON-запрос массового экспорта выбранных успешных training results.
- `MarkupExportRequest`, `MarkupExportTileInfo`, `MarkupExportInfo` - запрос и описание временного набора тестовой разметки с тайлами, превью, сводкой цель/факт, режимом исключения граничных объектов и сроком хранения.
- `TestSampleCreate`, `TestSampleUpdate`, `TestSampleTileUpdate`, `TestSampleOptimizeRequest`, `TestSamplePrimaryUpdate`, `TestSampleEvaluationPreviewRequest`, `TestSampleDownloadRequest`, `TestSampleBulkDownloadRequest` - создание и атомарное сохранение постоянной разметки, совместимые точечные изменения, ограничения оптимизации, запросы оценки, одиночного и группового скачивания.
- `TestSampleAnnotationsMerge` — ожидаемая `expected_revision`, название новой редакции `name` и список `tiles`; `TestSampleTileMerge` — `tile_index` и непересекающиеся группы существующих feature ID `groups`, минимум два ID в группе. `TestSampleSummary.content_revision` — текущая ревизия состава для optimistic lock, совместимый default `1`.
- `TestSampleMetric`, `TestSampleEvaluationInfo`, `TestSampleSummary`, `TestSampleDatasetGroup`, `TestSampleClassGroup`, `TestSampleCatalogResponse`, `TestSampleTileInfo`, `TestSampleDetail`, `TestSampleDraftPreview` - binary scalar и multiclass per-class/macro/micro/foreground метрики, task/schema/counts, каталог, редакторское описание с URL миниатюры и полноразмерного PNG и незаписываемый результат чернового расчёта.
- `TestSampleBatchItemCreate`, `TestSampleBatchCreate`, `TestSampleBatchItemInfo`, `TestSampleBatchInfo` - запрос одного задания и прогресс отдельной очереди нарезки; `items` нового запроса содержит ровно один датасет. `tile_size` выбирается из `512`, `768`, `1024`, `1536`, `2048`, `2560`, `3072`, `3584`; `min_image_count`/`image_count` задают диапазон итоговых тайлов, `min_object_count` — минимум объектов, `use_optimization` — подбор по прогнозу своей сети либо случайный допустимый состав без сети, `exclude_boundary_objects` — исключение граничных объектов для объектовой F1. Ответ содержит `queue_position`, `cancel_requested`, статус `queued|running|ok|partial|error|cancelled`, результат и ошибку; исторические группы остаются читаемыми.
- `TestSampleCreationSettings` - сохраняемые отдельно для датасета `tile_size`, `min_image_count`, `image_count`, `min_object_count`, `use_optimization`, `exclude_boundary_objects` с проверкой положительных квот и диапазона. `TestSampleBatchDatasetOption.creation_settings` возвращает их вместе с прежними данными датасета, сети и псевдоразметки; `TestSampleBatchClassOption` и `TestSampleBatchOptionsResponse` группируют варианты по классу.
- `TestSampleBatchMove` - `direction=up|down` для перемещения ожидающего задания внутри отдельной очереди создания разметок.
- `DatasetEditorDatasetInfo`, `DatasetEditorObjectType`, `DatasetEditorDatasetListResponse`, `DatasetEditorSceneInfo`, `DatasetEditorSceneListResponse`, `DatasetEditorSceneDetail`, `DatasetEditorPseudoMarkupInfo`, `DatasetEditorDraftSummary`, `DatasetEditorDraftInfo`, `DatasetEditorUserDraftInfo`, `DatasetEditorUserDraftListResponse` - каталог per-image датасетов и сцен с task/schema, class counts, source status, revision, пользовательским серверным черновиком, агрегированным списком черновиков пользователя, `valid_data_footprint` и состоянием псевдоразметки сцены.
- `DatasetEditorRebuildPreview`, `DatasetEditorRebuildRequest`, `DatasetEditorRebuildResult`, `DatasetEditorRebuildChange` - preview token, source/local changes, конфликты и атомарная пересборка `merge|replace`.
- `DatasetEditorRasterFolderInfo`, `DatasetEditorRasterInfo`, `DatasetEditorRasterBrowserResponse` - прямые папки и TIFF из разрешённого server-side каталога.
- `DatasetEditorAddScenesRequest`, `DatasetEditorSaveSceneRequest`, `DatasetEditorSaveDraftRequest`, `DatasetEditorDiscardDraftsResult`, `DatasetEditorPublishSceneRequest`, `DatasetEditorPublishRequest`, `DatasetEditorDeleteSceneRequest`, `DatasetEditorCopyRequest`, `DatasetEditorMutationResult`, `DatasetEditorCopyResult`, `DatasetEditorPublicationInfo` - серверные черновики, одиночные и атомарные batch optimistic-lock мутации, создание независимой именованной копии и статус публикации commit SHA.
- Все HTTP-ответы содержат `Server-Timing` и `X-Process-Time-Ms`; запросы дольше одной секунды журналируются с методом, путём, статусом и временем без query-параметров.
- `GET /api/v1/bootstrap` - агрегированный стартовый endpoint для frontend; старые catalog/template endpoints остаются рабочими.
- `GET /api/v1/dataset-catalog` и `POST /api/v1/dataset-catalog/sync` - иерархия редактора и явная идемпотентная синхронизация с MLMarkup.
- `POST|PATCH /api/v1/dataset-classes`, `PUT /api/v1/dataset-classes/{class_key}/primary-dataset`, `POST|PATCH /api/v1/managed-datasets` - создание и редактирование каталога. PATCH управляемого виртуального датасета меняет его имя и composition `sources/priority/color` с сохранением ключа; изменение composition при наличии пользовательских черновиков отклоняется. Назначение занятого обычного источника переносит существующий датасет с сохранением его ключа, а смена источника безопасно обменивает источники и увеличивает ревизии обеих сущностей.
- `DELETE /api/v1/dataset-editor/datasets/{dataset_key}` - удалить папку датасета Git-коммитом, мягко архивировать его строку в Postgres и снять назначение основным; задания, результаты и MLflow не удаляются, при активном задании операция отклоняется.
- `POST /api/v1/dataset-editor/datasets/{dataset_key}/copy` - создать под новым именем копию опубликованного датасета с новым ключом. Обычный набор получает отдельную Git-папку; управляемый — отдельную конфигурацию с теми же источниками, приоритетами, цветами и явно добавленными снимками. Черновики, шаблоны, задания и результаты не копируются.
- `GET /api/v1/dataset-editor/datasets` и `GET /api/v1/dataset-editor/datasets/{dataset_key}/scenes[/{annotation_name}]` - список редактируемых датасетов, сцены и GeoJSON.
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/download` - скачать ZIP опубликованной разметки: папки класса и датасета, поснимочные GeoJSON, имеющиеся companion-footprint и `.mlsystem2-dataset.json`. TIFF и пользовательские черновики не включаются. Управляемый набор экспортируется из готового актуального кэша; до его готовности возвращается ошибка без сборки в HTTP-процессе.
- `GET /api/v1/dataset-editor/drafts` - агрегированный по датасетам список сохранённых черновиков только вошедшего пользователя для навигации из редактора классов.
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/rasters` и `GET /api/v1/dataset-editor/datasets/{dataset_key}/raster/{image_path}` - серверный выбор снимков и авторизованный TIFF с HTTP Range.
- `GET|POST /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}/pseudo-markup` - получить готовый фрагмент текущей основной сети либо идемпотентно поставить срочный поснимочный инференс.
- `GET /api/v1/dataset-editor/pseudo-markup/{job_id}` - лёгкий polling поснимочного задания без Git-синхронизации и разрешения полного каталога.
- Frontend держит псевдоразметку отдельным слоем и в специальном режиме позволяет выбрать её объекты, скопировать только геометрию в пользовательский черновик с назначением semantic-типа или `hard_negative`, отменить добавление общей историей и дождаться обычного автосохранения без публикации. Для управляемого датасета hard negative назначается одному, нескольким либо всем исходным датасетам.
- В многоклассовом редакторе легенда со счётчиками работает как переключатель видимости типов в разметке и псевдоразметке. Наведение и клавиатурный фокус подсвечивают только видимые объекты на включённых слоях. Фильтр действует при переключении снимков до смены датасета, не меняет данные или счётчики и исключает скрытые объекты из выбора и привязки вершин.
- `PUT|DELETE /api/v1/dataset-editor/datasets/{dataset_key}/drafts/{annotation_name}` и `DELETE /api/v1/dataset-editor/datasets/{dataset_key}/drafts` - автоматически сохранить промежуточный пользовательский GeoJSON или пометку удаления без публикации либо удалить один/все свои черновики; при PUT полигональная геометрия восстанавливается и пересекается с valid-data footprint TIFF, поэтому частичный выход подрезается, а пустой результат удаляется без отказа всего черновика.
- При сохранении черновика и публикации числовые пользовательские атрибуты сравниваются по значению, включая вложенные массивы и словари: преобразование браузером `1.0` в `1` не считается редактированием. Настоящие изменения атрибутов по-прежнему запрещены; строки и логические значения отличаются от чисел.
- Проверка попадания объекта в footprint учитывает машинную точность мировых координат и длину границы: численная погрешность после пересечения допускается, реальный выход за контур — нет.
- `POST /api/v1/dataset-editor/datasets/{dataset_key}/drafts/publish` - атомарно опубликовать все сохранённые изменения и удаления пользователя одним Git-коммитом.
- `POST|PUT /api/v1/dataset-editor/datasets/{dataset_key}/scenes`, `PUT|DELETE /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}` - добавить отсутствующие TIFF/папку, атомарно опубликовать несколько GeoJSON, совместимо сохранить один GeoJSON или создать отменяемую пометку удаления сцены.
- `GET /api/v1/dataset-editor/publication/{commit}` - `publishing|published` по ancestry commit текущего live-релиза.
- `POST /api/v1/dataset-editor/datasets/{dataset_key}/rebuild/preview` и `POST /api/v1/dataset-editor/datasets/{dataset_key}/rebuild` - preview и атомарная merge/replace-пересборка combined dataset; изменение source или target после preview возвращает `409`.
- `GET /api/v1/files/{file_id}/download-by-type` - ZIP канонической multiclass-псевдоразметки с одним GeoJSON на object type.
- `POST /api/v1/model-export/triton-zip` - multipart endpoint для сборки zip-архива модели под
  `models-serving-service` и Triton CPU; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/{result_id}/triton-zip` - multipart endpoint для сборки такого же zip-архива
  из `checkpoints/best.pt` либо внешнего ZIP успешного результата в MLflow; для нативного checkpoint используется
  актуальный effective inference-шаблон по `architecture + class_key`, endpoint возвращает файл и не создаёт записей в БД.
- `POST /api/v1/results/training/triton-zip` - JSON endpoint для сборки общего zip-архива нескольких успешных
  нативных и импортированных результатов; каждая модель собирается тем же кодом, что одиночный экспорт результата, endpoint
  возвращает файл и не создает записей в БД. В metadata каждой нативной модели сохраняются нормализованный
  postprocess-конфиг и его SHA-256; низкоуровневый экспорт произвольного `.pt` не выбирает классовый шаблон.
  Кеш Geoalert допускает `null` вместо порога у импортированной ОКС500, возвращающей готовые классы;
  повторное использование модели и восстановление YAML сохраняют её исходный pipeline.
  Compose загружает фиксированные обязательные блоки с `enable_blocks={}`.
  Нативная загрузка и экспорт ОКС500 используют общую адаптацию пустого encoder skip к устройству входа;
  исходный ZIP и веса не изменяются. Версия кеша этой внешней модели — 5.
  В её YAML классы 1/2 дают markers/contour крыши, класс 3 — тень, класс 4 — стены.
  После экспорта фактическое число semantic channels фиксируется в описании выхода ONNX;
  strict model config Triton принимает HF-граф с изначально символической размерностью каналов.
  Бинарный `next_gen2` экспортирует FP32 `probabilities` и pipeline с `SlidingWindowSegmentation`
  (окно сети, шаг в половину окна, Gaussian sigma 0.25), затем строгий порог 0.9 либо явный override.
  SplitRaster сохраняет исходные пиксели под nodata, VectorizeMasks строит контуры по границам пикселей.
  Для серверного Compose runner расширения регистрируются из `_geoalert_notebook.py` перед загрузкой YAML:
  полные окна через Triton adapter, Gaussian-объединение с полосовым буфером и SplitRaster без изменения исходных пикселей.
  В заданиях сохраняется вариант конвейера: `next_gen2` применяет только явные параметры постобработки
  шаблона датасета, без автоматических профилей по числу снимков. Шаблон `segformer_b0` входит в начальные данные.
  Внешний потребитель ZIP также должен поддерживать эти брики. Профиль требует context=0 и исходного разрешения; metadata ZIP фиксирует
  `output_kind`, `inference_merge`, `inference_stride` и `requires_inference_brick`.
  Остальные профили сохраняют прежний UINT8 ABI маски.
- Одиночная и групповая формы экспорта предлагают редактируемое имя `<class.technical_name>_kanopus` либо
  исторически совместимое `<class.technical_name>_orto`; `model_name_stem` остаётся только fallback старого API.
- `POST /api/v1/scene-list-export` - multipart endpoint с `imagery_type=kanopus|ortho`, optional
  `include_footprints` и GeoJSON; рекурсивно находит TIFF с полигональными объектами. Без флага возвращает
  совместимый UTF-8 TXT относительных путей внутри корня типа снимков без расширений, с флагом — ZIP из этого
  TXT и GeoJSON футпринтов выбранных снимков в WGS84. Пустой список допустим, а повтор относительного пути без
  расширения отклоняется как неоднозначность TXT.
- `POST /api/v1/markup-export` - синхронно формирует временный набор тестовой разметки для датасета MLMarkup.
- `GET /api/v1/markup-export/{export_id}/tiles/{tile_index}/preview` - возвращает PNG-превью тайла с отдельными контурами GeoJSON-объектов.
- `GET /api/v1/markup-export/{export_id}/download` - возвращает плоский ZIP сформированного набора.
- `GET /api/v1/test-samples` и `POST /api/v1/test-samples` - иерархический каталог и создание постоянной тестовой разметки.
- `GET|PATCH|DELETE /api/v1/test-samples/{sample_id}` - просмотр, атомарное сохранение имени, основного статуса и полного состава либо удаление разметки.
- `POST /api/v1/test-samples/{sample_id}/merge-annotations` — идемпотентно создать неосновную редакцию бинарной выборки с объединением соседних объектов. Проверяются ревизия (`409`), ID, общие границы, пользовательские свойства и побитовое равенство семантической маски; TIFF копируются с проверкой SHA-256, прежняя выборка и MLMarkup не меняются. Превью пересоздаются, кэши не копируются, прямая F1 ставится через штатную очередь. Файловая публикация откатывается при ошибке транзакции. Журнал `annotation_revision.json` хранит автора, исходную ревизию, группы и хеши. Повтор идентичного запроса возвращает прежний результат.
- `PATCH /api/v1/test-samples/{sample_id}/tiles/{tile_index}` - включает или выключает тайл.
- `POST /api/v1/test-samples/reconcile` - идемпотентно ставит в inference-очередь отсутствующие и устаревшие прямые оценки всех сохранённых разметок текущими основными сетями классов.
- `POST /api/v1/test-samples/{sample_id}/evaluate` - принудительно ставит прямой пересчёт pixel/object F1 текущей основной сетью класса.
- `POST /api/v1/test-samples/{sample_id}/pseudo-markup` - идемпотентно ставит штатную полную псевдоразметку исходного датасета сетью, зафиксированной при создании набора, для предварительной оценки и оптимизации.
- `POST /api/v1/test-samples/{sample_id}/optimize` - подбирает состав из всех тайлов по основной метрике класса; request-поле старого клиента принимается, но не меняет выбор метрики.
- `POST /api/v1/test-samples/{sample_id}/evaluate-preview` и `POST /api/v1/test-samples/{sample_id}/optimize-preview` - рассчитывают F1 или оптимальный состав черновика без записи в БД.
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/thumbnail` - возвращает постоянную JPEG-миниатюру до `384×384`; для старых наборов создаёт её один раз из полного PNG.
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview` и `GET /api/v1/test-samples/{sample_id}/download` - полноразмерный PNG с отдельными контурами объектов и ZIP сохранённых включённых тайлов; GeoJSON копируется без изменения, а помеченные JPEG строят контуры по его instance-ID.
- `POST /api/v1/test-samples/{sample_id}/download` - ZIP явно выбранных тайлов текущего черновика без изменения разметки в БД; флаг `include_previews` оставляет полный состав либо только TIFF и GeoJSON.
- `POST /api/v1/test-samples/download` - несжатый ZIP явно выбранных сохранённых разметок, не более одной на класс; до восьми разметок готовятся параллельно, каждая в папке `<класс>_<исходный датасет>`.
- `GET /api/v1/test-sample-batches/options` - сгруппированные по классу явные варианты `датасет → его сеть → точная полная псевдоразметка`; сеть другого датасета класса не подставляется.
- `POST /api/v1/test-sample-batches/options/{dataset_key}/pseudo-markup` - идемпотентно ставит полную псевдоразметку выбранного датасета его выбранной сетью.
- `POST /api/v1/test-sample-batches` - добавить одно задание создания разметки готового датасета старого или поснимочного формата. С оптимизацией фиксируются идентификаторы сети и полной псевдоразметки; без неё они не требуются.
- `GET /api/v1/test-sample-batches` - отдельная очередь нарезки: выполняемые задания первыми, ожидающие по приоритету, затем до десяти последних завершённых. `GET /api/v1/test-sample-batches/latest` и `GET /api/v1/test-sample-batches/{batch_id}` - последний запуск и состояние указанного задания, включая исторические группы.
- `PUT /api/v1/test-sample-batches/options/{dataset_key}/settings` - сохранить `TestSampleCreationSettings` выбранного датасета без создания задания и без смены версии его разметки.
- `TestSampleCreationSettings`, `TestSampleBatchItemCreate`, `TestSampleBatchItemInfo` и `TestSampleCreate` содержат `min_object_area_m2` (конечное неотрицательное число, по умолчанию 0). Порог сохраняется в датасете и снимке параметров задания. Исходные positive-объекты меньшей площади на эллипсоиде WGS84 исключаются до подбора тайлов, включая случайный режим; отверстия и части MultiPolygon учитываются. Площадь 0 сохраняет прежний состав, зоны разметки не фильтруются. Миграция `20261006_0030` добавляет порог к элементу очереди, историческим заданиям назначает 0.
- `POST /api/v1/test-sample-batches/{batch_id}/move` - переместить ожидающее задание в направлении `TestSampleBatchMove.direction`; `DELETE /api/v1/test-sample-batches/{batch_id}` - удалить ожидающее задание. Для выполняемого оба действия запрещены.
- `POST /api/v1/test-sample-batches/{batch_id}/cancel` - запросить отмену выполняемого задания; следующее ждёт завершения очистки. Все изменения очереди требуют сессию пользователя.
- `POST /api/v1/training-jobs` - поставить ручное обучение; при `run_inference_after_training=true` успешное завершение идемпотентно ставит штатную полную псевдоразметку того же датасета полученной сетью.
- `GET /api/v1/results/training/{result_id}/continue` — `TrainingContinuationOptions`: начальные значения
  `additional_epochs`, `additional_time_sec`, `early_stopping_patience` из исходного задания (при отсутствии
  лимита времени — 3600 секунд), `last_checkpoint_available` после проверки артефактов завершённого MLflow run.
- `POST /api/v1/results/training/{result_id}/continue` — новый ручной training job от выбранного checkpoint.
  `TrainingContinuationCreate` принимает три строго положительных целых лимита, `checkpoint=best|last`
  (по умолчанию best) и UUID `request_id` для идемпотентной отправки. Иные параметры не принимаются.
  Оба маршрута требуют пользовательскую сессию. Источник — успешный нативный result с завершённым training job
  и сохранёнными параметрами; импортированные модели и незавершённые работы не подходят. Worker скачивает
  `best.pt`/`final.pt` из исходного завершённого запуска и передаёт через существующий `train.initial_checkpoint_uri`.
  При недоступности артефакта новый этап получает ошибку, обучение с нуля не начинается.
  Приватный `_training_continuation.py` копирует конфигурацию без применения текущего шаблона, сохраняет
  исходные версию и снимок разметки, восстанавливает их из MLflow при утрате временной папки. При наличии
  используется полный `config/resolved_train_config.json`; исторический резерв — исходный YAML запуска.
  Все новые задания сохраняют полный валидированный снимок обучения `SystemSettings` без блока `inference` в `run.yml` до старта процесса.
  Оптимизатор/scheduler и счётчики создаются заново; это дополнительный этап от весов, а не восстановление
  процесса после паузы. Диалог предлагает `run_inference_after_training` и `secondary_priority` с исходными
  значениями: обе опции можно изменить для нового этапа. Эти поля в `TrainingContinuationCreate` допускают
  `null`/отсутствие для наследования при запросе старого клиента; `TrainingContinuationOptions` возвращает
  фактические булевы значения. Изменение опции при повторе одного `request_id` отклоняется, как изменение
  лимитов или чекпойнта. Используются штатные создание псевдоразметки и приоритет очереди; новые механизмы
  доставки результата не вводятся. Ссылки на уже поставленные задания
  и запрос прежней остановки очищаются. Исходный result, его звезда и псевдоразметки не меняются.
  `TrainingResultInfo.can_continue_training` определяет наличие кнопки без сетевого обхода артефактов на
  каждой строке; `continued_from_result_id`, `continued_from_epoch` (для best) и `continued_from_checkpoint`
  обозначают происхождение нового этапа. Доступность файла уточняется при открытии диалога и отправке.
  UI группирует цепочки и ответвления продолжений с исходной сетью; псевдоразметки остаются под своим
  этапом. `PseudoMarkupResultInfo.checkpoint_epoch` и `PseudoMarkupViewInfo.checkpoint_epoch` показывают
  эпоху лучших весов исходного результата псевдоразметки (либо `null`, если эпоха неизвестна).
  Прежние MLflow run и checkpoint сохраняются; номера эпох относятся к своему этапу.
- Запуск строится как «архитектура → конвейер → предобученные веса». SegFormer B0/B1/B2/B3 поддерживают
  `legacy` и `next_gen2` с независимой галочкой `train.pretrained`; остальные сети — `legacy`.
  HF исключён из каталога и новых ручных/автоматических обучений. Исторические результаты и инференс сохранены.
  Профили и batch для архитектуры передаются через `ConfigSchema.pipeline_defaults`.
  `next_gen` отклоняется в новых заданиях и изменениях шаблонов; прежние шаблоны обновляются на next-gen2.
  История и уже созданные задания не меняются. Бинарный профиль next-gen2 поддерживает RGB и RGB+NIR.
- `POST /api/v1/jobs/{job_id}/stop-and-save-best` - кооперативно остановить выполняющееся ручное обучение и выдать `best.pt` максимальной валидационной F1 как успешный результат; до появления первого лучшего чекпойнта операция недоступна.
- `GET /api/v1/queues/count` - лёгкое число active jobs (`queued|running|paused`) для счётчика меню; `GET /api/v1/queues` остаётся полным снимком очереди.
- `PUT /api/v1/test-samples/{sample_id}/primary` - совместимо назначает, заменяет или снимает единственную основную разметку класса.
- `GET /api/v1/results/datasets/{dataset_key}`, `POST /api/v1/results/datasets/{dataset_key}/pseudo-markup` и `POST /api/v1/results/datasets/{dataset_key}/test-f1` - результаты датасета, ручная псевдоразметка и ручная постановка оценок всех успешных сетей этого датасета в inference-очередь.
- `POST|DELETE /api/v1/results/training/{result_id}/primary` - явное назначение успешной сети основной для её класса или снятие отметки; без назначения расчёты используют последнюю успешную сеть без звезды. Если эффективная сеть не изменилась, операция не создаёт заданий; при реальной смене обновляются только зависящие от неё сохранённые контрольные метрики, но не независимая матрица остальных сетей.
- `GET /api/v1/pseudolabel/classes`, `POST /api/v1/pseudolabel/jobs`, `GET|DELETE /api/v1/pseudolabel/jobs/{job_id}` и `GET /api/v1/pseudolabel/jobs/{job_id}/result` - серверное распознавание AOI без передачи клиентских растров; полный контракт описан в `docs/pseudolabel_api.md`.

Сохранённая контрольная метрика каждой тестовой разметки получается прямым инференсом эффективной сети класса:
явно назначенной либо, при отсутствии назначения, последней успешной. Неявный выбор не сохраняется и не показывает
звезду. Метрика фиксирует сеть, её обучающий датасет, ревизию состава, effective inference-шаблон, профиль и версию
evaluator. Поле датасета тестовой разметки означает источник её тайлов, а отдельные ссылки фиксируют использованные
сеть этого датасета и точную псевдоразметку. Эта пара используется отдельно для создания набора, черновых
`evaluate-preview`/`optimize-preview` и поснимочного кэша оптимизатора. Матрица `training_result_test_metrics`
остаётся независимым контуром: успешные сети всех
обычных датасетов класса оцениваются на единственной основной тестовой разметке класса. Для сети управляемого
датасета матрица хранит составную оценку: каждому semantic-каналу соответствует основная тестовая разметка его
исходного класса, а сигнатура задания включает весь набор sample/revision/enabled tiles. Pixel/object F1 каждого
класса считается независимо только на его выборке; общий F1 равен простому арифметическому среднему поклассовых
F1 без взвешивания по числу пикселей, объектов или тайлов. Автоматическая сверка охватывает три последние сети
каждого датасета; ручная команда датасета охватывает все его успешные сети. Изменение одной исходной основной
выборки пересчитывает только её semantic-канал зависимой управляемой сети, объединяет его с сохранёнными метриками
остальных каналов и заново считает среднее. При отсутствии полной сохранённой структуры выполняется полный расчёт.
Все задания прямого и матричного F1 имеют срочный inference-приоритет: queued-оценка выполняется до обучения, а
выполняющееся обучение кооперативно приостанавливается существующим `pause.request` и продолжается после оценок.
Семантическая PNG-маска объединяет пиксели одного класса и применяется только в пиксельном контуре оценки.
Соприкасающиеся исходные объекты остаются отдельными feature в GeoJSON; визуализация и помеченные JPEG проводят
границы по их instance-ID, а объектовая метрика сопоставляет эти векторные feature один к одному.
При включённом `exclude_boundary_objects` исходный объект должен полностью помещаться в footprint тайла;
пересекающий границу объект целиком исключается из отбора, квот, GeoJSON, маски, превью и объектовой метрики.
Флаг разрешён только для классов с основной объектовой метрикой и сохраняется в batch-запуске и тестовом наборе.
Каталог результатов выделяет основной датасет и на каждой карточке показывает F1 сети этого датасета: явно
отмеченной основной, если она принадлежит ему, либо последней успешной. Оценка берётся по общей основной тестовой
разметке класса из существующего результата; для управляемого датасета карточка отдельно показывает средний и
цветные поклассовые F1 по основным выборкам источников. Каждый поклассовый показатель занимает собственную строку
в границах своей карточки или ячейки; длинное имя переносится, а числовой F1 остаётся видимым. Блоки метрик не
конкурируют с названием датасета, количеством снимков и действиями. Открытие каталога не создаёт отдельный
расчёт.

Seed inference-шаблона рек идемпотентно добавляет Smooth `1 / 0.125`, меняет Simplify с `15` на `1 м` и
сохраняет явные пользовательские переопределения. В Geoalert pipeline речные полигоны помечаются в полосе одного
пикселя от границы растра, затем проходят цепочку
`FilterSmallObjects → RemoveSmallHoles → FilterCompactObjects → Smooth → Simplify`; служебный тег удаляется
последним brick. Compact-фильтр не имеет порога площади. Граничные и вырожденные геометрии сохраняются,
а невалидные не приводят к ошибке фильтрации. Озёрный шаблон не включает compact-фильтр и сглаживание.

## Список используемых данным модулем модулей и с какой целью

- `models.api` - получить публичный список поддерживаемых архитектур.
- `mlflow_adapter.api` - получить и создать MLflow experiments, прочитать лучший нативный checkpoint либо произвольный артефакт завершённого run и скачать его для инференса; сервис не пишет MLflow-метрики.
- `mlflow_adapter.contracts` - передать публичные DTO experiment, нативного checkpoint и артефакта завершённого run.
- `settings.contracts` - валидировать YAML-настройки, сформированные для запуска training CLI.
- `dataset_preparing.api` и `dataset_preparing.contracts` - сопоставлять legacy TXT и per-image GeoJSON с TIFF.

### Автоматизация

Автоматизация хранится в таблицах `automation_controls` и `automation_rules`. Правило задается парой
`dataset_key + architecture` и двумя флагами: `training_enabled` и `pseudo_markup_enabled`. `Custom` не участвует в
автоматизации. Git checkout и атомарный runtime с `.mlsystem2-release-metadata.json` сохраняют формат
`git:{commit_sha}`; иначе используется `fs:{mtime_ns}`. После изменения источника, типа снимков или метрики версия содержит управляемую ревизию и
файловую версию, поэтому активное правило видит значимое изменение.

HTTP API и worker очередей запускаются отдельными процессами; API не исполняет фоновые jobs и не материализует управляемые датасеты. Каталог создаёт идемпотентный запрос, а worker под межпроцессным lock строит по одному cache и атомарно публикует его. Готовая сводка устраняет повторный разбор GeoJSON; footprint переиспользуется из источника. Dispatch очереди выполняется с коротким интервалом, а синхронизация автоматизации — с отдельным более редким интервалом. При включенном глобальном switch автоматизация создаёт
auto training job для текущей версии датасета, если нет текущего auto result/job для этой версии. Defaults берутся
из активного шаблона конкретного датасета `(architecture, dataset_key)`. После смены ключа датасета осиротевшая
связь однозначно восстанавливается по имени `класс\датасет`; если датасетный шаблон не создан, используется
базовый шаблон сети `(architecture, null)`. После успешного auto training result с MLflow run id создается auto pseudo-markup job;
для per-image датасета временный TXT формируется из сопоставленных TIFF. Очередь jobs единая: ручная псевдоразметка имеет приоритет выше ручного обучения,
ручное обучение выше auto псевдоразметки, auto псевдоразметка выше auto обучения, а второстепенные ручные jobs идут после всех обычных. Auto jobs нельзя удалить или двигать
через endpoints очереди; снятие галочки отменяет соответствующие queued/running auto jobs. Если меняется версия
конкретного датасета, активные auto jobs предыдущей версии отменяются только для этого датасета и модели. Failed
auto attempt не ретраится до новой версии или снятия и повторного включения галочки.
Срочный поснимочный инференс редактора обгоняет queued jobs. Точный ключ дедупликации фиксирует основную сеть,
ревизию TIFF, effective inference-конфигурацию и версию алгоритма; повторные запросы возвращают существующий job,
а ошибочный job перезапускается только явно. Running training получает файловый
pause-request: на границе batch модель и optimizer state переносятся в CPU, CUDA освобождается, а job получает
`paused`. После завершения срочных jobs запрос снимается, тот же PID и MLflow-run продолжают обучение. Обычный
выполняющийся inference не прерывается.

При ручном запуске обучения оператор может включить штатный инференс по снимкам обучающего датасета. Опция
хранится как служебная часть задания и не попадает в train YAML. После успешного создания каждого результата
worker идемпотентно ставит обычный full pseudo-markup job через общий builder; ошибка постановки инференса не
меняет успешный статус уже завершённого обучения. Счётчик рядом с пунктом «Очередь» читает отдельный лёгкий
endpoint и показывает сумму `queued`, `running` и `paused`, а строка задания ведёт в тот же экран Job, что и
кнопка из результатов обучения. На экране задания обучения доступны переход к результатам его датасета
и открытие связанного запуска MLflow в новой вкладке; до появления ссылки кнопка MLflow недоступна.
Флаг `secondary_priority` также хранится только в служебной части задания и наследуется этой full-псевдоразметкой.
Второстепенный job запускается только при отсутствии разрешённых обычных jobs. При их появлении training
освобождает CUDA штатным batch-boundary pause, PyTorch inference сохраняет завершённые поснимочные результаты и
переносит модель в CPU между снимками, а Geoalert Compose между снимками выгружает модель из Triton. После
опустошения обычной очереди запрос паузы снимается и тот же процесс продолжает работу без повторной обработки
готовых снимков. Состояние `paused` в API одинаково для обучения и псевдоразметки.

Для active manual training очередь и страница Job открывают единый диалог остановки. Если локальный
`scratch/checkpoints/best.pt` уже создан, оператор может записать `stop-and-save-best.request`: API не завершает
процесс и не пишет MLflow, а конвейер обучения сам отбрасывает незавершённую эпоху, публикует чекпойнт лучшей по
F1 завершённой эпохи и завершает run со статусом `FINISHED`. Worker после exit code `0` создаёт обычный успешный
результат и сохраняет прежнюю семантику заказанного post-training inference. Альтернатива «без результата»
использует отмену с `SIGTERM` всей группе процессов и `KILLED` в MLflow. Остановка ожидает до пяти
секунд, затем отправляет `SIGKILL` оставшимся процессам и проверяет завершение ещё до двух секунд.
Временные данные и PID удаляются только после подтверждения; при ошибке задание сохраняется.
Worker проверяет также дочерние процессы после выхода лидера группы, чтобы следующий job не занял
GPU во время завершения предыдущего. Завершившиеся zombie-процессы не удерживают очередь.

Глобальное выключение автоматизации через `PUT /api/v1/automation/enabled` с `enabled=false` отменяет все active
auto jobs: queued rows уходят из очередей, running process получает SIGTERM, временная директория удаляется,
результаты получают `cancelled`, а известный MLflow run помечается как `KILLED`. При повторном включении worker
создает новые jobs по текущим правилам и версиям датасетов, а не восстанавливает старую очередь.

## Алгоритм работы и его особенности

Экспорт object f1 содержит ONNX с двумя вероятностями, YAML блока ObjectF1Segmentation и CPU runtime с инструкцией регистрации. Версия кэша экспорта — 4; ID сохраняются до векторизации. Межсценовый NMS вызывается только для групп пересекающихся объектов; независимые объекты проходят без объединения геометрии.

Для SegFormer B0/B1/B2/B3 доступен профиль object_f1 с подробным описанием. Схемы базовых и датасетных шаблонов обновляются штатно; batch size учитывает архитектуру и размер тайла по таблице next-gen2. В контрактах заданий/результатов сохранён pipeline_variant. Ортофото использует ObjectF1Segmentation в Geoalert/Triton с двумя картами вероятностей и общим inference API; нативный путь сохраняет instance ID. Экспорт и кэш отличают object_probabilities от бинарных масок. Новых endpoints и миграций нет.

Все SegFormer каталога допускают `next-gen2` на основе segFormer_train_hlam_main_v2.ipynb для Канопус и RGB-ортофотопланов.
Профиль задаётся ConfigSchema.pipeline_defaults; подробное описание — pipeline_descriptions.next_gen2.
Доступны предобученные веса, размер тайла 512/768/1024/1536 (default 512), максимум эпох (20), patience по validation loss (10) и лимит времени (null).
Форма показывает модель/датасет и размер, условия остановки и описание внизу с визуализацией тайла.
Шаг равен половине размера; batch для B0 — 16/8/4/2, B1 — 8/4/2/1, B2/B3 — 4/2/1/1 соответственно.
Аугментации, LR 1e-4, CrossEntropy + Tversky, seed 42 остаются фиксированными. Settings проверяет профиль.
Worker не переопределяет число загрузчиков: num_workers наследуется из серверного settings.yml (8).
При первом чтении старые next-gen2 шаблоны получают новый профиль и defaults; повторное чтение
сохраняет технические настройки. История jobs и MLflow не переписывается.
Нативный инференс `next-gen2` воспроизводит eval-ноутбук: полные окна, исходные значения TIFF без маскирования
nodata, нормализация каждого окна, softmax класса 1 и гауссово усреднение вероятностей перед строгим порогом
`> 0.9`; batch вдвое больше обучающего для архитектуры и размера окна. Диагностический порог validation `0.5` не меняется. Автоматическая морфология и упрощение по числу
снимков отключены; явный inference-шаблон по-прежнему применяется. Псевдоразметка и контрольная F1 используют
один алгоритм. AOI для этого варианта оценивает полные сцены и затем отбирает пересекающиеся объекты.
Версия профиля входит в сигнатуру контрольной оценки только для `next-gen2`, поэтому прежние метрики этого
варианта пересчитываются, а оценки остальных конвейеров остаются действующими.

Служебное восстановление исторических combined→managed наборов читает последний полный Git-снимок удалённой
папки и дополняет текущие binary-источники только непокрытой положительной геометрией. Hard negative из снимка и
актуальных источников канонизируются и одинаково записываются во все источники. Перед записью проверяется полное
покрытие, активные задания запрещены, автоматизация остаётся выключенной до публикации MLMarkup; успешные
checkpoint засчитываются новой версии без переобучения. Повторный dry-run не должен находить изменений.

Черновик редактора включает геометрию и отменяемую пометку удаления снимка. Отмена использует физическую клавишу
`Z`, поэтому одинаково принимает `Ctrl+Z` и `Ctrl+Я`. Кнопка карты раскрывает рабочее поле с инструментами через
браузерный полноэкранный API либо резервный режим на всю область окна, если API запрещён. `Esc` возвращает страницу,
а OpenLayers после перехода пересчитывает размер карты.
Черновик сохраняется только автоматически;
ручной кнопки сохранения нет. Серверное автосохранение перед валидацией пересекает каждый полигон с фактическим
footprint: частичный выход, включая nodata-провал, подрезается, полностью внешний объект удаляется, а нормализованный
GeoJSON становится новым состоянием клиента. Редактор допускает к добавлению только TIFF в `EPSG:3857`, при чтении
перепроецирует исторический GeoJSON в CRS актуального TIFF, а при сохранении канонизирует CRS и пересобирает companion-footprint.
В браузере TIFF уже добавленные снимки остаются в текущей сортировке, отмечаются
зелёной рамкой и недоступны для повторного выбора, а добавление всей папки пропускает их.

Per-image сцена состоит из supervision-разметки и companion-файла `*_footprint.geojson` в CRS TIFF. Каталог,
сопоставление сцен и подсчёты игнорируют companion как разметку; worker переносит его в immutable snapshot как часть
версии датасета. Редактор создаёт footprint по `dataset_mask`, пересборка восстанавливает все пары, а публикация
разметки обновляет companion по актуальному TIFF; публикация удаления снимка удаляет оба файла. Наборы, созданные
до введения companion-файла, остаются читаемыми.

Список пользователей задаётся вне Git через `MLSYSTEM2_TRAINING_UI_USERS_JSON`. У пользователя есть каноническое
имя, индивидуальный пароль, роль `admin|user` и optional aliases; ограничения по роли пока не включены. Cookie,
серверные черновики и Git author публикации используют каноническое имя.

Перед созданием поснимочного inference-job frontend отдельным read-only запросом проверяет готовую полную
псевдоразметку эффективной сети и отправляет POST только после ответа `unavailable`. Это правило редактора датасета
не применяется к тестовому набору: его создание выбирает сеть конкретного исходного датасета и требует полную
псевдоразметку точной пары с совпадающим `dataset_key` и покрытием всех текущих TIFF из `scenes_file`.
Изменение только версии обучающих полигонов не требует повторного инференса.

При публикации управляемого датасета hard negative направляется в один, несколько или все исходные датасеты по
выбранным semantic-типам. Материализация сохраняет slug источника в `_mlsystem2_class`, поэтому обучение штрафует
только соответствующий выход сети. Одинаковая геометрия с общим origin, выбранная для всех источников, хранится
как один hard negative без класса. Исторический hard negative без класса также трактуется как общий и при
обратной публикации записывается во все источники. Версия алгоритма материализации отдельно входит в ключ кэша,
но не в логическую версию датасета: после выкладки кэш пересобирается без постановки всех управляемых наборов на
автоматическое переобучение.

Каталог различает legacy по TXT, per-image binary по отсутствию TXT и manifest-backed `per_image_multiclass`; пустой per-image набор доступен редактору, но не обучению. Worker копирует GeoJSON и manifest в immutable snapshot, автоматически фиксирует multiclass task, `output_channels=N+1`, class balance и совместимый loss. Единая схема ручного запуска и редактора шаблонов содержит `train.background_weight`; seed добавляет отсутствующее значение `1.0` во все существующие шаблоны, а worker переносит выбранное значение в `run.yml`. Шаблоны с входом `768`, в которых context ещё не был задан, получают `tile_preparation.context=128`; явный `0` сохраняется. Редактор считает role/class/provenance системными полями, но разрешает явную переклассификацию. Изменённый GeoJSON автоматически сохраняется в `dataset_editor_drafts` отдельно для пользователя без Git и инференса, переживает перезагрузку страницы и удаляется при отмене либо после успешной batch-публикации под Git-lock; optimistic-lock конфликт возвращает `409`. Перетаскивание с зажатым колесом перемещает снимок и подавляет нативную автопрокрутку браузера внутри карты, поэтому страница остаётся неподвижной; левая кнопка задаёт рамку выбора всех попавших в неё вершин. Modify добавляет вершину кликом по ребру. Карта получает клавиатурный фокус при нажатии, а `Delete` обрабатывается тем же действием, что кнопка удаления: удаляет выбранные вершины через общую undo-историю; кольца с менее чем тремя оставшимися вершинами не меняются, остальные автоматически замыкаются. Если вершины не выбраны, `Delete` удаляет выделенный полигон через ту же историю. Список снимков показывает цветные счётчики каждого смыслового типа и hard negative из текущего черновика. Псевдоразметка скрыта по умолчанию: основная сеть используется только если обучена на открытом датасете, иначе выбирается его последняя успешная сеть. Чужие сети и основной датасет класса не подставляются; выбор общий с созданием тестовых разметок. Любой готовый полный результат выбранной сети, содержащий TIFF в `scenes_file`, обрезается по снимку независимо от исходного датасета результата; историческое имя файла без подпапки принимается только при однозначном сопоставлении TIFF. Отсутствующий снимок проходит через общий builder и runner штатной псевдоразметки как приоритетный one-off job и показывается отдельным нередактируемым слоем. Frontend кэширует результат при переключении снимков, polling использует отдельный лёгкий endpoint, а выбранную сцену запрашивает только после загрузки списка именно открытого датасета. Удаление датасета проверяет активные jobs, удаляет только его Git-дерево и мягко архивирует каталог без очистки исторических таблиц. Combined builder читает обе source-папки целиком, чинит/попускает invalid geometry с отчётом, применяет priority, вычитает positive из hard negative, ищет TIFF по valid-data footprint и создаёт стабильные feature ID/origin-key. Rebuild preview фиксирует filesystem и Git trees; merge сохраняет ручную версию конфликтующего объекта, replace полностью заменяет per-image файлы, после чего один commit публикует manifest и все GeoJSON. Псевдоразметка, F1 и AOI читают полный тайл и записывают только центр согласно context; старые результаты без context используют `0`. В F1 размер сохранённого тестового TIFF определяет только итоговую маску: вся его площадь покрывается окнами размера сети, а нативный runner предпочитает `sample_size/inference_context` checkpoint и использует конфигурацию исходного training-job только для старых checkpoint. `JobRow.tile_size` такого задания означает вход сети, не ширину тестового TIFF. Экспорт берёт context из metadata нового checkpoint либо ручного override старого. Для моделей Канопуса эти операции используют совместимый `pytorch_one_off`; модели ортофото всегда выполняются через кешируемый Triton export и штатный Geoalert `Compose`, включая внешние pipeline ЗУ500/ОКС500. Формат задания и итогового GeoJSON одинаков для обоих backend. Нативный checkpoint либо проверенный ZIP `external_torchscript`, multiclass class schema, Binary ABI и legacy DTO остаются совместимыми. Live-каталог редактор не изменяет.

Python-backend внешней модели ЗУ500 использует `torch/torchvision` из read-only mount окружения MLSystem2,
заданного `MLSYSTEM2_GEOALERT_TRITON_PYTHON_SITE_PACKAGES`; вычисление всё равно исполняется внутри Triton и
вызывается штатным бриком `NSPDParcels`. Для него контейнер Triton имеет `/dev/shm` не менее `1 GiB`.
Полная псевдоразметка публикуется только при `processed == unique_image_count` без failed/missing; `partial`
сохраняется как диагностика неуспешного job и не становится готовым результатом.
