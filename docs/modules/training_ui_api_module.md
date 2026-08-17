# Модуль training_ui_api

## Назначение

`training_ui_api` — FastAPI-сервис сайта MLSystem2. Он хранит UI-данные и управляемый каталог датасетов, синхронизирует опубликованный `/data/MLMarkup`, управляет очередями training/inference и предоставляет Git-backed редактор per-image разметки. Редактор пишет только в отдельный `mlmarkup-editor`, а обучение читает только атомарно опубликованный live-релиз.

Frontend — React + TypeScript + Vite SPA. TypeScript-типы генерируются из OpenAPI командой
`npm run generate:api --prefix frontend`, production static собирается в `frontend/dist`, а сервер Node.js
на проде не нужен.

## Публичный интерфейс

- `create_app() -> Any` - создает FastAPI-приложение.
- `get_openapi_schema() -> dict[str, Any]` - возвращает OpenAPI-схему сервиса.
- `main() -> None` - запускает `uvicorn` для сервиса.
- `worker_main() -> None` - отдельно запускает исполнителей общей очереди и batch-задач тестовых разметок.

## Публичные контракты

- `TrainingUIAPIError` - ошибка сервиса.
- `TemplateSource`, `JobType`, `JobSource`, `JobStatus`, `ResultStatus`, `StoredFileKind` - enum-значения API.
- `AppLink`, `AppLinksResponse` - ссылки Grafana/MLflow/MinIO.
- `BootstrapInfo` - стартовый DTO для React frontend: links, datasets, image folders, classes, models и оба набора templates одним ответом.
- `MLflowExperimentInfo`, `MLflowExperimentCreate` - experiments MLflow.
- `ImageryType`, `ImageryTypeInfo`, `DatasetFormat`, `DatasetInfo`, `DatasetObjectType`, `DatasetListResponse`, `ClassInfo`, `ClassListResponse` - управляемый каталог; `DatasetInfo` содержит `format=legacy|per_image|per_image_multiclass`, `task`, `object_types`, `combined`, source status/changes, class counts, legacy-файлы либо `annotations_dir`.
- `DatasetCatalogInfo`, `DatasetSourceInfo`, `DatasetClassCreate`, `DatasetClassUpdate`, `DatasetPrimaryDatasetUpdate`, `ManagedDatasetCreate`, `ManagedDatasetUpdate` - чтение и изменение активных классов и датасетов; мягко удалённые строки остаются в Postgres, но в каталог не входят.
- `ImageFolderInfo`, `ImageFolderListResponse` - папки подготовленных снимков из `MLSYSTEM2_IMAGES_ROOT` с количеством TIFF.
- `ModelInfo`, `ModelListResponse` - публичные модели из `models`.
- `ConfigField`, `ConfigSchema`, `TrainingTemplate`, `TrainingTemplateListResponse`, `TrainingTemplateCreate`, `TrainingTemplateUpdate`, `TrainingTemplateApplyField`, `InferenceTemplate`, `InferenceTemplateListResponse`, `InferenceTemplateCreate`, `InferenceTemplateUpdate`, `InferenceTemplateApplyField` - шаблоны обучения и инференса; `ConfigField` содержит `tooltip`, допустимые границы и optional `recommended_range` для UI-подсказок.
- Встроенный датасетный inference-шаблон хранит человекочитаемую цель, но при инициализации привязывается к ключу действующей строки каталога по паре `класс/имя`; это сохраняет специальные настройки после смены ключа или миграции legacy-датасета.
- `StoredFileInfo`, `CustomDatasetInfo` - загруженные файлы и custom datasets.
- `TrainingJobCreate`, `QueueEnabledUpdate`, `QueueControlInfo`, `JobSummary`, `QueueSnapshot`, `JobDetail` - задания и очереди.
- `PseudolabelJobCreate`, `PseudolabelClassInfo`, `PseudolabelClassListResponse`, `PseudolabelJobInfo`, `PseudolabelErrorInfo` - AOI, доступная зафиксированная модель, состояние и структурированная ошибка QGIS-контракта.
- `AutomationEnabledUpdate`, `AutomationRuleUpdate`, `AutomationRuleInfo`, `AutomationSnapshot` - глобальный выключатель и матрица автоматизации `датасет × модель`.
- `TrainingResultInfo`, `TrainingResultTestF1Info`, `PrimaryTestSampleInfo`, `PseudoMarkupResultInfo`, `DatasetResultsResponse`, `ResultClassInfo`, `ResultDatasetInfo`, `ResultClassListResponse`, `ResultChangeInfo`, `ResultChangesResponse` - результаты обучения, task/class schema, структурированные per-class метрики, отдельный test F1, основная разметка и карточки классов; multiclass pseudo result дополнительно содержит ZIP-download по типам.
- `TrainingResultExportItem`, `TrainingResultBatchExportRequest` - JSON-запрос массового экспорта выбранных успешных training results.
- `MarkupExportRequest`, `MarkupExportTileInfo`, `MarkupExportInfo` - запрос и описание временного набора тестовой разметки с тайлами, превью, сводкой цель/факт и сроком хранения.
- `TestSampleCreate`, `TestSampleUpdate`, `TestSampleTileUpdate`, `TestSampleOptimizeRequest`, `TestSamplePrimaryUpdate`, `TestSampleEvaluationPreviewRequest`, `TestSampleDownloadRequest`, `TestSampleBulkDownloadRequest` - создание и атомарное сохранение постоянной разметки, совместимые точечные изменения, ограничения оптимизации, запросы оценки, одиночного и группового скачивания.
- `TestSampleMetric`, `TestSampleEvaluationInfo`, `TestSampleSummary`, `TestSampleDatasetGroup`, `TestSampleClassGroup`, `TestSampleCatalogResponse`, `TestSampleTileInfo`, `TestSampleDetail`, `TestSampleDraftPreview` - binary scalar и multiclass per-class/macro/micro/foreground метрики, task/schema/counts, каталог, редакторское описание и незаписываемый результат чернового расчёта.
- `TestSampleBatchItemCreate`, `TestSampleBatchCreate`, `TestSampleBatchItemInfo`, `TestSampleBatchInfo` - запрос и прогресс группового создания; квадратный размер тайла выбирается из `512`, `768`, `1024`, `1536`, `2048`, `2560`, `3072`, `3584`, последний запуск хранит применённые настройки формы.
- `DatasetEditorDatasetInfo`, `DatasetEditorObjectType`, `DatasetEditorDatasetListResponse`, `DatasetEditorSceneInfo`, `DatasetEditorSceneListResponse`, `DatasetEditorSceneDetail`, `DatasetEditorPseudoMarkupInfo`, `DatasetEditorDraftSummary`, `DatasetEditorDraftInfo` - каталог per-image датасетов и сцен с task/schema, class counts, source status, revision, пользовательским серверным черновиком, `valid_data_footprint` и состоянием псевдоразметки сцены.
- `DatasetEditorRebuildPreview`, `DatasetEditorRebuildRequest`, `DatasetEditorRebuildResult`, `DatasetEditorRebuildChange` - preview token, source/local changes, конфликты и атомарная пересборка `merge|replace`.
- `DatasetEditorRasterFolderInfo`, `DatasetEditorRasterInfo`, `DatasetEditorRasterBrowserResponse` - прямые папки и TIFF из разрешённого server-side каталога.
- `DatasetEditorAddScenesRequest`, `DatasetEditorSaveSceneRequest`, `DatasetEditorSaveDraftRequest`, `DatasetEditorDiscardDraftsResult`, `DatasetEditorPublishSceneRequest`, `DatasetEditorPublishRequest`, `DatasetEditorDeleteSceneRequest`, `DatasetEditorMutationResult`, `DatasetEditorPublicationInfo` - серверные черновики, одиночные и атомарные batch optimistic-lock мутации и статус публикации commit SHA.
- `GET /api/v1/bootstrap` - агрегированный стартовый endpoint для frontend; старые catalog/template endpoints остаются рабочими.
- `GET /api/v1/dataset-catalog` и `POST /api/v1/dataset-catalog/sync` - иерархия редактора и явная идемпотентная синхронизация с MLMarkup.
- `POST|PATCH /api/v1/dataset-classes`, `PUT /api/v1/dataset-classes/{class_key}/primary-dataset`, `POST|PATCH /api/v1/managed-datasets` - создание и редактирование каталога. Назначение занятого источника переносит существующий датасет с сохранением его ключа, а смена источника безопасно обменивает источники и увеличивает ревизии обеих сущностей.
- `DELETE /api/v1/dataset-editor/datasets/{dataset_key}` - удалить папку датасета Git-коммитом, мягко архивировать его строку в Postgres и снять назначение основным; задания, результаты и MLflow не удаляются, при активном задании операция отклоняется.
- `GET /api/v1/dataset-editor/datasets` и `GET /api/v1/dataset-editor/datasets/{dataset_key}/scenes[/{annotation_name}]` - список редактируемых датасетов, сцены и GeoJSON.
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/rasters` и `GET /api/v1/dataset-editor/datasets/{dataset_key}/raster/{image_path}` - серверный выбор снимков и авторизованный TIFF с HTTP Range.
- `GET|POST /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}/pseudo-markup` - получить готовый фрагмент текущей основной сети либо идемпотентно поставить срочный поснимочный инференс.
- `GET /api/v1/dataset-editor/pseudo-markup/{job_id}` - лёгкий polling поснимочного задания без Git-синхронизации и разрешения полного каталога.
- `PUT|DELETE /api/v1/dataset-editor/datasets/{dataset_key}/drafts/{annotation_name}` и `DELETE /api/v1/dataset-editor/datasets/{dataset_key}/drafts` - автоматически сохранить промежуточный пользовательский GeoJSON или пометку удаления без публикации либо удалить один/все свои черновики.
- `POST /api/v1/dataset-editor/datasets/{dataset_key}/drafts/publish` - атомарно опубликовать все сохранённые изменения и удаления пользователя одним Git-коммитом.
- `POST|PUT /api/v1/dataset-editor/datasets/{dataset_key}/scenes`, `PUT|DELETE /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}` - добавить отсутствующие TIFF/папку, атомарно опубликовать несколько GeoJSON, совместимо сохранить один GeoJSON или создать отменяемую пометку удаления сцены.
- `GET /api/v1/dataset-editor/publication/{commit}` - `publishing|published` по ancestry commit текущего live-релиза.
- `POST /api/v1/dataset-editor/datasets/{dataset_key}/rebuild/preview` и `POST /api/v1/dataset-editor/datasets/{dataset_key}/rebuild` - preview и атомарная merge/replace-пересборка combined dataset; изменение source или target после preview возвращает `409`.
- `GET /api/v1/files/{file_id}/download-by-type` - ZIP канонической multiclass-псевдоразметки с одним GeoJSON на object type.
- `POST /api/v1/model-export/triton-zip` - multipart endpoint для сборки zip-архива модели под
  `models-serving-service` и Triton CPU; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/{result_id}/triton-zip` - multipart endpoint для сборки такого же zip-архива
  из `checkpoints/best.pt` либо внешнего ZIP успешного результата в MLflow; endpoint возвращает файл и не создаёт записей в БД.
- `POST /api/v1/results/training/triton-zip` - JSON endpoint для сборки общего zip-архива нескольких успешных
  нативных и импортированных результатов; каждая модель собирается тем же кодом, что одиночный экспорт результата, endpoint
  возвращает файл и не создает записей в БД.
- `POST /api/v1/scene-list-export` - multipart endpoint с `imagery_type=kanopus|ortho`, optional
  `include_footprints` и GeoJSON; рекурсивно находит TIFF с полигональными объектами. Без флага возвращает
  совместимый UTF-8 TXT относительных путей внутри корня типа снимков без расширений, с флагом — ZIP из этого
  TXT и GeoJSON футпринтов выбранных снимков в WGS84. Пустой список допустим, а повтор относительного пути без
  расширения отклоняется как неоднозначность TXT.
- `POST /api/v1/markup-export` - синхронно формирует временный набор тестовой разметки для датасета MLMarkup.
- `GET /api/v1/markup-export/{export_id}/tiles/{tile_index}/preview` - возвращает PNG-превью тайла с контуром маски.
- `GET /api/v1/markup-export/{export_id}/download` - возвращает плоский ZIP сформированного набора.
- `GET /api/v1/test-samples` и `POST /api/v1/test-samples` - иерархический каталог и создание постоянной тестовой разметки.
- `GET|PATCH|DELETE /api/v1/test-samples/{sample_id}` - просмотр, атомарное сохранение имени, основного статуса и полного состава либо удаление разметки.
- `PATCH /api/v1/test-samples/{sample_id}/tiles/{tile_index}` - включает или выключает тайл.
- `POST /api/v1/test-samples/reconcile` - идемпотентно ставит в inference-очередь отсутствующие и устаревшие прямые оценки всех сохранённых разметок текущими основными сетями классов.
- `POST /api/v1/test-samples/{sample_id}/evaluate` - принудительно ставит прямой пересчёт pixel/object F1 текущей основной сетью класса.
- `POST /api/v1/test-samples/{sample_id}/pseudo-markup` - идемпотентно ставит штатную полную псевдоразметку исходного датасета текущей основной сетью класса для предварительной оценки и оптимизации.
- `POST /api/v1/test-samples/{sample_id}/optimize` - подбирает состав из всех тайлов по основной метрике класса; request-поле старого клиента принимается, но не меняет выбор метрики.
- `POST /api/v1/test-samples/{sample_id}/evaluate-preview` и `POST /api/v1/test-samples/{sample_id}/optimize-preview` - рассчитывают F1 или оптимальный состав черновика без записи в БД.
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview` и `GET /api/v1/test-samples/{sample_id}/download` - постоянное превью и ZIP сохранённых включённых тайлов.
- `POST /api/v1/test-samples/{sample_id}/download` - ZIP явно выбранных тайлов текущего черновика без изменения разметки в БД; флаг `include_previews` оставляет полный состав либо только TIFF и GeoJSON.
- `POST /api/v1/test-samples/download` - несжатый ZIP явно выбранных сохранённых разметок, не более одной на класс; до восьми разметок готовятся параллельно, каждая в папке `<класс>_<исходный датасет>`.
- `POST /api/v1/test-sample-batches`, `GET /api/v1/test-sample-batches/latest` и `GET /api/v1/test-sample-batches/{batch_id}` - запуск и прогресс последовательного группового создания для готовых датасетов старого и поснимочного формата; клиент применяет тот же критерий доступности, что и сервер.
- `PUT /api/v1/test-samples/{sample_id}/primary` - совместимо назначает, заменяет или снимает единственную основную разметку класса.
- `GET /api/v1/results/datasets/{dataset_key}`, `POST /api/v1/results/datasets/{dataset_key}/pseudo-markup` и `POST /api/v1/results/datasets/{dataset_key}/test-f1` - результаты датасета, ручная псевдоразметка и постановка недостающих либо устаревших оценок в inference-очередь.
- `POST|DELETE /api/v1/results/training/{result_id}/primary` - явное назначение успешной сети основной для её класса или снятие отметки; без назначения расчёты используют последнюю успешную сеть без звезды.
- `GET /api/v1/pseudolabel/classes`, `POST /api/v1/pseudolabel/jobs`, `GET|DELETE /api/v1/pseudolabel/jobs/{job_id}` и `GET /api/v1/pseudolabel/jobs/{job_id}/result` - серверное распознавание AOI без передачи клиентских растров; полный контракт описан в `docs/pseudolabel_api.md`.

Сохранённая контрольная метрика каждой тестовой разметки получается прямым инференсом эффективной сети класса:
явно назначенной либо, при отсутствии назначения, последней успешной. Неявный выбор не сохраняется и не показывает
звезду. Метрика фиксирует сеть, её обучающий датасет, ревизию состава, effective inference-шаблон, профиль и версию
evaluator. Поле датасета тестовой разметки означает только источник её тайлов. Псевдоразметка этого источника
используется отдельно для создания набора, черновых `evaluate-preview`/`optimize-preview` и поснимочного кэша
оптимизатора. Матрица `training_result_test_metrics` остаётся независимым контуром: все успешные сети всех
датасетов класса оцениваются на единственной основной тестовой разметке класса.
Каталог результатов выделяет основной датасет и показывает F1 эффективной сети класса на всех карточках его
датасетов, не создавая отдельных расчётов для каждой карточки.

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

HTTP API и worker очередей запускаются отдельными процессами; API не исполняет фоновые jobs. Worker перед dispatch очередей вызывает синхронизацию автоматизации. При включенном глобальном switch он создает
auto training job для текущей версии датасета, если нет текущего auto result/job для этой версии. Defaults берутся
из активного шаблона конкретного датасета `(architecture, dataset_key)`, а если он не создан - из базового шаблона
сети `(architecture, null)`. После успешного auto training result с MLflow run id создается auto pseudo-markup job;
для per-image датасета временный TXT формируется из сопоставленных TIFF. Очередь jobs единая: ручная псевдоразметка имеет приоритет выше ручного обучения,
ручное обучение выше auto псевдоразметки, auto псевдоразметка выше auto обучения. Auto jobs нельзя удалить или двигать
через endpoints очереди; снятие галочки отменяет соответствующие queued/running auto jobs. Если меняется версия
конкретного датасета, активные auto jobs предыдущей версии отменяются только для этого датасета и модели. Failed
auto attempt не ретраится до новой версии или снятия и повторного включения галочки.
Срочный поснимочный инференс редактора обгоняет queued jobs. Точный ключ дедупликации фиксирует основную сеть,
ревизию TIFF, effective inference-конфигурацию и версию алгоритма; повторные запросы возвращают существующий job,
а ошибочный job перезапускается только явно. Running training получает файловый
pause-request: на границе batch модель и optimizer state переносятся в CPU, CUDA освобождается, а job получает
`paused`. После завершения срочных jobs запрос снимается, тот же PID и MLflow-run продолжают обучение. Уже
выполняющийся inference не прерывается.

Глобальное выключение автоматизации через `PUT /api/v1/automation/enabled` с `enabled=false` отменяет все active
auto jobs: queued rows уходят из очередей, running process получает SIGTERM, временная директория удаляется,
результаты получают `cancelled`, а известный MLflow run помечается как `KILLED`. При повторном включении worker
создает новые jobs по текущим правилам и версиям датасетов, а не восстанавливает старую очередь.

## Алгоритм работы и его особенности

Служебное восстановление исторических combined→managed наборов читает последний полный Git-снимок удалённой
папки и дополняет текущие binary-источники только непокрытой положительной геометрией. Hard negative из снимка и
актуальных источников канонизируются и одинаково записываются во все источники. Перед записью проверяется полное
покрытие, активные задания запрещены, автоматизация остаётся выключенной до публикации MLMarkup; успешные
checkpoint засчитываются новой версии без переобучения. Повторный dry-run не должен находить изменений.

Черновик редактора включает геометрию и отменяемую пометку удаления снимка. Он сохраняется только автоматически;
ручной кнопки сохранения нет. В браузере TIFF уже добавленные снимки остаются в текущей сортировке, отмечаются
зелёной рамкой и недоступны для повторного выбора, а добавление всей папки пропускает их.

Per-image сцена состоит из supervision-разметки и companion-файла `*_footprint.geojson` в CRS TIFF. Каталог,
сопоставление сцен и подсчёты игнорируют companion как разметку; worker переносит его в immutable snapshot как часть
версии датасета. Редактор создаёт footprint по `dataset_mask`, пересборка восстанавливает все пары, а публикация
удаления снимка удаляет оба файла. Наборы, созданные до введения companion-файла, остаются читаемыми.

Список пользователей задаётся вне Git через `MLSYSTEM2_TRAINING_UI_USERS_JSON`. У пользователя есть каноническое
имя, индивидуальный пароль, роль `admin|user` и optional aliases; ограничения по роли пока не включены. Cookie,
серверные черновики и Git author публикации используют каноническое имя.

Перед созданием поснимочного inference-job frontend отдельным read-only запросом проверяет готовую полную
псевдоразметку эффективной сети и отправляет POST только после ответа `unavailable`. Если legacy `scenes_file`
удалён миграцией, результат того же датасета остаётся полным при точном совпадении `dataset_key` и управляемой
версии, которая фиксирует состав сцен, включая пустые предсказания.

Каталог различает legacy по TXT, per-image binary по отсутствию TXT и manifest-backed `per_image_multiclass`; пустой per-image набор доступен редактору, но не обучению. Worker копирует GeoJSON и manifest в immutable snapshot, автоматически фиксирует multiclass task, `output_channels=N+1`, class balance и совместимый loss. Шаблоны с входом `768`, в которых context ещё не был задан, получают `tile_preparation.context=128`; явный `0` сохраняется. Редактор считает role/class/provenance системными полями, но разрешает явную переклассификацию. Изменённый GeoJSON автоматически сохраняется в `dataset_editor_drafts` отдельно для пользователя без Git и инференса, переживает перезагрузку страницы и удаляется при отмене либо после успешной batch-публикации под Git-lock; optimistic-lock конфликт возвращает `409`. Перетаскивание с зажатым колесом перемещает снимок и подавляет нативную автопрокрутку браузера внутри карты, поэтому страница остаётся неподвижной; левая кнопка задаёт рамку выбора всех попавших в неё вершин. Modify добавляет вершину кликом по ребру, а `Delete` удаляет выбранную группу через общую undo-историю; кольца с менее чем тремя оставшимися вершинами не меняются, остальные автоматически замыкаются. Список снимков показывает цветные счётчики каждого смыслового типа и hard negative из текущего черновика. Псевдоразметка скрыта по умолчанию: явная основная сеть класса имеет приоритет, а без звезды редактор выбирает последнюю сеть открытого датасета с резервом на последнюю сеть класса. Любой готовый полный результат выбранной сети, содержащий TIFF в `scenes_file`, обрезается по снимку независимо от исходного датасета результата; историческое имя файла без подпапки принимается только при однозначном сопоставлении TIFF. Отсутствующий снимок проходит через общий builder и runner штатной псевдоразметки как приоритетный one-off job и показывается отдельным нередактируемым слоем. Frontend кэширует результат при переключении снимков, polling использует отдельный лёгкий endpoint, а выбранную сцену запрашивает только после загрузки списка именно открытого датасета. Удаление датасета проверяет активные jobs, удаляет только его Git-дерево и мягко архивирует каталог без очистки исторических таблиц. Combined builder читает обе source-папки целиком, чинит/попускает invalid geometry с отчётом, применяет priority, вычитает positive из hard negative, ищет TIFF по valid-data footprint и создаёт стабильные feature ID/origin-key. Rebuild preview фиксирует filesystem и Git trees; merge сохраняет ручную версию конфликтующего объекта, replace полностью заменяет per-image файлы, после чего один commit публикует manifest и все GeoJSON. Нативная псевдоразметка, F1 и AOI читают полный тайл и записывают только центр согласно context; старые результаты без context используют `0`. Экспорт берёт context из metadata нового checkpoint либо ручного override старого. Псевдоразметка, F1 и AOI используют нативный checkpoint либо проверенный ZIP `external_torchscript`; multiclass инференс, метрики и экспорт сохраняют class schema. Binary ABI и legacy DTO остаются совместимыми. Live-каталог редактор не изменяет.
