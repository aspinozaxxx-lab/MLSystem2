# Модуль training_ui_api

## Назначение

`training_ui_api` — отдельный FastAPI-сервис сайта MLSystem2. Он отдает публичный HTTP API для frontend, хранит UI-данные обучения и управляемый каталог датасетов в Postgres БД/схеме, синхронизирует с ним новые папки `/data/MLMarkup`, управляет очередями training/inference и не выполняет прямой доступ frontend к БД.

Frontend — React + TypeScript + Vite SPA. TypeScript-типы генерируются из OpenAPI командой
`npm run generate:api --prefix frontend`, production static собирается в `frontend/dist`, а сервер Node.js
на проде не нужен.

## Публичный интерфейс

- `create_app() -> Any` - создает FastAPI-приложение.
- `get_openapi_schema() -> dict[str, Any]` - возвращает OpenAPI-схему сервиса.
- `main() -> None` - запускает `uvicorn` для сервиса.

## Публичные контракты

- `TrainingUIAPIError` - ошибка сервиса.
- `TemplateSource`, `JobType`, `JobSource`, `JobStatus`, `ResultStatus`, `StoredFileKind` - enum-значения API.
- `AppLink`, `AppLinksResponse` - ссылки Grafana/MLflow/MinIO.
- `BootstrapInfo` - стартовый DTO для React frontend: links, datasets, image folders, classes, models и оба набора templates одним ответом.
- `MLflowExperimentInfo`, `MLflowExperimentCreate` - experiments MLflow.
- `ImageryType`, `ImageryTypeInfo`, `DatasetInfo`, `DatasetListResponse`, `ClassInfo`, `ClassListResponse` - управляемый каталог классов и вложенных датасетов; класс задаёт `kanopus|ortho`, а датасет содержит метрику, число каналов, источник, его состояние, `scenes_file`, positive `annotation_file`, optional `hard_negative_annotation_file` и `diagnostics`.
- `DatasetCatalogInfo`, `DatasetSourceInfo`, `DatasetClassCreate`, `DatasetClassUpdate`, `DatasetPrimaryDatasetUpdate`, `ManagedDatasetCreate`, `ManagedDatasetUpdate` - чтение и изменение классов и датасетов без удаления.
- `ImageFolderInfo`, `ImageFolderListResponse` - папки подготовленных снимков из `MLSYSTEM2_IMAGES_ROOT` с количеством TIFF.
- `ModelInfo`, `ModelListResponse` - публичные модели из `models`.
- `ConfigField`, `ConfigSchema`, `TrainingTemplate`, `TrainingTemplateListResponse`, `TrainingTemplateCreate`, `TrainingTemplateUpdate`, `TrainingTemplateApplyField`, `InferenceTemplate`, `InferenceTemplateListResponse`, `InferenceTemplateCreate`, `InferenceTemplateUpdate`, `InferenceTemplateApplyField` - шаблоны обучения и инференса; `ConfigField` содержит `tooltip`, допустимые границы и optional `recommended_range` для UI-подсказок.
- `StoredFileInfo`, `CustomDatasetInfo` - загруженные файлы и custom datasets.
- `TrainingJobCreate`, `QueueEnabledUpdate`, `QueueControlInfo`, `JobSummary`, `QueueSnapshot`, `JobDetail` - задания и очереди.
- `AutomationEnabledUpdate`, `AutomationRuleUpdate`, `AutomationRuleInfo`, `AutomationSnapshot` - глобальный выключатель и матрица автоматизации `датасет × модель`.
- `TrainingResultInfo`, `TrainingResultTestF1Info`, `PrimaryTestSampleInfo`, `PseudoMarkupResultInfo`, `DatasetResultsResponse`, `ResultClassInfo`, `ResultDatasetInfo`, `ResultClassListResponse`, `ResultChangeInfo`, `ResultChangesResponse` - результаты обучения, число входных каналов, отдельный тестовый F1 сети, основная разметка и карточки классов; активные DTO содержат `job_id` и прогресс связанного задания.
- `TrainingResultExportItem`, `TrainingResultBatchExportRequest` - JSON-запрос массового экспорта выбранных успешных training results.
- `MarkupExportRequest`, `MarkupExportTileInfo`, `MarkupExportInfo` - запрос и описание временного набора тестовой разметки с тайлами, превью, сводкой цель/факт и сроком хранения.
- `TestSampleCreate`, `TestSampleUpdate`, `TestSampleTileUpdate`, `TestSampleOptimizeRequest`, `TestSamplePrimaryUpdate`, `TestSampleEvaluationPreviewRequest`, `TestSampleDownloadRequest`, `TestSampleBulkDownloadRequest` - создание и атомарное сохранение постоянной разметки, совместимые точечные изменения, ограничения оптимизации, запросы оценки, одиночного и группового скачивания.
- `TestSampleMetric`, `TestSampleEvaluationInfo`, `TestSampleSummary`, `TestSampleDatasetGroup`, `TestSampleClassGroup`, `TestSampleCatalogResponse`, `TestSampleTileInfo`, `TestSampleDetail`, `TestSampleDraftPreview` - метрики, каталог, редакторское описание и незаписываемый результат чернового расчёта тестовых разметок.
- `TestSampleBatchItemCreate`, `TestSampleBatchCreate`, `TestSampleBatchItemInfo`, `TestSampleBatchInfo` - запрос и прогресс группового создания; квадратный размер тайла выбирается из `512`, `768`, `1024`, `1536`, `2048`, `2560`, `3072`, `3584`, последний запуск хранит применённые настройки формы.
- `GET /api/v1/bootstrap` - агрегированный стартовый endpoint для frontend; старые catalog/template endpoints остаются рабочими.
- `GET /api/v1/dataset-catalog` и `POST /api/v1/dataset-catalog/sync` - иерархия редактора и явная идемпотентная синхронизация с MLMarkup.
- `POST|PATCH /api/v1/dataset-classes`, `PUT /api/v1/dataset-classes/{class_key}/primary-dataset`, `POST|PATCH /api/v1/managed-datasets` - создание и редактирование каталога; удаляющих endpoints нет. Назначение занятого источника переносит существующий датасет с сохранением его ключа, а смена источника безопасно обменивает источники и увеличивает ревизии обеих сущностей.
- `POST /api/v1/model-export/triton-zip` - multipart endpoint для сборки zip-архива модели под
  `models-serving-service` и Triton CPU; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/{result_id}/triton-zip` - multipart endpoint для сборки такого же zip-архива
  из `checkpoints/best.pt` успешного результата обучения в MLflow; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/triton-zip` - JSON endpoint для сборки общего zip-архива нескольких успешных
  результатов обучения; каждая модель собирается тем же кодом, что одиночный экспорт результата, endpoint
  возвращает файл и не создает записей в БД.
- `POST /api/v1/scene-list-export` - multipart endpoint с `imagery_type=kanopus|ortho` и GeoJSON; рекурсивно
  находит TIFF с полигональными объектами и возвращает UTF-8 TXT из имён файлов без расширений. Пустой список
  допустим, совпадающие имена подходящих TIFF отклоняются как неоднозначные.
- `POST /api/v1/markup-export` - синхронно формирует временный набор тестовой разметки для датасета MLMarkup.
- `GET /api/v1/markup-export/{export_id}/tiles/{tile_index}/preview` - возвращает PNG-превью тайла с контуром маски.
- `GET /api/v1/markup-export/{export_id}/download` - возвращает плоский ZIP сформированного набора.
- `GET /api/v1/test-samples` и `POST /api/v1/test-samples` - иерархический каталог и создание постоянной тестовой разметки.
- `GET|PATCH|DELETE /api/v1/test-samples/{sample_id}` - просмотр, атомарное сохранение имени, основного статуса и полного состава либо удаление разметки.
- `PATCH /api/v1/test-samples/{sample_id}/tiles/{tile_index}` - включает или выключает тайл.
- `POST /api/v1/test-samples/{sample_id}/evaluate` - пересчитывает пиксельный и объектный F1.
- `POST /api/v1/test-samples/{sample_id}/optimize` - подбирает состав из всех тайлов по основной метрике класса; request-поле старого клиента принимается, но не меняет выбор метрики.
- `POST /api/v1/test-samples/{sample_id}/evaluate-preview` и `POST /api/v1/test-samples/{sample_id}/optimize-preview` - рассчитывают F1 или оптимальный состав черновика без записи в БД.
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview` и `GET /api/v1/test-samples/{sample_id}/download` - постоянное превью и ZIP сохранённых включённых тайлов.
- `POST /api/v1/test-samples/{sample_id}/download` - ZIP явно выбранных тайлов текущего черновика без изменения разметки в БД; флаг `include_previews` оставляет полный состав либо только TIFF и GeoJSON.
- `POST /api/v1/test-samples/download` - несжатый ZIP явно выбранных сохранённых разметок, не более одной на датасет; до восьми разметок готовятся параллельно, каждая в папке `<класс>_<датасет>`.
- `POST /api/v1/test-sample-batches`, `GET /api/v1/test-sample-batches/latest` и `GET /api/v1/test-sample-batches/{batch_id}` - запуск и прогресс последовательного группового создания.
- `PUT /api/v1/test-samples/{sample_id}/primary` - совместимо назначает, заменяет или снимает основную разметку точного датасета.
- `GET /api/v1/results/datasets/{dataset_key}`, `POST /api/v1/results/datasets/{dataset_key}/pseudo-markup` и `POST /api/v1/results/datasets/{dataset_key}/test-f1` - результаты датасета, ручная псевдоразметка и постановка недостающих либо устаревших оценок в inference-очередь.

## Список используемых данным модулем модулей и с какой целью

- `models.api` - получить публичный список поддерживаемых архитектур.
- `mlflow_adapter.api` - получить и создать MLflow experiments, прочитать лучший checkpoint training run и скачать `checkpoints/best.pt` для псевдоразметки; сервис не пишет MLflow-метрики.
- `mlflow_adapter.contracts` - передать публичные DTO создания experiment и summary лучшего checkpoint.
- `settings.contracts` - валидировать YAML-настройки, сформированные для запуска training CLI.

### Автоматизация

Автоматизация хранится в таблицах `automation_controls` и `automation_rules`. Правило задается парой
`dataset_key + architecture` и двумя флагами: `training_enabled` и `pseudo_markup_enabled`. `Custom` не участвует в
автоматизации. До первого изменения настроек `DatasetInfo.version` сохраняет формат `git:{commit_sha}` или
`fs:{mtime_ns}`. После изменения источника, типа снимков или метрики версия содержит управляемую ревизию и
файловую версию, поэтому активное правило видит значимое изменение.

Worker перед dispatch очередей вызывает синхронизацию автоматизации. При включенном глобальном switch он создает
auto training job для текущей версии датасета, если нет текущего auto result/job для этой версии. Defaults берутся
из активного шаблона конкретного датасета `(architecture, dataset_key)`, а если он не создан - из базового шаблона
сети `(architecture, null)`. После успешного auto training result с MLflow run id создается auto pseudo-markup job
по txt того же датасета. Очередь jobs единая: ручная псевдоразметка имеет приоритет выше ручного обучения,
ручное обучение выше auto псевдоразметки, auto псевдоразметка выше auto обучения. Auto jobs нельзя удалить или двигать
через endpoints очереди; снятие галочки отменяет соответствующие queued/running auto jobs. Если меняется версия
конкретного датасета, активные auto jobs предыдущей версии отменяются только для этого датасета и модели. Failed
auto attempt не ретраится до новой версии или снятия и повторного включения галочки.

Глобальное выключение автоматизации через `PUT /api/v1/automation/enabled` с `enabled=false` отменяет все active
auto jobs: queued rows уходят из очередей, running process получает SIGTERM, временная директория удаляется,
результаты получают `cancelled`, а известный MLflow run помечается как `KILLED`. При повторном включении worker
создает новые jobs по текущим правилам и версиям датасетов, а не восстанавливает старую очередь.

## Алгоритм работы и его особенности

Каталог транзакционно добавляет неизвестные папки MLMarkup и сохраняет `dataset_key`. Класс задаёт четыре канала `kanopus` либо RGB `ortho`. Создание списка сцен сопоставляет полигональный GeoJSON с контурами TIFF выбранного типа и отдаёт имена без расширений без сохранения. Worker запускает training/inference через общую очередь, метрики пишет только `train_pipeline`; одноразовая псевдоразметка и тестовый F1 принимают RGBA для трёхканального checkpoint, читая RGB. Тестовые разметки формируются вне очереди: после смены CRS полигоны восстанавливаются через `make_valid`, допустимый пул сохраняется при тайм-ауте вторичной оптимизации, невозможный минимум получает точный максимум. Одиночный ZIP содержит полный набор либо только TIFF/GeoJSON. Выбранные разметки готовятся максимум восемью потоками и последовательно собираются в `ZIP_STORED`; для каждого датасета допускается одна разметка, коллизии имён отклоняются. JPEG ограничен 300 KiB.
