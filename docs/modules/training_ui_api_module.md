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
- `DatasetEditorDatasetInfo`, `DatasetEditorObjectType`, `DatasetEditorDatasetListResponse`, `DatasetEditorSceneInfo`, `DatasetEditorSceneListResponse`, `DatasetEditorSceneDetail` - каталог per-image датасетов и сцен с task/schema, class counts, source status, revision и `valid_data_footprint` сцены.
- `DatasetEditorRebuildPreview`, `DatasetEditorRebuildRequest`, `DatasetEditorRebuildResult`, `DatasetEditorRebuildChange` - preview token, source/local changes, конфликты и атомарная пересборка `merge|replace`.
- `DatasetEditorRasterFolderInfo`, `DatasetEditorRasterInfo`, `DatasetEditorRasterBrowserResponse` - прямые папки и TIFF из разрешённого server-side каталога.
- `DatasetEditorAddScenesRequest`, `DatasetEditorSaveSceneRequest`, `DatasetEditorPublishSceneRequest`, `DatasetEditorPublishRequest`, `DatasetEditorDeleteSceneRequest`, `DatasetEditorMutationResult`, `DatasetEditorPublicationInfo` - одиночные и атомарные batch optimistic-lock мутации и статус публикации commit SHA.
- `GET /api/v1/bootstrap` - агрегированный стартовый endpoint для frontend; старые catalog/template endpoints остаются рабочими.
- `GET /api/v1/dataset-catalog` и `POST /api/v1/dataset-catalog/sync` - иерархия редактора и явная идемпотентная синхронизация с MLMarkup.
- `POST|PATCH /api/v1/dataset-classes`, `PUT /api/v1/dataset-classes/{class_key}/primary-dataset`, `POST|PATCH /api/v1/managed-datasets` - создание и редактирование каталога. Назначение занятого источника переносит существующий датасет с сохранением его ключа, а смена источника безопасно обменивает источники и увеличивает ревизии обеих сущностей.
- `DELETE /api/v1/dataset-editor/datasets/{dataset_key}` - удалить папку датасета Git-коммитом, мягко архивировать его строку в Postgres и снять назначение основным; задания, результаты и MLflow не удаляются, при активном задании операция отклоняется.
- `GET /api/v1/dataset-editor/datasets` и `GET /api/v1/dataset-editor/datasets/{dataset_key}/scenes[/{annotation_name}]` - список редактируемых датасетов, сцены и GeoJSON.
- `GET /api/v1/dataset-editor/datasets/{dataset_key}/rasters` и `GET /api/v1/dataset-editor/datasets/{dataset_key}/raster/{image_path}` - серверный выбор снимков и авторизованный TIFF с HTTP Range.
- `POST|PUT /api/v1/dataset-editor/datasets/{dataset_key}/scenes`, `PUT|DELETE /api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}` - добавить TIFF/папку, атомарно опубликовать несколько GeoJSON, совместимо сохранить один GeoJSON или удалить сцену.
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
- `POST /api/v1/test-samples/{sample_id}/evaluate` - пересчитывает пиксельный и объектный F1.
- `POST /api/v1/test-samples/{sample_id}/optimize` - подбирает состав из всех тайлов по основной метрике класса; request-поле старого клиента принимается, но не меняет выбор метрики.
- `POST /api/v1/test-samples/{sample_id}/evaluate-preview` и `POST /api/v1/test-samples/{sample_id}/optimize-preview` - рассчитывают F1 или оптимальный состав черновика без записи в БД.
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview` и `GET /api/v1/test-samples/{sample_id}/download` - постоянное превью и ZIP сохранённых включённых тайлов.
- `POST /api/v1/test-samples/{sample_id}/download` - ZIP явно выбранных тайлов текущего черновика без изменения разметки в БД; флаг `include_previews` оставляет полный состав либо только TIFF и GeoJSON.
- `POST /api/v1/test-samples/download` - несжатый ZIP явно выбранных сохранённых разметок, не более одной на датасет; до восьми разметок готовятся параллельно, каждая в папке `<класс>_<датасет>`.
- `POST /api/v1/test-sample-batches`, `GET /api/v1/test-sample-batches/latest` и `GET /api/v1/test-sample-batches/{batch_id}` - запуск и прогресс последовательного группового создания.
- `PUT /api/v1/test-samples/{sample_id}/primary` - совместимо назначает, заменяет или снимает основную разметку точного датасета.
- `GET /api/v1/results/datasets/{dataset_key}`, `POST /api/v1/results/datasets/{dataset_key}/pseudo-markup` и `POST /api/v1/results/datasets/{dataset_key}/test-f1` - результаты датасета, ручная псевдоразметка и постановка недостающих либо устаревших оценок в inference-очередь.
- `POST /api/v1/results/training/{result_id}/primary` - назначение успешной сети основной для её класса; этот выбор используется QGIS, групповым экспортом и публичным F1.
- `GET /api/v1/pseudolabel/classes`, `POST /api/v1/pseudolabel/jobs`, `GET|DELETE /api/v1/pseudolabel/jobs/{job_id}` и `GET /api/v1/pseudolabel/jobs/{job_id}/result` - серверное распознавание AOI без передачи клиентских растров; полный контракт описан в `docs/pseudolabel_api.md`.

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

Worker перед dispatch очередей вызывает синхронизацию автоматизации. При включенном глобальном switch он создает
auto training job для текущей версии датасета, если нет текущего auto result/job для этой версии. Defaults берутся
из активного шаблона конкретного датасета `(architecture, dataset_key)`, а если он не создан - из базового шаблона
сети `(architecture, null)`. После успешного auto training result с MLflow run id создается auto pseudo-markup job;
для per-image датасета временный TXT формируется из сопоставленных TIFF. Очередь jobs единая: ручная псевдоразметка имеет приоритет выше ручного обучения,
ручное обучение выше auto псевдоразметки, auto псевдоразметка выше auto обучения. Auto jobs нельзя удалить или двигать
через endpoints очереди; снятие галочки отменяет соответствующие queued/running auto jobs. Если меняется версия
конкретного датасета, активные auto jobs предыдущей версии отменяются только для этого датасета и модели. Failed
auto attempt не ретраится до новой версии или снятия и повторного включения галочки.

Глобальное выключение автоматизации через `PUT /api/v1/automation/enabled` с `enabled=false` отменяет все active
auto jobs: queued rows уходят из очередей, running process получает SIGTERM, временная директория удаляется,
результаты получают `cancelled`, а известный MLflow run помечается как `KILLED`. При повторном включении worker
создает новые jobs по текущим правилам и версиям датасетов, а не восстанавливает старую очередь.

## Алгоритм работы и его особенности

Каталог различает legacy по TXT, per-image binary по отсутствию TXT и manifest-backed `per_image_multiclass`; пустой per-image набор доступен редактору, но не обучению. Worker копирует GeoJSON и manifest в immutable snapshot, автоматически фиксирует multiclass task, `output_channels=N+1`, class balance и совместимый loss. Шаблоны с входом `768`, в которых context ещё не был задан, получают `tile_preparation.context=128`; явный `0` сохраняется. Редактор считает role/class/provenance системными полями, но разрешает явную переклассификацию, публикует batch под Git-lock и возвращает `409` при optimistic-lock конфликте. Список снимков показывает цветные счётчики каждого смыслового типа и hard negative из уже рассчитанных `class_counts`. Удаление датасета проверяет активные jobs, удаляет только его Git-дерево и мягко архивирует каталог без очистки исторических таблиц. Combined builder читает обе source-папки целиком, чинит/пропускает invalid geometry с отчётом, применяет priority, вычитает positive из hard negative, ищет TIFF по valid-data footprint и создаёт стабильные feature ID/origin-key. Rebuild preview фиксирует filesystem и Git trees; merge сохраняет ручную версию конфликтующего объекта, replace полностью заменяет per-image файлы, после чего один commit публикует manifest и все GeoJSON. Нативная псевдоразметка, F1 и AOI читают полный тайл и записывают только центр согласно context; старые результаты без context используют `0`. Экспорт берёт context из metadata нового checkpoint либо ручного override старого. Псевдоразметка, F1 и AOI используют нативный checkpoint либо проверенный ZIP `external_torchscript`; multiclass инференс, метрики и экспорт сохраняют class schema. Binary ABI и legacy DTO остаются совместимыми. Live-каталог редактор не изменяет.
