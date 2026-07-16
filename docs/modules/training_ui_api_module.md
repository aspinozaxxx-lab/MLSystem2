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
- `DatasetInfo`, `DatasetListResponse`, `ClassInfo`, `ClassListResponse` - совместимое представление управляемого каталога; `DatasetInfo` содержит класс, подкласс, метрику, тип снимков, источник, его состояние, `scenes_file`, positive `annotation_file`, optional `hard_negative_annotation_file` и `diagnostics`.
- `DatasetCatalogInfo`, `DatasetSourceInfo`, `ImageTypeInfo`, `DatasetSubclassInfo`, `DatasetClassCreate`, `DatasetClassUpdate`, `DatasetPrimarySubclassUpdate`, `DatasetSubclassCreate`, `DatasetSubclassUpdate`, `ManagedDatasetCreate`, `ManagedDatasetUpdate` - чтение и изменение классов, подклассов и датасетов без удаления.
- `ImageFolderInfo`, `ImageFolderListResponse` - папки подготовленных снимков из `MLSYSTEM2_IMAGES_ROOT` с количеством TIFF.
- `ModelInfo`, `ModelListResponse` - публичные модели из `models`.
- `ConfigField`, `ConfigSchema`, `TrainingTemplate`, `TrainingTemplateListResponse`, `TrainingTemplateCreate`, `TrainingTemplateUpdate`, `TrainingTemplateApplyField`, `InferenceTemplate`, `InferenceTemplateListResponse`, `InferenceTemplateCreate`, `InferenceTemplateUpdate`, `InferenceTemplateApplyField` - шаблоны обучения и инференса; `ConfigField` содержит `tooltip`, допустимые границы и optional `recommended_range` для UI-подсказок.
- `StoredFileInfo`, `CustomDatasetInfo` - загруженные файлы и custom datasets.
- `TrainingJobCreate`, `QueueEnabledUpdate`, `QueueControlInfo`, `JobSummary`, `QueueSnapshot`, `JobDetail` - задания и очереди.
- `AutomationEnabledUpdate`, `AutomationRuleUpdate`, `AutomationRuleInfo`, `AutomationSnapshot` - глобальный выключатель и матрица автоматизации `датасет × модель`.
- `TrainingResultInfo`, `TrainingResultTestF1Info`, `PrimaryTestSampleInfo`, `PseudoMarkupResultInfo`, `ClassResultsResponse`, `ResultClassInfo`, `ResultVariantInfo`, `ResultClassListResponse`, `ResultChangeInfo`, `ResultChangesResponse` - результаты обучения, отдельный тестовый F1 сети, основная разметка и карточки классов; активные DTO содержат `job_id` и прогресс связанного задания.
- `TrainingResultExportItem`, `TrainingResultBatchExportRequest` - JSON-запрос массового экспорта выбранных успешных training results.
- `MarkupExportRequest`, `MarkupExportTileInfo`, `MarkupExportInfo` - запрос и описание временного набора тестовой разметки с тайлами, превью, сводкой цель/факт и сроком хранения.
- `TestSampleCreate`, `TestSampleUpdate`, `TestSampleTileUpdate`, `TestSampleOptimizeRequest`, `TestSamplePrimaryUpdate`, `TestSampleEvaluationPreviewRequest`, `TestSampleDownloadRequest` - создание и атомарное сохранение постоянной разметки, совместимые точечные изменения, ограничения оптимизации, запросы оценки и скачивания черновика.
- `TestSampleMetric`, `TestSampleEvaluationInfo`, `TestSampleSummary`, `TestSampleVariantGroup`, `TestSampleClassGroup`, `TestSampleCatalogResponse`, `TestSampleTileInfo`, `TestSampleDetail`, `TestSampleDraftPreview` - метрики, каталог, редакторское описание и незаписываемый результат чернового расчёта тестовых разметок.
- `TestSampleBatchItemCreate`, `TestSampleBatchCreate`, `TestSampleBatchItemInfo`, `TestSampleBatchInfo` - запрос и прогресс группового создания; последний запуск хранит применённые настройки формы.
- `GET /api/v1/bootstrap` - агрегированный стартовый endpoint для frontend; старые catalog/template endpoints остаются рабочими.
- `GET /api/v1/dataset-catalog` и `POST /api/v1/dataset-catalog/sync` - иерархия редактора и явная идемпотентная синхронизация с MLMarkup.
- `POST|PATCH /api/v1/dataset-classes`, `PUT /api/v1/dataset-classes/{class_key}/primary-subclass`, `POST|PATCH /api/v1/dataset-subclasses`, `POST|PATCH /api/v1/managed-datasets` - создание и редактирование каталога; удаляющих endpoints нет.
  Назначение занятого источника пустому подклассу переносит его датасет, а смена источника существующего датасета безопасно обменивает источники и увеличивает ревизии обеих сущностей.
- `POST /api/v1/model-export/triton-zip` - multipart endpoint для сборки zip-архива модели под
  `models-serving-service` и Triton CPU; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/{result_id}/triton-zip` - multipart endpoint для сборки такого же zip-архива
  из `checkpoints/best.pt` успешного результата обучения в MLflow; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/triton-zip` - JSON endpoint для сборки общего zip-архива нескольких успешных
  результатов обучения; каждая модель собирается тем же кодом, что одиночный экспорт результата, endpoint
  возвращает файл и не создает записей в БД.
- `POST /api/v1/markup-export` - синхронно формирует временный набор тестовой разметки для варианта MLMarkup.
- `GET /api/v1/markup-export/{export_id}/tiles/{tile_index}/preview` - возвращает PNG-превью тайла с контуром маски.
- `GET /api/v1/markup-export/{export_id}/download` - возвращает плоский ZIP сформированного набора.
- `GET /api/v1/test-samples` и `POST /api/v1/test-samples` - иерархический каталог и создание постоянной тестовой разметки.
- `GET|PATCH|DELETE /api/v1/test-samples/{sample_id}` - просмотр, атомарное сохранение имени, основного статуса и полного состава либо удаление разметки.
- `PATCH /api/v1/test-samples/{sample_id}/tiles/{tile_index}` - включает или выключает тайл.
- `POST /api/v1/test-samples/{sample_id}/evaluate` - пересчитывает пиксельный и объектный F1.
- `POST /api/v1/test-samples/{sample_id}/optimize` - подбирает состав из всех тайлов по основной метрике класса; request-поле старого клиента принимается, но не меняет выбор метрики.
- `POST /api/v1/test-samples/{sample_id}/evaluate-preview` и `POST /api/v1/test-samples/{sample_id}/optimize-preview` - рассчитывают F1 или оптимальный состав черновика без записи в БД.
- `GET /api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview` и `GET /api/v1/test-samples/{sample_id}/download` - постоянное превью и ZIP сохранённых включённых тайлов.
- `POST /api/v1/test-samples/{sample_id}/download` - ZIP явно выбранных тайлов текущего черновика без изменения разметки в БД.
- `POST /api/v1/test-sample-batches`, `GET /api/v1/test-sample-batches/latest` и `GET /api/v1/test-sample-batches/{batch_id}` - запуск и прогресс последовательного группового создания.
- `PUT /api/v1/test-samples/{sample_id}/primary` - совместимо назначает, заменяет или снимает основную разметку точного варианта.
- `GET /api/v1/test-samples/primary/download` - ZIP основных разметок с отдельной папкой `Класс_вариант`.
- `POST /api/v1/results/classes/{class_key}/test-f1` - ставит в inference-очередь отсутствующие, ошибочные и устаревшие оценки успешных сетей варианта.

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

Источник истины для каталога — таблицы `dataset_classes`, `dataset_subclasses` и `datasets`. При старте и перед
чтением `/datasets`, `/classes` или редактора сервис транзакционно импортирует неизвестные исторические ключи и
новые папки MLMarkup, но не удаляет записи, не следует за переименованием папки и не затирает ручные имя,
метрику, тип снимков, источник или основной подкласс. `main` назначается основным только до ручного выбора.
Путь MLMarkup хранится относительно разрешённого корня и проверяется после `resolve`; снимки берутся из всего
`MLSYSTEM2_IMAGES_ROOT` или выбранной верхнеуровневой папки. Первый импорт сохраняет используемые
`dataset_key`, новые сущности получают UUID. Training worker передаёт выбранные
`dataset.images_dir` и `train.quality_metric`, а `training_results` сохраняет snapshot метрики; best checkpoint
читается из MLflow по `val/quality_f1` с pixel fallback для старых запусков.

Сервис читает настройки только из env vars. Frontend авторизуется через cookie-session с пользователем и паролем из тех же env vars, что старый сайт: `MLSYSTEM_FRONTEND_USER`/`MLSYSTEM_FRONTEND_PASSWORD`, либо новые `MLSYSTEM2_TRAINING_UI_USER`/`MLSYSTEM2_TRAINING_UI_PASSWORD`. Postgres доступен только из FastAPI; frontend работает через `/api/v1/*` и может отдаваться тем же сервисом из `MLSYSTEM2_TRAINING_UI_FRONTEND_DIST`. Перед чтением `/datasets`, `/classes` и `/results/classes` сервис синхронизирует неизвестные папки MLMarkup с каталогом БД, поэтому новая папка появляется без релиза. Источник MLMarkup читается как `класс/вариант`: `DatasetInfo.name` имеет вид `Класс\вариант`, старые `DatasetInfo.key` сохраняют этот формат, а новые получают UUID; `ClassInfo.variants` содержит выбираемые варианты. Frontend не выбирает класс целиком. Внутри папки варианта поддерживается один TXT со списком сцен, один ordinary positive GeoJSON и optional `hard_negative.geojson`; `hard_negative.geojson` никогда не выбирается как positive annotation, а неоднозначные наборы обычных GeoJSON возвращают diagnostics вместо случайного выбора. `updated_at` варианта берется из последнего git-коммита, затронувшего папку конкретного варианта в `MLSYSTEM2_MLMARKUP_ROOT`; если каталог не является git checkout, используется filesystem mtime как fallback. Шаблоны обучения хранятся в `training_templates`, шаблоны инференса - в `inference_templates`; базовый шаблон сети имеет `dataset_key=null`, а шаблон конкретного датасета хранит `dataset_key` и `parent_template_id`. В шаблонах обучения остаются только параметры обычного binary tile-training, включая `tile_preparation.positive_factor`, `tile_preparation.hard_negative_factor`, `tile_preparation.background_factor`, `train.pos_weight`, `train.hard_negative_weight`, `train.max_train_batches_per_epoch`, `train.max_val_batches_per_epoch` и nullable `train.max_training_time_sec`; воркеры, prefetch, seed, `train.device=cuda`, task и каналы модели берутся из `settings.yml`. Схема параметров содержит компактный label, подробный `tooltip`, допустимые границы и рекомендуемый диапазон, чтобы frontend мог показывать одинаковые подсказки на странице шаблонов, запуска и просмотра задания. Сервис валидирует, что сумма трех tile factors равна `1`; `hard_negative_factor > 0` допустим без `hard_negative.geojson`, потому что train sampler переносит недостающий hard-negative budget в positive внутри общего marked-бюджета. `train.hard_negative_weight` описывается как pixel-level множитель loss только для pixels hard-negative supervision mask, а не как вес всего tile. Шаблоны инференса хранят только параметры геометрической постобработки, соответствующие Geoalert-конфигу: очистку маски, фильтр площади, удаление дыр, `Simplify` и фильтр компактных водных объектов. Ручное изменение ставит `source=manual` и увеличивает `version`, reset возвращает baseline. Кнопка установки поля для всех шаблонов меняет только выбранный ключ во всех существующих шаблонах соответствующего типа. Очередь показывает только `queued` и `running`; completed/cancelled/failed не попадают в таблицу. Фоновый worker берет первый queued job из единой очереди, запускает обучение через публичный CLI `python -m mlsystem2.cli.train --settings ... --run ...` или псевдоразметку через runner `mlsystem2.training_ui_api._pseudo_runner` и обновляет статусы по exit code. Training worker всегда записывает все три tile factors и `train.hard_negative_weight` в `run.yml`, а также передает `hard_negative_annotation_file`, если он найден у встроенного MLMarkup dataset. Training-процесс пишет id созданного MLflow run во временный файл `mlflow_run_id`, поэтому worker обновляет `training_results.mlflow_run_id` еще во время `running`. MLflow-метрики, отчеты и артефакты пишет только `train_pipeline`; UI worker после завершения лишь читает через `mlflow_adapter.api` лучший `val/quality_f1` и эпоху, используя пиксельный fallback для старых запусков, чтобы заполнить поля `f1_score` и `epoch` в `training_results`. Для queued inference job типа `pseudo-markup` worker пишет `pseudo_config.yaml` с `inference_backend=pytorch_one_off`, скачивает `checkpoints/best.pt` через `mlflow_adapter.api`, загружает checkpoint через `models.api`, выполняет PyTorch-инференс по txt списку снимков и сохраняет итоговый GeoJSON в `stored_files`. Этот путь не создает Triton model archive, Geoalert pipeline YAML или запись в Triton model repository; ручной Triton export остается отдельными endpoints `triton-zip`. Геометрия скачиваемого GeoJSON репроецируется из CRS снимка в `EPSG:4326`, как в Geoalert inference. Перед записью итогового скачиваемого GeoJSON раннер сливает пересекающиеся и касающиеся полигоны через `unary_union`, а per-scene GeoJSON оставляет диагностическим выводом. Report содержит `inference_backend=pytorch_one_off`, `triton_model=null`, `feature_count_before_merge`, финальный `feature_count`, `postprocess_merge_overlaps=true` и `postprocess_merge_policy=overlap_or_touch`. После обработки или ошибки загрузки checkpoint раннер удаляет ссылки на модель и вызывает `torch.cuda.empty_cache()` при CUDA. Перед инференсом раннер удаляет повторные TIFF, выбранные по строкам txt, и автоматически выбирает профиль постобработки по числу уникальных снимков: `none` для `<=5`, `detail_v2` для `6..50`, `strong` для `>=51`; затем значения из активного шаблона инференса `(architecture, dataset_key)` переопределяют только явно заданные поля. Задание псевдоразметки, созданное от результата обучения, сохраняет в config `mlflow_run_id`, `checkpoint_artifact_path=checkpoints/best.pt`, threshold лучшего checkpoint, `inference_template_id`, `inference_template_config` и при доступном MLflow полный `checkpoint_uri`. `GET /api/v1/results/changes` отдает сначала queued/running jobs, затем последние 20 успешных изменений результатов; активные строки открывают страницу класса. На странице класса обучение и псевдоразметка со статусом `queued`/`running` отображаются в основной таблице `ClassResultsResponse` и открывают страницу параметров задания по `job_id`. При pause/delete running job получает SIGTERM группе процесса; `train_pipeline` завершает MLflow run как `KILLED`, tmp-папка удаляется, job возвращается в `queued` при pause или становится `cancelled` при delete.

Для профиля псевдоразметки `strong` (`>=51` уникальный снимок) не применяется `binary_closing`; маска чистится порогом `48 px`, полигоны меньше `3000 м²` фильтруются, дырки меньше `5000 м²` удаляются, контур упрощается на `15 м`.

На странице запуска обучения frontend выбирает по умолчанию существующий MLflow experiment с максимальным числовым `experiment_id`. Поле `Новое имя experiment` показывается только при выборе пункта `Новый experiment`. Поле ручного `MLflow run name` на странице запуска не используется: worker не передает `--run-name`, если имя не задано, а `train_pipeline` и `mlflow_adapter` формируют имя автоматически по tag `class`.

Для датасета `Реки\main` создан датасетный шаблон инференса `smp_segformer_b2` под вариант 18: `min_area=10000 м²`, `min_hole_area=5000 м²`, `Simplify=15 м` и включенный фильтр компактных объектов (`min_isoperimetric_quotient=0.25`, `max_bbox_ratio=3.5`). Фильтр удаляет компактные полигоны, похожие на пруды и озера, и оставляет вытянутые речные объекты.

Endpoint `POST /api/v1/model-export/triton-zip` принимает `.pt` checkpoint и имя модели.
Backend загружает checkpoint через `models.api.load_checkpoint`, берет threshold только из
`metadata.val_best_threshold` и завершает экспорт ошибкой, если threshold в checkpoint отсутствует. `sample_size`
берется из `metadata.sample_size`, а если его нет, запрос может передать ручной `sample_size`. Backend экспортирует binary segmentation модель в ONNX с uint8 mask output,
совместимый со старым Triton CPU service (`opset 17`, `IR version 8`), создает `config.pbtxt` с
динамическим batch для output mask (`dims: [ -1, 1, -1, -1 ]`), `instance_group KIND_CPU` и возвращает внешний архив `<model_name>_export.zip`. В корне
внешнего архива лежит `export_metadata.json`, pipeline YAML лежит в `pipelines/<model_name>_triton.yaml`, а
чистый архив для `models-serving-service` лежит в `models-serving-service/<model_name>.zip`. Внутренний архив
содержит только каталог `<model_name>` с Triton model repository файлами, поэтому после распаковки проходит
проверка `models-serving-service` по наличию каталога модели и туда не попадают pipeline или metadata.
Страница `Экспорт моделей` берет варианты `Класс\вариант` из bootstrap, для каждого варианта читает последний
успешный training result через `/results/classes/{class_key}`, по умолчанию отмечает `main`, позволяет задать имя
и optional `sample_size` на модель и отправляет выбранные строки в `POST /api/v1/results/training/triton-zip`.
Общий архив содержит `models-serving-service/<model_name>.zip`, `pipelines/<model_name>_triton.yaml`,
`metadata/<model_name>_export_metadata.json` и корневой `export_metadata.json`.

Алгоритм нарезки тестовой разметки не входит в training/inference-очередь, MLflow, S3 или конвейер обучения.
По TXT выбранного варианта он сопоставляет сцены с TIFF только внутри `MLSYSTEM2_IMAGES_ROOT`, исключает окна за
границами растра, с невалидной `dataset_mask`, пикселями без данных или полностью чёрными пикселями и детерминированно выбирает
заданное число тайлов через `scipy.optimize.milp`. Приоритеты выбора: максимальное число родительских папок,
максимальное число исходных TIFF и минимальное отклонение от целевого числа объектов. Кандидаты строятся также
вдоль протяжённой геометрии: один объект разрешено использовать в разных тайлах и считать в каждом отдельно;
пересечения запрещены, а касание разрешается только как явно отмеченный запасной режим.
Для тайла создаются `tile_001.tif`, `tile_001.geojson`, `tile_001_mask.png` и `tile_001_preview.png`: COG TIFF без
ресэмплинга, бинарная маска и превью, а геометрии обрезанного GeoJSON с исходными свойствами и CRS нормализуются
до `MultiPolygon` для открытия в QGIS одним слоем. Архив именуется по русскому имени класса, например
`вырубки_test_markup.zip`. Совместимый временный экспорт хранится один час в `scratch_root/markup-exports`.
Постоянные тестовые разметки хранят метаданные в `test_samples` и `test_sample_tiles`, а файлы без TTL — в
`stored_files_root/test-samples/{uuid}`. Выключение тайла сохраняет файлы, исключает его из ZIP и помечает прежние
метрики устаревшими. Одиночный и общий ZIP основных разметок сортируют включённые внутренние индексы и независимо
перенумеровывают их как `tile001..tileNNN`. Вместе с TIFF, GeoJSON и маской архив получает полноразмерные
`rgb/nrg/ngr` JPEG с жёлтым двухпиксельным контуром и без него. JPEG строятся из каналов
`RED, GRN, BLU, NIR` только при скачивании, а двоичный подбор максимального качества `1..95` обеспечивает
жёсткий предел `300 KiB` на файл. Постоянные файлы и технический `/markup-export` сохраняют прежний формат.
Редактор передаёт текущие индексы тайлов в `POST` скачивания; сборка использует их только для временного ZIP и
не меняет сохранённые флаги, ревизию состава или метрики.
Пиксельный F1 считается по TP/FP/FN растрированных масок, объектный — по максимальному
взаимно-однозначному сопоставлению с `IoU ≥ 0,5`. Источник — последняя успешная псевдоразметка, у которой
`class_key` и входной `dataset_key` точно равны ключу варианта разметки; новый подходящий результат запускает
автоматический пересчёт. Редактор рассчитывает оптимизацию и F1 без записи, а одним `PATCH` сохраняет имя,
основной статус и состав. Оптимизатор рассматривает все тайлы независимо от текущего состояния, максимизирует
агрегированный F1 и при равенстве предпочитает территории, число объектов, исходные снимки и меньший состав.
Групповой запуск хранится в `test_sample_batches` и `test_sample_batch_items` и обрабатывается отдельным
последовательным исполнителем нарезки, а не очередью обучения или инференса. Для итоговых `N` тайлов и минимума
`M` объектов и диапазона `Nmin..Nmax` он ищет максимально достижимый непересекающийся пул от `3 × Nmax` до
`Nmin`, целится в `3M` появлений объектов и после оптимизации сохраняет от `Nmin` до `Nmax` тайлов включёнными.
Неуспешная строка не оставляет разметку или файлы, а после перезапуска текущая строка запускается повторно.
Последний запуск возвращает размер, минимум и максимум тайлов, минимумы объектов и метрики как следующие
значения формы по умолчанию. Если старый клиент не передал `min_image_count`, минимум принимается равным
`image_count`, сохраняя прежний режим точного числа тайлов.

Флаг `test_samples.is_primary` уникален по точному `dataset_key`. Общий ZIP включает только основные разметки и
только включённые тайлы. Таблица `training_result_test_metrics` хранит пиксельные и объектовые TP/FP/FN каждой успешной сети.
Задание `test_sample_f1` использует общую inference-очередь и тот же `_pseudo_runner`, но запускает checkpoint
непосредственно на TIFF основной разметки: каждый тайл обрабатывается независимо, применяется threshold checkpoint
и актуальный inference-шаблон, после чего маска сравнивается с `tile_NNN_mask.png`. Базовый профиль
`none/detail_v2/strong` берётся из псевдоразметки, которой оценена разметка, либо из числа снимков точного датасета.
Малые или компактные компоненты внутри тайла фильтруются штатно, а касающиеся границы TIFF фрагменты сохраняются,
поскольку их полная геометрия находится за пределами тайла. Результат применяется только при совпадении основной
разметки и ревизии состава; смена разметки, состава, профиля, эффективного шаблона или версии оценщика делает его
устаревшим. Новая успешная сеть и изменение основной разметки ставят оценки в очередь автоматически; при старте,
смене профиля или inference-шаблона отсутствующие и устаревшие оценки восстанавливаются идемпотентно, а ошибка той
же ревизии не повторяется циклически. Этот расчёт не пишет данные в MLflow и не
изменяет `train_pipeline`.
На странице результатов успешная строка обучения имеет кнопку `zip`: frontend предлагает имя
`{имя geojson-разметки без расширения}_kanopus`, отправляет его в
`POST /api/v1/results/training/{result_id}/triton-zip`, а backend скачивает `checkpoints/best.pt` через
`mlflow_adapter.api.download_run_artifact` и использует тот же сборщик архива.
