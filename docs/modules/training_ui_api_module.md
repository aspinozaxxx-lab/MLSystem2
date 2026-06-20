# Модуль training_ui_api

## Назначение

`training_ui_api` — отдельный FastAPI-сервис сайта MLSystem2. Он отдает публичный HTTP API для frontend, хранит UI-данные обучения в Postgres БД/схеме, сканирует `/data/MLMarkup` для актуальных датасетов и классов, управляет очередями training/inference и не выполняет прямой доступ frontend к БД.

## Публичный интерфейс

- `create_app() -> Any` - создает FastAPI-приложение.
- `get_openapi_schema() -> dict[str, Any]` - возвращает OpenAPI-схему сервиса.
- `main() -> None` - запускает `uvicorn` для сервиса.

## Публичные контракты

- `TrainingUIAPIError` - ошибка сервиса.
- `TemplateSource`, `JobType`, `JobSource`, `JobStatus`, `ResultStatus`, `StoredFileKind` - enum-значения API.
- `AppLink`, `AppLinksResponse` - ссылки Grafana/MLflow/MinIO.
- `MLflowExperimentInfo`, `MLflowExperimentCreate` - experiments MLflow.
- `DatasetInfo`, `DatasetListResponse`, `ClassInfo`, `ClassListResponse` - датасеты и классы MLMarkup; `DatasetInfo` содержит `scenes_file`, positive `annotation_file`, optional `hard_negative_annotation_file` и `diagnostics`.
- `ImageFolderInfo`, `ImageFolderListResponse` - папки подготовленных снимков из `MLSYSTEM2_IMAGES_ROOT` с количеством TIFF.
- `ModelInfo`, `ModelListResponse` - публичные модели из `models`.
- `ConfigField`, `ConfigSchema`, `TrainingTemplate`, `TrainingTemplateListResponse`, `TrainingTemplateCreate`, `TrainingTemplateUpdate`, `TrainingTemplateApplyField`, `InferenceTemplate`, `InferenceTemplateListResponse`, `InferenceTemplateCreate`, `InferenceTemplateUpdate`, `InferenceTemplateApplyField` - шаблоны обучения и инференса; `ConfigField` содержит `tooltip`, допустимые границы и optional `recommended_range` для UI-подсказок.
- `StoredFileInfo`, `CustomDatasetInfo` - загруженные файлы и custom datasets.
- `TrainingJobCreate`, `QueueEnabledUpdate`, `QueueControlInfo`, `JobSummary`, `QueueSnapshot`, `JobDetail` - задания и очереди.
- `AutomationEnabledUpdate`, `AutomationRuleUpdate`, `AutomationRuleInfo`, `AutomationSnapshot` - глобальный выключатель и матрица автоматизации `датасет × модель`.
- `TrainingResultInfo`, `PseudoMarkupResultInfo`, `ClassResultsResponse`, `ResultChangeInfo`, `ResultChangesResponse` - результаты обучения, псевдоразметки и последние изменения; активные DTO результата содержат необязательный `job_id` и показывают фактический статус `queued`/`running` связанного задания.
- `POST /api/v1/model-export/triton-zip` - multipart endpoint для сборки zip-архива модели под
  `models-serving-service` и Triton CPU; endpoint возвращает файл и не создает записей в БД.
- `POST /api/v1/results/training/{result_id}/triton-zip` - multipart endpoint для сборки такого же zip-архива
  из `checkpoints/best.pt` успешного результата обучения в MLflow; endpoint возвращает файл и не создает записей в БД.

## Список используемых данным модулем модулей и с какой целью

- `models.api` - получить публичный список поддерживаемых архитектур.
- `mlflow_adapter.api` - получить и создать MLflow experiments, прочитать лучший checkpoint training run и скачать `checkpoints/best.pt` для псевдоразметки; сервис не пишет MLflow-метрики.
- `mlflow_adapter.contracts` - передать публичные DTO создания experiment и summary лучшего checkpoint.
- `settings.contracts` - валидировать YAML-настройки, сформированные для запуска training CLI.

### Автоматизация

Автоматизация хранится в таблицах `automation_controls` и `automation_rules`. Правило задается парой
`dataset_key + architecture` и двумя флагами: `training_enabled` и `pseudo_markup_enabled`. `Custom` не участвует в
автоматизации. `DatasetInfo.version` вычисляется как `git:{commit_sha}` по последнему коммиту папки варианта
MLMarkup, а если git checkout недоступен - как `fs:{mtime_ns}` по filesystem mtime.

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

Сервис читает настройки только из env vars. Frontend авторизуется через cookie-session с пользователем и паролем из тех же env vars, что старый сайт: `MLSYSTEM_FRONTEND_USER`/`MLSYSTEM_FRONTEND_PASSWORD`, либо новые `MLSYSTEM2_TRAINING_UI_USER`/`MLSYSTEM2_TRAINING_UI_PASSWORD`. Postgres доступен только из FastAPI; frontend работает через `/api/v1/*` и может отдаваться тем же сервисом из `MLSYSTEM2_TRAINING_UI_FRONTEND_DIST`. MLMarkup не кэшируется: `/datasets`, `/classes` и `/results/classes` сканируют каталог при каждом запросе, поэтому новая папка появляется без релиза. MLMarkup читается как `класс/вариант`: `DatasetInfo.key` и `DatasetInfo.name` имеют вид `Класс\вариант`, а `ClassInfo.variants` содержит выбираемые варианты. Frontend не выбирает класс целиком. Внутри папки варианта поддерживается один TXT со списком сцен, один ordinary positive GeoJSON и optional `hard_negative.geojson`; `hard_negative.geojson` никогда не выбирается как positive annotation, а неоднозначные наборы обычных GeoJSON возвращают diagnostics вместо случайного выбора. `updated_at` варианта берется из последнего git-коммита, затронувшего папку конкретного варианта в `MLSYSTEM2_MLMARKUP_ROOT`; если каталог не является git checkout, используется filesystem mtime как fallback. Шаблоны обучения хранятся в `training_templates`, шаблоны инференса - в `inference_templates`; базовый шаблон сети имеет `dataset_key=null`, а шаблон конкретного датасета хранит `dataset_key` и `parent_template_id`. В шаблонах обучения остаются только параметры обычного binary tile-training, включая `tile_preparation.positive_factor`, `tile_preparation.hard_negative_factor`, `tile_preparation.background_factor`, `train.pos_weight`, `train.hard_negative_weight`, `train.max_train_batches_per_epoch`, `train.max_val_batches_per_epoch` и nullable `train.max_training_time_sec`; воркеры, prefetch, seed, `train.device=cuda`, task и каналы модели берутся из `settings.yml`. Схема параметров содержит компактный label, подробный `tooltip`, допустимые границы и рекомендуемый диапазон, чтобы frontend мог показывать одинаковые подсказки на странице шаблонов, запуска и просмотра задания. Сервис валидирует, что сумма трех tile factors равна `1`; `hard_negative_factor > 0` допустим без `hard_negative.geojson`, потому что train sampler переносит недостающий hard-negative budget в positive внутри общего marked-бюджета. Шаблоны инференса хранят только параметры геометрической постобработки, соответствующие Geoalert-конфигу: очистку маски, фильтр площади, удаление дыр, `Simplify` и фильтр компактных водных объектов. Ручное изменение ставит `source=manual` и увеличивает `version`, reset возвращает baseline. Кнопка установки поля для всех шаблонов меняет только выбранный ключ во всех существующих шаблонах соответствующего типа. Очередь показывает только `queued` и `running`; completed/cancelled/failed не попадают в таблицу. Фоновый worker берет первый queued job из единой очереди, запускает обучение через публичный CLI `python -m mlsystem2.cli.train --settings ... --run ...` или псевдоразметку через runner `mlsystem2.training_ui_api._pseudo_runner` и обновляет статусы по exit code. Training worker всегда записывает все три tile factors и `train.hard_negative_weight` в `run.yml`, а также передает `hard_negative_annotation_file`, если он найден у встроенного MLMarkup dataset. Training-процесс пишет id созданного MLflow run во временный файл `mlflow_run_id`, поэтому worker обновляет `training_results.mlflow_run_id` еще во время `running`. MLflow-метрики, отчеты и артефакты пишет только `train_pipeline`; UI worker после завершения лишь читает через `mlflow_adapter.api` лучший `val/best_threshold_pixel_f1` и эпоху, чтобы заполнить поля `f1_score` и `epoch` в `training_results`. Для queued inference job типа `pseudo-markup` worker пишет `pseudo_config.yaml` с `inference_backend=pytorch_one_off`, скачивает `checkpoints/best.pt` через `mlflow_adapter.api`, загружает checkpoint через `models.api`, выполняет PyTorch-инференс по txt списку снимков и сохраняет итоговый GeoJSON в `stored_files`. Этот путь не создает Triton model archive, Geoalert pipeline YAML или запись в Triton model repository; ручной Triton export остается отдельными endpoints `triton-zip`. Геометрия скачиваемого GeoJSON репроецируется из CRS снимка в `EPSG:4326`, как в Geoalert inference. Перед записью итогового скачиваемого GeoJSON раннер сливает пересекающиеся и касающиеся полигоны через `unary_union`, а per-scene GeoJSON оставляет диагностическим выводом. Report содержит `inference_backend=pytorch_one_off`, `triton_model=null`, `feature_count_before_merge`, финальный `feature_count`, `postprocess_merge_overlaps=true` и `postprocess_merge_policy=overlap_or_touch`. После обработки или ошибки загрузки checkpoint раннер удаляет ссылки на модель и вызывает `torch.cuda.empty_cache()` при CUDA. Перед инференсом раннер удаляет повторные TIFF, выбранные по строкам txt, и автоматически выбирает профиль постобработки по числу уникальных снимков: `none` для `<=5`, `detail_v2` для `6..50`, `strong` для `>=51`; затем значения из активного шаблона инференса `(architecture, dataset_key)` переопределяют только явно заданные поля. Задание псевдоразметки, созданное от результата обучения, сохраняет в config `mlflow_run_id`, `checkpoint_artifact_path=checkpoints/best.pt`, threshold лучшего checkpoint, `inference_template_id`, `inference_template_config` и при доступном MLflow полный `checkpoint_uri`. `GET /api/v1/results/changes` отдает сначала queued/running jobs, затем последние 20 успешных изменений результатов; активные строки открывают страницу класса. На странице класса обучение и псевдоразметка со статусом `queued`/`running` отображаются в основной таблице `ClassResultsResponse` и открывают страницу параметров задания по `job_id`. При pause/delete running job получает SIGTERM группе процесса; `train_pipeline` завершает MLflow run как `KILLED`, tmp-папка удаляется, job возвращается в `queued` при pause или становится `cancelled` при delete.

Для профиля псевдоразметки `strong` (`>=51` уникальный снимок) не применяется `binary_closing`; маска чистится порогом `48 px`, полигоны меньше `3000 м²` фильтруются, дырки меньше `5000 м²` удаляются, контур упрощается на `15 м`.

На странице запуска обучения frontend выбирает по умолчанию существующий MLflow experiment с максимальным числовым `experiment_id`. Поле `Новое имя experiment` показывается только при выборе пункта `Новый experiment`. Поле ручного `MLflow run name` на странице запуска не используется: worker не передает `--run-name`, если имя не задано, а `train_pipeline` и `mlflow_adapter` формируют имя автоматически по tag `class`.

Для датасета `Реки\main` создан датасетный шаблон инференса `smp_segformer_b2` под вариант 18: `min_area=10000 м²`, `min_hole_area=5000 м²`, `Simplify=15 м` и включенный фильтр компактных объектов (`min_isoperimetric_quotient=0.25`, `max_bbox_ratio=3.5`). Фильтр удаляет компактные полигоны, похожие на пруды и озера, и оставляет вытянутые речные объекты.

Страница `Экспорт модели` отправляет `.pt` checkpoint и имя модели в `POST /api/v1/model-export/triton-zip`.
Backend загружает checkpoint через `models.api.load_checkpoint`, берет threshold только из
`metadata.val_best_threshold` и завершает экспорт ошибкой, если threshold в checkpoint отсутствует. `sample_size`
берется из `metadata.sample_size`; для старых checkpoint без этого поля frontend показывает popup и повторяет
запрос с ручным `sample_size`. Backend экспортирует binary segmentation модель в ONNX с uint8 mask output,
создает `config.pbtxt` с `instance_group KIND_CPU` и возвращает внешний архив `<model_name>_export.zip`. В корне
внешнего архива лежит `export_metadata.json`, pipeline YAML лежит в `pipelines/<model_name>_triton.yaml`, а
чистый архив для `models-serving-service` лежит в `models-serving-service/<model_name>.zip`. Внутренний архив
содержит только каталог `<model_name>` с Triton model repository файлами, поэтому после распаковки проходит
проверка `models-serving-service` по наличию каталога модели и туда не попадают pipeline или metadata.
На странице результатов успешная строка обучения имеет кнопку `zip`: frontend предлагает имя
`{имя geojson-разметки без расширения}_kanopus`, отправляет его в
`POST /api/v1/results/training/{result_id}/triton-zip`, а backend скачивает `checkpoints/best.pt` через
`mlflow_adapter.api.download_run_artifact` и использует тот же сборщик архива.
