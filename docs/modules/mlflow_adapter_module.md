# Модуль mlflow_adapter

## Назначение

`mlflow_adapter` изолирует работу с MLflow от остальных модулей: открывает запуск, пишет отчеты, метрики, артефакты и завершает запуск. Модуль не принимает решений о качестве модели и не управляет обучением.

## Публичный интерфейс

- `list_experiments(tracking_uri: str) -> list[MLflowExperiment]` - возвращает доступные experiments из указанного MLflow tracking URI.
- `create_experiment(request: MLflowExperimentRequest) -> MLflowExperiment` - создает experiment или возвращает существующий с тем же именем.
- `get_best_training_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint | None` - читает F1 выбранного критерия (`val/object_f1`, `val/macro_pixel_f1` либо `val/best_threshold_pixel_f1`), сохраняет приоритет исторической `val/quality_f1` и возвращает эпоху, F1, threshold и ссылку на `checkpoints/best.pt`.
- `get_usable_training_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint | None` - дополнительно требует статус MLflow `FINISHED`, threshold и фактический артефакт `checkpoints/best.pt`; используется при выборе модели для нового инференса.
- `get_finished_run_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> MLflowRunArtifactInfo | None` - возвращает точный файловый артефакт только завершённого запуска без требования training-метрик.
- `get_training_epoch_progress(tracking_uri: str, run_id: str) -> MLflowTrainingProgress` - возвращает число завершённых эпох по истории `train/epoch_time_sec`.
- `download_run_artifact(tracking_uri: str, run_id: str, artifact_path: str, dst_dir: str | Path) -> MLflowDownloadedArtifact` - скачивает артефакт запуска в локальную рабочую папку вызывающего модуля.
- `start_run(request: MLflowStartRunRequest) -> MLflowRunRef` - создает или отключает MLflow run.
- `log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None` - пишет отчет подготовки датасета.
- `log_dataset_artifacts(run: MLflowRunRef, files: dict[str, str | Path]) -> None` - пишет исходные txt/geojson файлы датасета в папку `dataset` артефактов MLflow.
- `log_tile_preparation(run: MLflowRunRef, report: dict[str, object]) -> None` - пишет отчет подготовки тайлов.
- `log_run_config(run: MLflowRunRef, config_path: str | Path) -> None` - пишет YAML-конфиг запуска.
- `log_training_epoch(run: MLflowRunRef, metrics: EpochMetrics) -> None` - пишет перечисленные ниже loss, learning rate, pixel, optional object и threshold метрики одной эпохи и tag `quality_metric`.
- `log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None` - пишет итоговые `train/epochs_total`, `train/training_time_sec`, `train/best_quality_f1`, `train/best_threshold_pixel_f1`.
- `log_metric_descriptions(run: MLflowRunRef) -> None` — добавляет русский справочник уже записанных метрик, включая завершённые запуски; не пишет числовых метрик, не открывает запуск и не меняет его статус. Использует только публичный клиент MLflow.
- `log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None` - пишет историю обучения и checkpoint-файлы.
- `log_timing_report(run: MLflowRunRef, report: TimingReport) -> None` - пишет отчет времени выполнения.
- `log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None` - пишет итоговый отчет конвейера.
- `end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None` - завершает MLflow run.
- `mark_run_killed(tracking_uri: str, run_id: str) -> None` - переводит активный запуск в `KILLED`.

## Публичные контракты

- `MLflowAdapterError` - ошибка адаптера MLflow.
- `MLflowExperiment` - поля `experiment_id`, `name`, `lifecycle_stage`.
- `MLflowExperimentRequest` - поля `tracking_uri`, `name`.
- `MLflowRunStatus` - статусы `FINISHED`, `FAILED`, `KILLED`.
- `MLflowStartRunRequest` - поля `enabled`, `tracking_uri`, `experiment_name`, `dataset`, `run_name`, `tags`.
- `MLflowRunRef` - поля `run_id`, `experiment_name`, `tracking_uri`, `active`.
- `MLflowArtifactRef` - ссылка на артефакт MLflow.
- `MLflowBestCheckpoint` - поля `tracking_uri`, `run_id`, `metric_name`, `quality_metric`, `f1_score`, `epoch`, `artifact_path`, `artifact_uri`, `threshold`.
- `MLflowDownloadedArtifact` - поля `run_id`, `artifact_path`, `local_path`.
- `MLflowRunArtifactInfo` - поля `tracking_uri`, `run_id`, `artifact_path`, `artifact_uri` произвольного файла завершённого запуска.
- `MLflowTrainingProgress` - поле `completed_epochs`.

## Список используемых данным модулем модулей и с какой целью

- `mlflow` - прочитать/создать experiments, создать run, записать metrics/artifacts, скачать artifact запуска и завершить run.
- `dataset_preparing.contracts` - тип отчета подготовки датасета.
- `train.contracts` - типы `EpochMetrics` и `TrainResult`.
- `train_pipeline.contracts` - типы итоговых отчетов и timing report.

### Публикуемые MLflow-метрики

Список MLflow-метрик является частью архитектурного контракта модуля. Любое добавление, удаление или переименование метрики требует предварительного изменения этого документа. Каждый публичный API, который пишет MLflow-метрики, обязан явно перечислять имена записываемых метрик в разделе публичного интерфейса.

`log_training_epoch` пишет только epoch-метрики с `step=metrics.epoch`:

- `train/loss`
- `val/loss`
- `val/best_threshold`
- `val/best_pixel_threshold`
- `val/best_threshold_pixel_f1`
- `val/pixel_f1`
- `val/pixel_precision`
- `val/pixel_recall`
- `val/best_threshold_precision`
- `val/best_threshold_recall`
- `val/best_threshold_object_f1` — только при наличии object validation
- `val/best_threshold_object_precision` — только при наличии object validation
- `val/best_threshold_object_recall` — только при наличии object validation
- `val/object_f1` — только при наличии object validation
- `val/object_precision` — только при наличии object validation
- `val/object_recall` — только при наличии object validation
- `train/epoch_time_sec`
- `train/learning_rate` — во всех вариантах обучения, если значение задано; скорость обучения текущей эпохи до шага scheduler
- `train/region_loss`, `train/boundary_loss`, `val/region_loss`, `val/boundary_loss` — диагностика object f1/v2.
- `val/macro_pixel_f1`, `val/macro_pixel_precision`, `val/macro_pixel_recall`, `val/macro_pixel_iou`, `val/micro_pixel_f1`, `val/micro_pixel_precision`, `val/micro_pixel_recall` — доступные агрегаты сцен или типов.
- `val/foreground_pixel_f1`, `val/foreground_pixel_precision`, `val/foreground_pixel_recall` — объединение типов multiclass.
- `val/class/{slug}/{precision|recall|f1|iou}` — метрики каждого типа multiclass.
- `val/performed` — `0/1` для разреженной next-gen validation
- `val/scene/{scene_id}/{pixel_f1|pixel_precision|pixel_recall|fixed_0_5_*}` — для next-gen
- `val/fixed_0_5_pixel_f1`, `val/fixed_0_5_pixel_precision`, `val/fixed_0_5_pixel_recall` — для next-gen
- Только next-gen3/4: `sampling/draws`, `sampling/foreground_draws`, `sampling/background_draws`, `sampling/background_fraction`, `sampling/unique_foreground`, `sampling/unique_background`, `sampling/positive_repeats_max_observed`, `sampling/hard_background_draws`, `sampling/background_groups`, `sampling/next_background_draws`, `sampling/next_background_fraction`.
- Только next-gen3/4: `val/empty_pixel_fpr`, `val/empty_tile_fpr`, `val/foreground_recall` и их сглаженные варианты с суффиксом `_ema`. Счётчики и причины не превращаются в числовые метрики: полная структура `EpochMetrics.adaptive_sampling` сохраняется в `reports/sampling/epoch_NNNN.json`, итоговая — `reports/adaptive_sampling.json`; test включает отдельную диагностику ошибок фона. Пиковая VRAM использует существующую `train/peak_vram_bytes`.

`log_training_metrics` пишет только итоговые run-level метрики без `step`:

- `train/epochs_total`
- `train/training_time_sec`
- `train/best_quality_f1`
- `train/best_object_f1` — вместо `train/best_quality_f1` при объектном критерии checkpoint.
- `train/best_threshold_pixel_f1`
- `train/best_macro_pixel_f1`, `train/best_confidence_threshold` — если доступны.
- `train/stopped_early` — команда остановки с сохранением; не индикатор срабатывания patience.
- `train/peak_vram_bytes` — для next-gen
- `test/loss`, `test/pixel_f1`, `test/pixel_precision`, `test/pixel_recall`, `test/threshold`, `test/checkpoint_epoch`; для multiclass дополнительно `test/macro_pixel_f1`, `test/micro_pixel_f1`, `test/foreground_pixel_f1`; для object f1/v2 — `test/object_f1`, `test/object_precision`, `test/object_recall`. Вложенные отчёты не становятся числовыми метриками.

## Алгоритм работы и его особенности

Открывает запуск с готовыми dataset/tags; без имени использует класс, дату и следующий номер. Конфиги, разметку, отчёты и checkpoint сохраняет как артефакты; scalar params нормализует. Эпохи без validation не получают val-полей, кроме performed. Best читается по метаданным критерия: next-gen2/3/4 — минимум val/loss, object — object F1; историческая quality_f1 сохраняет приоритет. При равенстве берётся ранняя эпоха; usable требует FINISHED, threshold и best.pt.

Справочник каждой записанной метрики — reports/metric_descriptions.html и .json; контекст берётся из безопасных тегов. HTML автономный, названия имеют подсказки. Обновляется при изменении ключей/контекста, но не чисел. В Description заменяется только помеченная ссылка; пользовательский текст не обрезается при лимите 8000 символов. Сбой справки не прерывает обучение. Явный API дополняет и завершённые запуски без открытия/смены статуса. Новая метрика требует описания и теста покрытия. Подробности — в docs/architecture.md.
