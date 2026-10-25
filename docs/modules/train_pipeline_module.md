# Модуль train_pipeline

## Назначение

`train_pipeline` оркестрирует обучение: управляет MLflow run, готовит датасет, создаёт train/val DataLoader и модель, запускает обучение и сохраняет отчёты.

## Публичный интерфейс

- `run_train_pipeline(request: TrainPipelineRequest) -> TrainPipelineResult` — выполнить полный конвейер обучения.

## Публичные контракты

- `TrainPipelineError` — невосстановимая ошибка конвейера.
- `PipelineStatus` — `succeeded|failed`.
- `ModuleTiming` — `module`, `elapsed_sec`, `details`.
- `TimingReport` — `total_pipeline_time_sec`, `modules`.
- `PipelineReport` — `status`, `message`, `dataset_status`, `errors`, `warnings`, `artifacts`.
- `TrainPipelineRequest` — optional `run_name`.
- `TrainPipelineResult` — `status`, `mlflow_run`, `timings`, `report`.

Next-gen3 использует те же публичные запросы: три полных части оконного разбиения next-gen2, но ограниченный адаптивный train. ModelSpec фиксирует реальную политику сэмплера, а не веса 7/8/1, и `valid_pixel_policy`; отчёты содержат квоты и неизменный бюджет шагов. Validation/test полные; их индексы не передаются train-контроллеру. Сохранена оговорка о географических пересечениях окон.

Next-gen4 наследует эту оркестрацию, передавая stride=tile_size и точное имя варианта всем loaders, модели, train и MLflow. ModelSpec дополнительно фиксирует `mask_output_nodata=true`; split manifest содержит общую неперекрывающуюся сетку зон исходного TIFF. Размер тайла выбирается в конфигурации, а не закрепляется оркестратором.

`next_gen4_global_norm` наследует next-gen4, но запрашивает histogram только train-окон и формирует поканальные `low/high` в `ModelSpec.parameters.preprocessing` до создания модели. Validation/test статистику не собирают. Происхождение и объём входят в диагностику MLflow; resume требует совпадения варианта и preprocessing checkpoint. Нормализация выполняется моделью, не копируется в loaders или отдельный predictor.

`next_gen4_augmentation_v3` наследует эту оркестрацию и добавляет `ModelSpec.parameters.augmentation_policy=validity_guard_v3`. Защиту выполняет tile_preparation, статистика по-прежнему собирается до аугментаций только по train. Точное имя профиля передаётся всем loaders, train и MLflow; смешивание со старыми вариантами при продолжении не допускается.

`next_gen4_augmentation_v4` дополнительно сохраняет `augmentation_policy=validity_guard_v4_context_crop_v1` и точный `context_crop` из архитектуры. Существующий train-loader предоставляет отдельный `dense_window_loader`, который передаётся в `TrainRequest.dense_train_loader` сквозь счётчик без учёта аудита в обучающих показах. Исходная статистика и индексы train сохраняются; train/test-аудит лучших весов не управляет обучением.

`next_gen4_consistency_v1` передаётся всем прежним фасадам как отдельный профиль v4.
Оркестратор фиксирует `context_consistency` в ModelSpec и сверяет её при продолжении;
`diagnostic_checkpoint_interval_sec` передаётся из settings в TrainConfig без изменения
правил выбора best, нормализации, разбиения и автоматического теста.

`next_gen4_consistency_v2` использует ту же оркестрацию, дополнительно сохраняет точный
`decoder_normalization` из архитектуры и сверяет его при продолжении. Нормализатор создаёт
models; loaders, loss, split, глобальные train-границы и выбор лучших весов не меняются.

`next_gen4_consistency_v3` наследует v2; `_model_spec` сохраняет двунаправленную политику
`bidirectional_confident_teacher_kl_v1` и усреднение активных направлений. Продолжение
строго сверяет эту политику и прежний нормализатор; данные и критерий best не меняются.

`next_gen4_consistency_v4` наследует оркестрацию v3 и сохраняет точную политику
`bidirectional_confident_teacher_kl_with_gt_fallback_v1` с
`uncovered_supervision=ground_truth_cross_entropy_v1`. Продолжение сверяет всю
политику и нормализатор. Источник дополнительного надзора — только target текущего
train batch; данные, split, нормализация, основной loss и выбор best прежние.

`next_gen4_consistency_v5` наследует подготовку и оркестрацию v4, сохраняет точную
политику `full_query_detached_crop_kv_gt_v1` из архитектуры и прежний нормализатор.
Resume строго сверяет их. Train получает тот же batch и target, управляет
дополнительным надзором полного query-входа при изменённом K/V-контексте;
данные, split, исходная статистика и выбор best не меняются.

`next_gen4_consistency_v6` передаётся тем же фасадам, сохраняет точную политику
`full_query_detached_crop_kv_with_reference_gt_v1` и нормализатор v5; resume строго
сверяет их. Надзор query и источника по собственным train-target выполняет train,
работу с K/V и пару logits — models. Данные, loaders, нормализация и best прежние.

`next_gen4_consistency_v7` сохраняет и проверяет точную политику
`full_query_detached_geometry_kv_with_reference_gt_v1` и нормализатор v6.
Данные и выбор best прежние; новую reference-геометрию исполняет train,
пару logits и detached K/V — models через прежние публичные интерфейсы.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — текущие настройки и путь YAML.
- `mlflow_adapter.api` — run, метрики, отчёты и артефакты.
- `dataset_preparing.api` — получить проверенный `PreparedDataset` со сценами.
- `tile_preparation.api` — создать train/val loaders.
- `models.api` — создать модель либо загрузить checkpoint.
- `inference.api/contracts` — единый сохранённый профиль разделения объектов v2.
- `train.api` — выполнить обучение.

## Алгоритм работы и его особенности

`object_f1_v2` передаёт три train-веса, отдельную схему фон/внутренность/граница и общий DTO разделения. ModelSpec фиксирует полосу8px, адаптацию ширины, усиление веса границы3, разрешение наложений и версию экспорта5. История калибровки и выбранные параметры записываются в поснимочный отчёт. Resume запрещает смешивать варианты; test не подбирает порог.

До подготовки датасета CUDA проверяется небольшой операцией на выбранном устройстве с синхронизацией; ошибка сохраняется в отчёте MLflow и завершает run без подготовки тайлов. CPU не требует этой проверки.

Оркестратор передаёт описание виртуальных зон из PreparedScene в TileSceneSource. Родительская сцена, ID, геометрия и пиксельное окно входят в идентичность датасета и артефакты разбиения; исторические незональные ревизии сохраняют прежний расчёт.

Число модельных входных каналов передаётся в подготовку и все loaders независимо от конвейера: ортофото использует RGB, Канопус — RGB+NIR. Для трёхканального входа подготовка разрешает совместный набор RGB и RGB+alpha; alpha используется как valid mask и не подаётся в модель. Четырёхканальный вход с alpha отклоняется, поскольку маска не является NIR.

`object_f1` использует профиль next-gen2, но scene_groups 60/20/20 и instance masks во всех loaders. Подготовка разрешает RGB+alpha, loader получает число модельных каналов. ModelSpec содержит output_layout, object_separation, boundary_width, input_channel_policy и версию экспорта; metric tag — val/object_f1. Split manifest и метрики по полным снимкам сохраняются.

Для `next_gen2` оркестратор создаёт train/val/test loaders: полные окна 512/768/1024/1536 с шагом в половину окна, случайное
разбиение 60/20/20 без spatial purge. TrainConfig получает веса классов только по train-маскам.
Модель — выбранная SegFormer (SMP B0/B1/B2/B3, исторически HF B0): бинарная либо N+1 выходов для multiclass,
RGB или RGB+NIR по типу датасета. TrainConfig получает N+1 train-весов и схему всех типов.
Предобученные веса определяются train.pretrained в обоих конвейерах.
Metadata содержит фактический band contract, размер окна, SHA-256
segFormer_train_hlam_main_v2.ipynb, split, sampler 7/8/1 и scheduler по validation loss.
Tag checkpoint_selection_metric=val_loss задаётся до первой эпохи; выбор весов и ранняя остановка
используют минимум loss. Test проверяется после обучения и сохраняется через mlflow_adapter.

Конвейер передаёт в подготовку ожидаемые каналы и `uint8`, затем преобразует каждый `PreparedScene` в `TileSceneSource`. Оба loader получают один `TileSplitRequest`; split выполняется по окнам независимых TIFF. Legacy binary передаёт общие positive/hard-negative GeoJSON, per-image — локальный GeoJSON каждой сцены, legacy multiclass — `class_annotations`, а manifest-backed per-image multiclass — единую schema `classes`; binary val также получает instance masks. Перед созданием модели task и число выходов строго сверяются с подготовленным датасетом, а Python, NumPy, Torch и CUDA инициализируются единым `tile_preparation.seed`. В MLflow seed сохраняется в тегах и конфигурации, каталог `dataset/` получает TXT/GeoJSON legacy либо все GeoJSON и manifest из `annotations_dir`; class schema и структурированные метрики пишутся отдельно. Счётчики loader фиксируют сцены, разрешение и числа окон по каждому TIFF, valid-footprint, sampling и cache. Затем создаётся/загружается модель, `train_model` получает размер полного входа, context, seed и progress sink, а конвейер пишет epoch metrics, checkpoints, tile/timing/pipeline reports и корректно завершает MLflow run.

Поле `train.pipeline_variant` является единственным переключателем. Для `legacy` запросы, model spec и отчёты
сохраняют прежние значения. Для `next_gen` конвейер требует band contract `RED,GRN,BLU,NIR`, создаёт scene-fold
с spatial purge, передаёт valid-mask и при robust profile собирает histogram до создания модели. `ModelSpec`
получает variant, preprocessing, band contract, split/fold, scheduler, threshold policy и pretrained provenance.
Checkpoint resume сверяет архитектуру, каналы и preprocessing, но HF-модель строится из сохранённой config.

После next-gen run конвейер дополняет `TrainResult.diagnostics`: resolved settings JSON, split manifest,
preprocessing, runtime/commit/CUDA/packages, зафиксированную dataset revision и validation по сценам. Commit
берётся из Git checkout, а в production-копии без `.git` — из проверенного 40-символьного `DEPLOYED_COMMIT`. Эти данные
передаются существующему публичному фасаду `mlflow_adapter`; нового верхнеуровневого модуля нет.

До создания модели конвейер ограничивает внутренние Torch CPU pools значениями
`MLSYSTEM2_TORCH_NUM_THREADS` и `MLSYSTEM2_TORCH_NUM_INTEROP_THREADS`; серверный worker передаёт `4` и `2`.
Это ограничение не меняет число DataLoader workers или объём `prefetch_epochs`.
Для next-gen2 действует отдельный ограниченный prefetch_factor=2. Фактические num_workers,
prefetch_factor, persistent_workers и pin_memory каждого train/val/test-загрузчика сохраняются
в loader_runtime существующего отчёта tile_preparation в MLflow.
