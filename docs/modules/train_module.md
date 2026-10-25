# Модуль train

## Назначение

`train` выполняет реальный PyTorch training loop на готовых train/val DataLoader, отправляет progress events через sink и возвращает `TrainResult`. Модуль не решает, что именно писать в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` - обучает переданную модель и возвращает результат обучения.

## Публичные контракты

- `TrainError` - ошибка обучения.
- `TrainClassDefinition` - `id`, `slug`, `name`, `color`, `priority`; полный класс checkpoint и MLflow.
- `TrainConfig` - поля task/metric, `pipeline_variant`, `class_weights` (три веса для object_f1_v2, два для binary next-gen2/object_f1, N+1 для multiclass next-gen2, иначе пустой список), optional `object_separation` и `calibrate_object_threshold` для v2, validation interval, threshold mode, optional Gaussian A/B, optimizer/loss, threshold, patience, batch/time limits и class schema. Профиль разделения должен совпадать с вариантом; исторический object_f1 не допускает его переопределения.
- `EpochMetrics` - поля эпохи, `validation_performed`, optional val loss/метрики, train_region_loss/train_boundary_loss/val_region_loss/val_boundary_loss, `val_object_separation`, `val_object_threshold_sweep`, learning rate, binary per-scene/pixel/object либо multiclass per-class, macro, micro и foreground метрики. `learning_rate` содержит скорость обучения текущей эпохи до шага scheduler, в том числе на эпохах без validation.
- `CheckpointArtifact` - поля `uri`, `label`.
- `TrainProgressEvent` - поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` - протокол приема событий прогресса.
- `TrainRequest` - поля `model`, `train_loader`, `val_loader`, optional `test_loader`, optional `dense_train_loader` (только v4: исходные плотные train-окна без аугментаций для eval лучших весов), `config`, `checkpoint_dir`, `sample_size`, `run_metadata`.
- `TrainResult` - история/checkpoint/порог/остановка и `diagnostics`.

`TrainConfig.pipeline_variant=next_gen3|next_gen4|next_gen4_global_norm|next_gen4_augmentation_v3`: прежний loss и критерий next-gen2, обязательная valid-mask и адаптивный train-сэмплер. `EpochMetrics.adaptive_sampling` по умолчанию пуст: профили сохраняют квоты, нормированные ошибки полной validation, сглаживание и причину следующего решения. После test контроллер не вызывается. Checkpoint и итоговая диагностика сохраняют полное состояние и историю; продолжение от весов начинает новый контроллер. Различие сетки и защита аугментаций принадлежат tile_preparation, глобальная нормализация встроена в переданную модель, не реализуется в train. Правила — в `docs/architecture.md`.

`TrainConfig.pipeline_variant` также принимает `next_gen4_augmentation_v4` с тем же loss/выбором. `EpochMetrics.val_dense_window_metrics` (по умолчанию `{}`) содержит только для v4 группы плотности 0,9/0,99 при допустимой площади ≥0,99: число окон и TP/FP/FN, precision/recall/F1 при 0,5/0,9. Multiclass оценивается как объединение объектов с суммой их вероятностей; неоднозначные отрицательные области исключены. Пустые группы имеют `null` метрик. После выбора best отдельные eval-аудиты train и test сохраняются в `TrainResult.diagnostics.dense_train_windows/test_dense_windows`, без влияния на веса, scheduler, early stopping и пороги.

`TrainConfig` также принимает `pipeline_variant=next_gen4_consistency_v1` и optional
`diagnostic_checkpoint_interval_sec` (положительные секунды либо `None`). Новый профиль
наследует v4; дополнительный KL с весом 0,25 сравнивает eval-прогноз полного train-окна
без градиента и немасштабированной вырезки только на правильных уверенных (>0,9)
валидных пикселях учителя, с равным весом классов. Дополнительный backward идёт после
основного, до единственного optimizer step; статистики нормализации не меняются.
`EpochMetrics.train_context_consistency` содержит счётчики и средний KL полной истории.
Периодические текущие срезы и атомарный индекс фиксируют фактическое время и эпоху;
они не меняют best, test или состояние непрерывного обучения. Подробности в архитектуре.

`next_gen4_consistency_v2` полностью наследует этот цикл, диагностику и тот же KL.
Замена нормализации декодера принадлежит models: train не преобразует слои или веса.

`next_gen4_consistency_v3` сохраняет цикл v2, но разрешает уверенный правильный учитель
в любом из двух направлений. KL каждого направления балансируется по классам; активные
направления усредняются с общим весом 0,25. Учитель отсоединён от градиента, ученик получает
его и для полного окна. История дополнительно хранит `forward_selected_batches`,
`reverse_selected_batches`, `forward_supervised_pixels`, `reverse_supervised_pixels`,
`forward_kl_sum`, `reverse_kl_sum`, `mean_forward_kl`, `mean_reverse_kl`.
Общий счётчик пикселей суммирует направления, не уникальные координаты.

`next_gen4_consistency_v4` наследует v3. Дополнение объединённых масок правильного
уверенного учителя внутри допустимой train-разметки получает CE обоих контекстов.
KL и CE усредняются по присутствующим классам и активным направлениям с прежним
общим весом 0,25; старые ветки неизменны. Невалидные и неоднозначные пиксели исключены.
История дополнительно хранит `fallback_selected_batches`, `fallback_supervised_pixels`
(посещения двух направлений), `context_loss_sum`, `teacher_contribution_sum`,
`fallback_ce_contribution_sum` и соответствующие `mean_context_loss`,
`mean_teacher_contribution`, `mean_fallback_ce_contribution`. Прежние направленные
KL-счётчики относятся только к уверенным учителям; общий `selected_batches` в v4
включает и пакеты только с прямым надзором. Validation/test не получают новый loss.

`next_gen4_consistency_v5` сохраняет цикл, но дополнительный CE обучает все допустимые
пиксели полного query-входа при отсоединённых K/V случайной внутренней вырезки.
Train выбирает вырезку и full valid-mask, исключает class-negative, усредняет CE
равными средними присутствующих классов и применяет вес0,25 один раз. Два прохода
eval и один optimizer step, режимы восстанавливаются при ошибке. Attention выполняет
models через аргумент `context_reference`; private-код models не импортируется.
История содержит attempted/selected batches, supervised_pixels, reference_valid_pixels,
half_side_selected_batches, three_quarter_side_selected_batches, context_loss_sum,
mean_context_loss; направленных KL/teacher/fallback-счётчиков у v5 нет.

`next_gen4_consistency_v6` сохраняет adversarial query v5 и добавляет CE обычного
прогноза источника по его собственным обрезанным target/valid-mask. Обе цели
исключают invalid/nodata/class-negative, балансируются по присутствующим классам;
среднее двух CE умножается на0,25 один раз. Оба прохода eval, режимы восстанавливаются,
основной CE/Tversky и один optimizer step прежние. Train получает пару logits через
тот же публичный forward модели, не импортирует private models.
История хранит `query_supervised_pixels`, `reference_supervised_pixels`, их сумму
`supervised_pixels`, `query_contribution_sum`, `reference_contribution_sum` и
соответствующие mean; contribution — половина CE, сумма даёт `context_loss_sum`
до общего веса. Остальные счётчики отбора v5 сохраняются; KL/teacher/fallback нет.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` - публичный контракт модели, которую нужно обучить.
- `tile_preparation.contracts` — метка отрицательных примеров, `SamplingFeedback` и протокол `AdaptiveSamplingController`; частные реализации загрузчиков не импортируются.
- `models.api` - сохранить best/final checkpoint через публичный API.
- `metrics.api` - сопоставить объекты один к одному и рассчитать объектовую F1.
- `inference.api/contracts` — общая сборка вероятностей, выделение ID и сохранённые параметры разделения для object_f1 и object_f1_v2.
- `torch` - выполнить обучение, optimizer, scheduler, losses и tensor operations; импортируется лениво.

## Алгоритм работы и его особенности

`object_f1_v2` обучает три softmax-класса weighted CE/Tversky; region/boundary loss — диагностические Dice полной площади и границы. Полная validation калибрует boundary threshold 0.35/0.5/0.65 по object F1, precision и близости к 0.5. Готовое разделение на исходном пороге оценивается первым; остальные пороги переиспользуют дисковые карты. Независимые сцены проверяются CPU-пулом до четырёх потоков, не больше loader.num_workers, с консервативным ограничением по доступной RAM и размеру сцены; без сведений о RAM — один поток. Пороги одной сцены последовательны, порядок отчёта постоянен. Управление CUDA/паузой остаётся в главном потоке; рабочие потоки ждут между порогами, закрытие карт — только после их завершения. Упаковка пар ID ускоряет точный подсчёт пересечений, формулы не меняются. Журнал разделяет обучение, инференс проверки и CPU-метрики. Настройки checkpoint входят в ModelSpec/metadata; новые checkpoint фиксируют `object_separation_algorithm=interior_geodesic_v1`, старые метрики не переписываются. Test использует best без калибровки. Поснимочно сохраняются полнота границ, склейки/дробления (перекрытие≥10%) и числа объектов.

Все профили принимают optional `batch_meta.valid_pixels` для размеченных зон. Маска исключает пиксели из каждого компонента loss и метрик, включая multiclass и class-specific hard negative; нормализация loss использует допустимые пиксели. Полностью пустая маска даёт конечный нулевой loss и нулевые градиенты. При отсутствии зон сохраняются прежние правила профиля. Object f1 собирает виртуальные сцены и игнорирует искусственную границу зоны.

`object_f1` получает три logits, сохраняет loss области next-gen2 и добавляет половину BCE/Dice границ. `EpochMetrics` дополняется train_region_loss/train_boundary_loss/val_region_loss/val_boundary_loss. Validation/test использует общий `inference.api`: Gaussian сборку полных сцен и watershed. Best/early stopping — максимум object F1, scheduler — минимум суммарного val_loss. Пороги 0.5; компоненты loss и объектовая test-метрика сохраняются.

`next_gen2` получает два binary logits либо N+1 multiclass logits, использует 25% weighted CrossEntropy + 75% SMP multiclass
Tversky (alpha/beta 0.75/0.25), AdamW без clipping и ReduceLROnPlateau(min, patience=3, factor=0.5).
Loss усредняется по числу примеров; validation полная каждую эпоху. Best и early stopping используют
минимум validation loss, при равенстве сохраняется ранняя эпоха. После штатного завершения best
загружается локально и один раз оценивается через optional test_loader; test не влияет на обучение.
Multiclass validation/test используют фиксированный порог 0.5 и per-class/macro/micro/foreground оценки.
Частичная отрицательная разметка исключена из CE/Tversky и добавляет штраф только назначенному классу;
полностью невалидный batch даёт конечный нулевой loss. Общая геометрия масок сохраняется в loader.
Диагностика содержит checkpoint_selection и test_metrics (loss, pixel precision/recall/F1, порог и эпоху).
Пауза и остановка с сохранением остаются на границе пакетов; при ручной остановке test не запускается.

`train_model` переносит модель на `config.device` и создаёт AdamW. `legacy` побитово сохраняет cosine scheduler и validation каждой эпохи. `next_gen` использует `ReduceLROnPlateau(mode=max,factor=0.5,patience=3,min_lr=1e-7)` по полной scene-macro val F1; validation выполняется на эпохе 1, по интервалу и перед штатным завершением, а early stopping считает только validation-события. Между ними `EpochMetrics` содержит train loss и `validation_performed=false` без выдуманных val-значений.

Input batch от `tile_preparation`: `(images, masks)` или `(images, masks, batch_meta)`. В `legacy` nodata, raster mask и padding приходят как background `0`. `next_gen` требует `batch_meta.valid_pixels`; эта маска синхронно обрезается с target и полностью исключает невалидные пиксели из loss и метрик. Forward, resize logits и core-crop сохраняют прежний контракт.

Binary loss поддерживает `bce_dice`, `focal_dice`, `focal_tversky`. Формулы и прежняя обработка background/nodata в `legacy` не изменены; в `next_gen` те же веса применяются только к valid pixels и нормируются по их весу. Legacy threshold grid не меняется. Next-gen `fixed` применяет ровно настроенный threshold; `optimize` использует потоковую 4096-bin histogram и tie-break F1 → precision → больший threshold. Per-scene метрики считаются на едином глобальном пороге, primary pixel score — их невзвешенное среднее, а агрегированная pixel-метрика сохраняется как micro. Для object-primary используется заданная расширенная сетка порогов.

`background_weight` задаёт базовый вес фона, `pos_weight` усиливает positive через BCE/focal, а
`hard_negative_weight` дополнительно умножает штраф supervision `-1`. BCE/focal используют per-pixel weight,
Dice масштабирует foreground probability на фоне, Tversky — false-positive term; значения `1` сохраняют
исходную формулу. `focal_tversky` остаётся суммой focal и Tversky. Предсказанные connected components
сопоставляются с instance masks один к одному при `IoU ≥ 0,5`; расчёты «порог × тайл» выполняются
детерминированно ограниченным CPU-пулом. Для pixel-primary object F1 считается точным вторым проходом на
выбранном pixel threshold; приближение ближайшим порогом не используется.

В legacy мультиклассовый режим требует `task=multiclass` и `loss=cross_entropy` или `loss=cross_entropy_dice`; next-gen2 использует `cross_entropy_tversky`. Logits имеют форму `[B,num_classes,H,W]`, mask — `[B,H,W] long`. Перед loss общий hard-negative `-1` заменяется на background class `0`; `cross_entropy` считается через `torch.nn.functional.cross_entropy`; `cross_entropy_dice` добавляет Dice loss по softmax probabilities для foreground классов `1..N`, исключая background `0`. Nodata и padding уже представлены background class `0` и участвуют в loss и validation. `background_weight` задаёт вес класса `0`, а `hard_negative_weight` дополнительно усиливает общий hard negative для всех foreground-каналов. Class-specific hard negative исключается из базового multiclass loss и добавляет штраф `-log(1-p_class) × background_weight × hard_negative_weight` только выбранному foreground-каналу; другие типы в этой области не считаются ошибкой. В validation такой пиксель участвует как отрицательный только для назначенного класса, исключается из остальных поклассовых и агрегированной foreground-метрики. Validation применяет `softmax → argmax → confidence threshold`, считает precision/recall/F1/IoU отдельно по типам, macro pixel F1/precision/recall/IoU, micro F1 и foreground-vs-background F1. Один threshold выбирается по максимуму macro pixel F1, затем macro precision, затем по большему threshold.

В остальных вариантах best checkpoint и early stopping используют `val_quality_f1`; для multiclass это macro pixel F1. Для `quality_metric=objects` требуется binary val batch с `object_instances`; multiclass с объектовой метрикой отклоняется валидацией. Metadata checkpoint сохраняет task, полную class schema, выбранный confidence threshold, структурированные validation-метрики, loss, входной `sample_size`, `inference_context`, `inference_core_size`, `seed` и `train_config`. Старые checkpoint без новых полей остаются совместимыми и при экспорте получают `context=0`, если оператор не задал его вручную.

`max_train_batches_per_epoch` ограничивает train для smoke/debug. `legacy` сохраняет optional val-limit; в `next_gen` он запрещён. `max_training_time_sec` инициирует обязательную validation после завершённой train-эпохи и штатное сохранение final checkpoint.

Опциональная next-gen Gaussian A/B диагностика после обучения загружает `best.pt` локально, объединяет полные
окна 512/stride 256 с `sigma=patch_size/4`, проверяет покрытие и нулевой прогноз на nodata и записывает отдельные
core-crop/Gaussian метрики. Collated valid mask принимается в формах `[B,H,W]` и `[B,1,H,W]`; размерность и
соответствие batch проверяются до накопления сцены. Диагностика не изменяет production metadata, ONNX и
Geoalert core-crop.

Если задан `MLSYSTEM2_TRAINING_CONTROL_DIR`, train loop после каждого train/validation batch проверяет
`pause.request`. При паузе модель и optimizer state переносятся в CPU, CUDA освобождается и атомарно создаётся
маркер `paused` с тем же token. После удаления запроса состояние возвращается на исходное device; процесс,
DataLoader, scheduler, история эпох и MLflow-run не пересоздаются. DataLoader workers кооперативно прекращают
подготовку новых тайлов только после подтверждения `paused` с token текущего запроса: до этого train loop
может ждать batch или сброса итератора. Подтверждённая пауза освобождает GPU и основную CPU/IO-нагрузку.

В том же control-каталоге `stop-and-save-best.request` означает штатно завершить обучение с уже созданным
`best.pt`. Запрос проверяется на границе каждого train/validation batch и во время паузы. Незавершённая эпоха
не добавляется в history; модель из её текущего состояния не сохраняется. Результат получает
`stopped_early=true`, `final_checkpoint_path=None` и единственный checkpoint-артефакт `best`, выбранный по
критерию конвейера среди завершённых эпох. Если до запроса не завершилась ни одна эпоха и `best.pt` отсутствует,
обучение не может быть выдано как успешный результат.

Train loop проверяет `images`, `masks`, `logits`, `loss`, `train_loss` и `val_loss` на finite values, чтобы ошибка обучения была диагностируемой до создания `EpochMetrics`. После backward применяется фиксированный gradient clipping `max_norm=1.0`. Non-finite gradient skip - аварийная защита, а не нормальный путь обучения: один batch может быть пропущен, но этот счетчик не входит в публичные метрики; если пропусков больше внутреннего аварийного лимита, обучение завершается `TrainError`. Если за эпоху не выполнен ни один optimizer step, обучение также завершается `TrainError`.
