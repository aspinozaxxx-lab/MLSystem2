# Модуль train

## Назначение

`train` выполняет реальный PyTorch training loop на готовых train/val DataLoader, отправляет progress events через sink и возвращает `TrainResult`. Модуль не решает, что именно писать в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` - обучает переданную модель и возвращает результат обучения.

## Публичные контракты

- `TrainError` - ошибка обучения.
- `TrainClassDefinition` - `id`, `slug`, `name`, `color`, `priority`; полный класс checkpoint и MLflow.
- `TrainConfig` - поля task/metric, `pipeline_variant`, validation interval, threshold mode, optional Gaussian A/B, optimizer/loss, threshold, patience, batch/time limits и class schema.
- `EpochMetrics` - поля эпохи, `validation_performed`, optional val loss/метрики, learning rate, binary per-scene/pixel/object либо multiclass per-class, macro, micro и foreground метрики.
- `CheckpointArtifact` - поля `uri`, `label`.
- `TrainProgressEvent` - поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` - протокол приема событий прогресса.
- `TrainRequest` - поля `model`, `train_loader`, `val_loader`, `config`, `checkpoint_dir`, `sample_size`, `run_metadata`.
- `TrainResult` - история/checkpoint/порог/остановка и `diagnostics`.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` - публичный контракт модели, которую нужно обучить.
- `models.api` - сохранить best/final checkpoint через публичный API.
- `metrics.api` - сопоставить объекты один к одному и рассчитать объектовую F1.
- `torch` - выполнить обучение, optimizer, scheduler, losses и tensor operations; импортируется лениво.

## Алгоритм работы и его особенности

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

Multiclass режим требует `task=multiclass` и `loss=cross_entropy` или `loss=cross_entropy_dice`. Logits имеют форму `[B,num_classes,H,W]`, mask — `[B,H,W] long`. Перед loss общий hard-negative `-1` заменяется на background class `0`; `cross_entropy` считается через `torch.nn.functional.cross_entropy`; `cross_entropy_dice` добавляет Dice loss по softmax probabilities для foreground классов `1..N`, исключая background `0`. Nodata и padding уже представлены background class `0` и участвуют в loss и validation. `background_weight` задаёт вес класса `0`, а `hard_negative_weight` дополнительно усиливает общий hard negative для всех foreground-каналов. Class-specific hard negative исключается из базового multiclass loss и добавляет штраф `-log(1-p_class) × background_weight × hard_negative_weight` только выбранному foreground-каналу; другие типы в этой области не считаются ошибкой. В validation такой пиксель участвует как отрицательный только для назначенного класса, исключается из остальных поклассовых и агрегированной foreground-метрики. Validation применяет `softmax → argmax → confidence threshold`, считает precision/recall/F1/IoU отдельно по типам, macro pixel F1/precision/recall/IoU, micro F1 и foreground-vs-background F1. Один threshold выбирается по максимуму macro pixel F1, затем macro precision, затем по большему threshold.

Best checkpoint и early stopping используют `val_quality_f1`; для multiclass это macro pixel F1. Для `quality_metric=objects` требуется binary val batch с `object_instances`; multiclass с объектовой метрикой отклоняется валидацией. Metadata checkpoint сохраняет task, полную class schema, выбранный confidence threshold, структурированные validation-метрики, loss, входной `sample_size`, `inference_context`, `inference_core_size`, `seed` и `train_config`. Старые checkpoint без новых полей остаются совместимыми и при экспорте получают `context=0`, если оператор не задал его вручную.

`max_train_batches_per_epoch` ограничивает train для smoke/debug. `legacy` сохраняет optional val-limit; в `next_gen` он запрещён. `max_training_time_sec` инициирует обязательную validation после завершённой train-эпохи и штатное сохранение final checkpoint.

Опциональная next-gen Gaussian A/B диагностика после обучения загружает `best.pt` локально, объединяет полные
окна 512/stride 256 с `sigma=patch_size/4`, проверяет покрытие и нулевой прогноз на nodata и записывает отдельные
core-crop/Gaussian метрики. Она не изменяет production metadata, ONNX и Geoalert core-crop.

Если задан `MLSYSTEM2_TRAINING_CONTROL_DIR`, train loop после каждого train/validation batch проверяет
`pause.request`. При паузе модель и optimizer state переносятся в CPU, CUDA освобождается и атомарно создаётся
маркер `paused` с тем же token. После удаления запроса состояние возвращается на исходное device; процесс,
DataLoader, scheduler, история эпох и MLflow-run не пересоздаются. DataLoader workers кооперативно прекращают
подготовку новых тайлов на время запроса, поэтому пауза освобождает не только GPU, но и основную CPU/IO-нагрузку.

В том же control-каталоге `stop-and-save-best.request` означает штатно завершить обучение с уже созданным
`best.pt`. Запрос проверяется на границе каждого train/validation batch и во время паузы. Незавершённая эпоха
не добавляется в history; модель из её текущего состояния не сохраняется. Результат получает
`stopped_early=true`, `final_checkpoint_path=None` и единственный checkpoint-артефакт `best`, выбранный по
максимальной F1 завершённых эпох. Если до запроса не завершилась ни одна эпоха и `best.pt` отсутствует,
обучение не может быть выдано как успешный результат.

Train loop проверяет `images`, `masks`, `logits`, `loss`, `train_loss` и `val_loss` на finite values, чтобы ошибка обучения была диагностируемой до создания `EpochMetrics`. После backward применяется фиксированный gradient clipping `max_norm=1.0`. Non-finite gradient skip - аварийная защита, а не нормальный путь обучения: один batch может быть пропущен, но этот счетчик не входит в публичные метрики; если пропусков больше внутреннего аварийного лимита, обучение завершается `TrainError`. Если за эпоху не выполнен ни один optimizer step, обучение также завершается `TrainError`.
