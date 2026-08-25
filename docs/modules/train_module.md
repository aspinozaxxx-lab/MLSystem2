# Модуль train

## Назначение

`train` выполняет реальный PyTorch training loop на готовых train/val DataLoader, отправляет progress events через sink и возвращает `TrainResult`. Модуль не решает, что именно писать в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` - обучает переданную модель и возвращает результат обучения.

## Публичные контракты

- `TrainError` - ошибка обучения.
- `TrainClassDefinition` - `id`, `slug`, `name`, `color`, `priority`; полный класс checkpoint и MLflow.
- `TrainConfig` - поля `task`, `quality_metric`, `epochs`, `batch_size`, `seed`, `inference_context`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `background_weight`, `hard_negative_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, optional batch/time limits и `class_schema` (`class_slugs` принимается для legacy-совместимости).
- `EpochMetrics` - поля эпохи, loss, выбранной `quality_metric`, лучшего порога, binary pixel/object метрик либо multiclass per-class, macro, micro и foreground pixel-метрик.
- `CheckpointArtifact` - поля `uri`, `label`.
- `TrainProgressEvent` - поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` - протокол приема событий прогресса.
- `TrainRequest` - поля `model`, `train_loader`, `val_loader`, `config`, `checkpoint_dir`, `sample_size`.
- `TrainResult` - поля `history`, `epochs_total`, `training_time_sec`, `best_checkpoint_path`, `final_checkpoint_path`, `artifacts`, `task`, `class_schema`, `best_threshold`, `stopped_early`.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` - публичный контракт модели, которую нужно обучить.
- `models.api` - сохранить best/final checkpoint через публичный API.
- `metrics.api` - сопоставить объекты один к одному и рассчитать объектовую F1.
- `torch` - выполнить обучение, optimizer, scheduler, losses и tensor operations; импортируется лениво.

## Алгоритм работы и его особенности

`train_model` переносит модель на `config.device`, создает AdamW и cosine scheduler. На каждой эпохе выполняются train loop, validation loop, scheduler step и формируется `EpochMetrics`. В начале эпохи отправляется `TrainProgressEvent(message="epoch_started")`, после validation и сохранения best checkpoint отправляется `TrainProgressEvent(message="epoch_finished", metrics=metrics)`.

Input batch от `tile_preparation`: `(images, masks)` или `(images, masks, batch_meta)`. `images: torch.float32 [B,C,H,W]` с raw values без нормализации. В binary режиме `masks: torch.float32 [B,1,H,W]`, в multiclass режиме `masks: torch.long [B,H,W]`; это единая supervision mask со значениями `-1=hard negative`, `0=background`, `1` или `1..N=positive`. Для управляемого multiclass-датасета `batch_meta.class_hard_negative_masks` может дополнительно содержать `bool[B,N,H,W]` с отрицательной разметкой отдельно для каждого semantic-типа. Train loop декодирует общий `-1` в target background `0`; nodata, невалидная raster mask и padding уже приходят как обычный background `0`. Forward берет `.logits`, если поле есть, и resize logits до полного mask size при необходимости. Затем logits, target, обе hard-negative mask и binary instance masks синхронно обрезаются на `inference_context` с четырёх сторон. Loss, validation F1, подбор threshold, best checkpoint и early stopping используют только этот полезный центр; контекстная рамка даёт признаки, но не градиент и не метрики. При `inference_context=0` поведение прежнее.

Binary loss поддерживает `bce_dice`, `focal_dice`, `focal_tversky`. Все background pixels, включая nodata и padding, входят во все части loss, а ложный foreground на них учитывается как false positive в пиксельных и объектовых метриках. `background_weight` задаёт их базовый вес, `pos_weight` усиливает positive pixels через BCE/focal, а `hard_negative_weight` дополнительно умножает штраф только на pixels, которые в supervision mask были `-1`. BCE/focal применяют weight map к per-pixel loss; Dice масштабирует вклад foreground probability на background pixels; Tversky масштабирует false-positive term. Значения весов `1` сохраняют прежнюю формулу. `focal_tversky` соответствует старому MLSystem: это сумма focal loss и tversky loss, а не квадрат tversky loss. Train loop возвращает только общий `train_loss`. Binary validation проверяет threshold candidates `[0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]` и считает на каждом пороге пиксельные и объектовые TP/FP/FN. Предсказанные connected components сопоставляются с instance masks один к одному при `IoU ≥ 0,5`; каждый тайл оценивается отдельно. Независимые расчёты объектовой метрики для пар «порог × тайл» выполняются ограниченным пулом не более чем из 8 CPU-потоков, а итоговые счётчики агрегируются последовательно и детерминированно. `quality_metric` выбирает лучший threshold и основную F1, при этом обе группы метрик остаются в `EpochMetrics`.

Multiclass режим требует `task=multiclass` и `loss=cross_entropy` или `loss=cross_entropy_dice`. Logits имеют форму `[B,num_classes,H,W]`, mask — `[B,H,W] long`. Перед loss общий hard-negative `-1` заменяется на background class `0`; `cross_entropy` считается через `torch.nn.functional.cross_entropy`; `cross_entropy_dice` добавляет Dice loss по softmax probabilities для foreground классов `1..N`, исключая background `0`. Nodata и padding уже представлены background class `0` и участвуют в loss и validation. `background_weight` задаёт вес класса `0`, а `hard_negative_weight` дополнительно усиливает общий hard negative для всех foreground-каналов. Class-specific hard negative исключается из базового multiclass loss и добавляет штраф `-log(1-p_class) × background_weight × hard_negative_weight` только выбранному foreground-каналу; другие типы в этой области не считаются ошибкой. В validation такой пиксель участвует как отрицательный только для назначенного класса, исключается из остальных поклассовых и агрегированной foreground-метрики. Validation применяет `softmax → argmax → confidence threshold`, считает precision/recall/F1/IoU отдельно по типам, macro pixel F1/precision/recall/IoU, micro F1 и foreground-vs-background F1. Один threshold выбирается по максимуму macro pixel F1, затем macro precision, затем по большему threshold.

Best checkpoint и early stopping используют `val_quality_f1`; для multiclass это macro pixel F1. Для `quality_metric=objects` требуется binary val batch с `object_instances`; multiclass с объектовой метрикой отклоняется валидацией. Metadata checkpoint сохраняет task, полную class schema, выбранный confidence threshold, структурированные validation-метрики, loss, входной `sample_size`, `inference_context`, `inference_core_size`, `seed` и `train_config`. Старые checkpoint без новых полей остаются совместимыми и при экспорте получают `context=0`, если оператор не задал его вручную.

`max_train_batches_per_epoch` и `max_val_batches_per_epoch` ограничивают число batch в эпохе только для smoke/debug запусков. `max_training_time_sec` проверяется после каждой эпохи и завершает обучение штатно, чтобы сохранить final checkpoint.

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
