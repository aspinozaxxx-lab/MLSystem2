# Модуль train

## Назначение

`train` выполняет реальный PyTorch training loop на готовых train/val DataLoader, отправляет progress events через sink и возвращает `TrainResult`. Модуль не решает, что именно писать в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` - обучает переданную модель и возвращает результат обучения.

## Публичные контракты

- `TrainError` - ошибка обучения.
- `TrainClassDefinition` - `id`, `slug`, `name`, `color`, `priority`; полный класс checkpoint и MLflow.
- `TrainConfig` - поля `task`, `quality_metric`, `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `hard_negative_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, optional batch/time limits и `class_schema` (`class_slugs` принимается для legacy-совместимости).
- `EpochMetrics` - поля эпохи, loss, выбранной `quality_metric`, лучшего порога, binary pixel/object метрик либо multiclass per-class, macro, micro и foreground pixel-метрик.
- `CheckpointArtifact` - поля `uri`, `label`.
- `TrainProgressEvent` - поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` - протокол приема событий прогресса.
- `TrainRequest` - поля `model`, `train_loader`, `val_loader`, `config`, `checkpoint_dir`, `sample_size`.
- `TrainResult` - поля `history`, `epochs_total`, `training_time_sec`, `best_checkpoint_path`, `final_checkpoint_path`, `artifacts`, `task`, `class_schema`, `best_threshold`.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` - публичный контракт модели, которую нужно обучить.
- `models.api` - сохранить best/final checkpoint через публичный API.
- `metrics.api` - сопоставить объекты один к одному и рассчитать объектовую F1.
- `torch` - выполнить обучение, optimizer, scheduler, losses и tensor operations; импортируется лениво.

## Алгоритм работы и его особенности

`train_model` переносит модель на `config.device`, создает AdamW и cosine scheduler. На каждой эпохе выполняются train loop, validation loop, scheduler step и формируется `EpochMetrics`. В начале эпохи отправляется `TrainProgressEvent(message="epoch_started")`, после validation и сохранения best checkpoint отправляется `TrainProgressEvent(message="epoch_finished", metrics=metrics)`.

Input batch от `tile_preparation`: `(images, masks)` или `(images, masks, batch_meta)`. `images: torch.float32 [B,C,H,W]` с raw values без нормализации. В binary режиме `masks: torch.float32 [B,1,H,W]`, в multiclass режиме `masks: torch.long [B,H,W]`; это единая supervision mask со значениями `-1=hard negative`, `0=background`, `1` или `1..N=positive`. Train loop декодирует `-1` в target background `0`; nodata, невалидная raster mask и padding уже приходят как обычный background `0`. Forward берет `.logits`, если поле есть, и resize logits до mask size при необходимости.

Binary loss поддерживает `bce_dice`, `focal_dice`, `focal_tversky`. Все background pixels, включая nodata и padding, имеют базовый вес `1` и входят во все части loss, а ложный foreground на них учитывается как false positive в пиксельных и объектовых метриках. `pos_weight` усиливает positive pixels через BCE/focal, а `hard_negative_weight` усиливает штраф только на pixels, которые в supervision mask были `-1`. BCE/focal применяют weight map к per-pixel loss; Dice усиливает вклад foreground probability только в hard-negative background pixels; Tversky усиливает только false-positive term. `focal_tversky` соответствует старому MLSystem: это сумма focal loss и tversky loss, а не квадрат tversky loss. Train loop возвращает только общий `train_loss`. Binary validation проверяет threshold candidates `[0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]` и считает на каждом пороге пиксельные и объектовые TP/FP/FN. Предсказанные connected components сопоставляются с instance masks один к одному при `IoU ≥ 0,5`; каждый тайл оценивается отдельно. Независимые расчёты объектовой метрики для пар «порог × тайл» выполняются ограниченным пулом не более чем из 8 CPU-потоков, а итоговые счётчики агрегируются последовательно и детерминированно. `quality_metric` выбирает лучший threshold и основную F1, при этом обе группы метрик остаются в `EpochMetrics`.

Multiclass режим требует `task=multiclass` и `loss=cross_entropy` или `loss=cross_entropy_dice`. Logits имеют форму `[B,num_classes,H,W]`, mask — `[B,H,W] long`. Перед loss hard-negative `-1` заменяется на background class `0`; `cross_entropy` считается через `torch.nn.functional.cross_entropy`; `cross_entropy_dice` добавляет Dice loss по softmax probabilities для foreground классов `1..N`, исключая background `0`. Nodata и padding уже представлены background class `0` и участвуют в loss и validation. `hard_negative_weight` усиливает только hard-negative pixels: для CE это per-pixel weight, для multiclass Dice — foreground probabilities классов `1..N` в этих pixels. Validation применяет `softmax → argmax → confidence threshold`, считает precision/recall/F1/IoU отдельно по типам, macro pixel F1/precision/recall/IoU, micro F1 и foreground-vs-background F1. Один threshold выбирается по максимуму macro pixel F1, затем macro precision, затем по большему threshold.

Best checkpoint и early stopping используют `val_quality_f1`; для multiclass это macro pixel F1. Для `quality_metric=objects` требуется binary val batch с `object_instances`; multiclass с объектовой метрикой отклоняется валидацией. Metadata checkpoint сохраняет task, полную class schema, выбранный confidence threshold, структурированные validation-метрики, loss, `sample_size` и `train_config`. Старые binary checkpoint и настройки без `quality_metric` остаются совместимыми.

`max_train_batches_per_epoch` и `max_val_batches_per_epoch` ограничивают число batch в эпохе только для smoke/debug запусков. `max_training_time_sec` проверяется после каждой эпохи и завершает обучение штатно, чтобы сохранить final checkpoint.

Train loop проверяет `images`, `masks`, `logits`, `loss`, `train_loss` и `val_loss` на finite values, чтобы ошибка обучения была диагностируемой до создания `EpochMetrics`. После backward применяется фиксированный gradient clipping `max_norm=1.0`. Non-finite gradient skip - аварийная защита, а не нормальный путь обучения: один batch может быть пропущен, но этот счетчик не входит в публичные метрики; если пропусков больше внутреннего аварийного лимита, обучение завершается `TrainError`. Если за эпоху не выполнен ни один optimizer step, обучение также завершается `TrainError`.
