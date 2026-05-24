# Модуль train

## Назначение

`train` выполняет реальный PyTorch training loop на готовых train/val DataLoader, отправляет progress events через sink и возвращает `TrainResult`. Модуль не решает, что именно писать в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` - обучает переданную модель и возвращает результат обучения.

## Публичные контракты

- `TrainError` - ошибка обучения.
- `TrainConfig` - поля `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`.
- `EpochMetrics` - поля `epoch`, `train_loss`, `train_optimizer_steps`, `train_skipped_optimizer_steps`, `val_loss`, `val_pixel_precision`, `val_pixel_recall`, `val_pixel_f1`, `val_positive_pixels`, `val_pred_positive_pixels`, `val_true_positive`, `val_false_positive`, `val_false_negative`, `epoch_time_sec`.
- `CheckpointArtifact` - поля `uri`, `label`.
- `TrainProgressEvent` - поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` - протокол приема событий прогресса.
- `TrainRequest` - поля `model`, `train_loader`, `val_loader`, `config`, `checkpoint_dir`.
- `TrainResult` - поля `history`, `epochs_total`, `training_time_sec`, `best_checkpoint_path`, `final_checkpoint_path`, `artifacts`.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` - публичный контракт модели, которую нужно обучить.
- `models.api` - сохранить best/final checkpoint через публичный API.
- `torch` - выполнить обучение, optimizer, scheduler, losses и tensor operations; импортируется лениво.

## Алгоритм работы и его особенности

`train_model` переносит модель на `config.device`, создает AdamW и cosine scheduler. На каждой эпохе выполняются train loop, validation loop, scheduler step и формируется `EpochMetrics`. В начале эпохи отправляется `TrainProgressEvent(message="epoch_started")`, после validation и сохранения best checkpoint отправляется `TrainProgressEvent(message="epoch_finished", metrics=metrics)`.

Input batch от `tile_preparation`: `(images, masks)` или `(images, masks, batch_meta)`. `images: torch.float32 [B,C,H,W]` с raw values без нормализации, `masks: torch.float32 [B,1,H,W]` binary `0/1`. `batch_meta` не участвует в loss. Forward берет `.logits`, если поле есть, и resize logits до mask size при необходимости.

Loss поддерживает `bce_dice`, `focal_dice`, `focal_tversky`. Validation считает `val_loss`, pixel precision, recall, f1 и диагностические счетчики TP/FP/FN, GT-positive pixels и predicted-positive pixels по `threshold` без threshold sweep. Early stopping идет по `val_pixel_f1`. `max_train_batches_per_epoch` и `max_val_batches_per_epoch` ограничивают число batch в эпохе только для smoke/debug запусков.

Train loop проверяет `images`, `masks`, `logits`, `loss` и итоговые loss-метрики на finite values, чтобы ошибка обучения была диагностируемой до создания `EpochMetrics`. После backward применяется фиксированный gradient clipping `max_norm=1.0`. Non-finite gradient skip - аварийная защита, а не нормальный путь обучения: один batch может быть пропущен и попадет в `train_skipped_optimizer_steps`, но если пропусков больше внутреннего аварийного лимита, обучение завершается `TrainError`. Если за эпоху не выполнен ни один optimizer step, обучение также завершается `TrainError`.
