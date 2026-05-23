# Модуль train

## Назначение

`train` выполняет реальный PyTorch training loop на готовых train/val DataLoader, отправляет progress events через sink и возвращает `TrainResult`. Модуль не решает, что именно писать в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` - обучает переданную модель и возвращает результат обучения.

## Публичные контракты

- `TrainError` - ошибка обучения.
- `TrainConfig` - поля `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`, `max_train_batches_per_epoch`, `max_val_batches_per_epoch`.
- `EpochMetrics` - поля `epoch`, `train_loss`, `val_loss`, `val_pixel_precision`, `val_pixel_recall`, `val_pixel_f1`, `epoch_time_sec`.
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

Input batch от `tile_preparation`: `images: torch.float32 [B,C,H,W]` с raw values без нормализации и `masks: torch.float32 [B,1,H,W]` binary `0/1`. Forward берет `.logits`, если поле есть, и resize logits до mask size при необходимости.

Loss поддерживает `bce_dice`, `focal_dice`, `focal_tversky`. Validation считает `val_loss`, pixel precision, recall и f1 по `threshold` без threshold sweep. Early stopping идет по `val_pixel_f1`. `max_train_batches_per_epoch` и `max_val_batches_per_epoch` ограничивают число batch в эпохе только для smoke/debug запусков.
