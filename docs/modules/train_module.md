# Модуль train

## Назначение

`train` обучает модель на готовых train/val DataLoader, отправляет события прогресса через sink и
возвращает `TrainResult`. Модуль не решает, что именно попадет в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` — обучает переданную модель на готовых DataLoader и возвращает результат обучения.

## Публичные контракты

- `TrainError` — ошибка обучения.
- `TrainConfig` — поля `epochs`, `batch_size`, `device`, `learning_rate`, `weight_decay`, `loss`, `focal_alpha`, `pos_weight`, `tversky_alpha`, `tversky_beta`, `threshold`, `early_stopping_patience`; не управляет параметрами построения DataLoader.
- `EpochMetrics` — поля `epoch`, `train_loss`, `val_loss`, `val_pixel_precision`, `val_pixel_recall`, `val_pixel_f1`, `epoch_time_sec`.
- `CheckpointArtifact` — поля `uri`, `label`.
- `TrainProgressEvent` — поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` — протокол приема событий прогресса.
- `TrainRequest` — поля `model`, `train_loader`, `val_loader`, `config`, `checkpoint_dir`.
- `TrainResult` — поля `history`, `epochs_total`, `training_time_sec`, `best_checkpoint_path`, `final_checkpoint_path`, `artifacts`.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` — публичный контракт модели, которую нужно обучить.
- `models.api` — сохранить best/final checkpoint через публичный API.

## Алгоритм работы и его особенности

Получает модель, готовые `train_loader` и `val_loader`, конфиг и директорию чекпойнтов. Выполняет `model.to(device)`, AdamW, cosine scheduler, BCE/Dice-family loss, forward по `images [B,C,H,W]`, берет `.logits` и resize до mask, валидирует каждую эпоху по threshold без sweep, считает train/val loss и pixel precision/recall/f1, сохраняет best/final checkpoints, останавливается early stopping по `val_pixel_f1`, отправляет progress events и возвращает `TrainResult`. Модуль не строит DataLoader.
