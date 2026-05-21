# Модуль train

## Назначение

`train` обучает модель на готовых train/val DataLoader, отправляет события прогресса через sink и
возвращает `TrainResult`. Модуль не решает, что именно попадет в MLflow.

## Публичный интерфейс

- `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult` — обучает переданную модель на готовых DataLoader и возвращает результат обучения.

## Публичные контракты

- `TrainError` — ошибка обучения.
- `TrainConfig` — поля `epochs`, `batch_size`, `device`; не управляет параметрами построения DataLoader.
- `EpochMetrics` — поля `epoch`, `f1_pixel`, `epoch_time_sec`.
- `CheckpointArtifact` — поля `uri`, `label`.
- `TrainProgressEvent` — поля `epoch`, `message`, `metrics`.
- `TrainProgressSink` — протокол приема событий прогресса.
- `TrainRequest` — поля `model`, `train_loader`, `val_loader`, `config`, `checkpoint_dir`.
- `TrainResult` — поля `history`, `epochs_total`, `training_time_sec`, `best_checkpoint_path`, `final_checkpoint_path`, `artifacts`.

## Список используемых данным модулем модулей и с какой целью

- `models.contracts` — публичный контракт модели, которую нужно обучить.

## Алгоритм работы и его особенности

Получает модель, готовые `train_loader` и `val_loader`, конфиг обучения и директорию чекпойнтов. Выполняет цикл обучения и валидации, отправляет события прогресса через `TrainProgressSink`, сохраняет чекпойнты в переданную директорию и возвращает `TrainResult`. Модуль не строит DataLoader и не управляет `num_workers` или `prefetch_factor`.
