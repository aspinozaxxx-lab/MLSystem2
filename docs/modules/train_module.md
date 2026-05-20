# Модуль train

## Назначение

`train` обучает модель на готовых источниках батчей тайлов, отправляет события прогресса через sink и
возвращает `TrainResult`. Модуль не решает, что именно попадет в MLflow.

## Публичный интерфейс

`train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult`

Параметры:

- `request.model`: handle модели.
- `request.train_source`: готовый источник train тайлов.
- `request.val_source`: готовый источник валидационных тайлов.
- `request.config`: конфигурация обучения.
- `request.checkpoint_dir`: директория или URI для чекпойнтов.
- `progress_sink`: необязательный обратный вызов для событий прогресса.

## Контракты

`TrainRequest`, `TrainResult`, `TrainConfig`, `TrainProgressEvent`, `TrainProgressSink`,
`EpochMetrics`, `CheckpointArtifact`, `TrainError`.

## Выходные артефакты

Путь к лучшему чекпойнту, необязательный путь к финальному чекпойнту, pixel F1 по эпохам, время
эпох, число эпох и полное время обучения.

## Что модуль НЕ делает

Не пишет в MLflow, не делит датасет, не нарезает тайлы, не читает YAML напрямую и не знает о
жизненном цикле запуска в конвейере обучения.

## Разрешенные зависимости

Необязательный `torch`, `models.contracts`, `tile_preparation.contracts` и `metrics.api`.

## Запрещенные пересечения

Не импортирует внутренности конвейера обучения, адаптера MLflow, подготовки датасета и подготовки
тайлов.

## MLflow

Напрямую не участвует. Результаты логирует `mlflow_adapter`.

## Временное профилирование

Владеет длительностями эпох и общей длительностью обучения внутри `TrainResult`.
