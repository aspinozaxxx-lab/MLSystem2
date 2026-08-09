# Модуль inference_pipeline

## Назначение

`inference_pipeline` оркестрирует приложение инференса из командной строки: получение текущих настроек,
вызов инференса, логирование в MLflow через адаптер, замеры времени и сборку результата.

## Публичный интерфейс

- `run_inference_pipeline(request: InferencePipelineRequest) -> InferencePipelineResult` — запускает полный конвейер инференса на текущих настройках процесса.

## Публичные контракты

- `InferencePipelineError` — невосстановимая ошибка конвейера инференса.
- `InferencePipelineRequest` — поле `run_name`.
- `InferencePipelineResult` — поля `status`, `mlflow_run`, `timings`, `report`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — получить текущие настройки через `get_settings`.
- `mlflow_adapter.api` — управлять запуском MLflow и записывать отчеты.
- `inference.api` — выполнить инференс.

## Алгоритм работы и его особенности

Получает settings через `get_settings`; YAML загружает CLI до запуска pipeline. Открывает запуск MLflow и передаёт имя датасета: stem `annotation_file` для legacy binary, имя `annotations_dir` для per-image binary или соединённые через `+` stem файлов классов для legacy multiclass. Затем формирует `InferenceRequest` по `settings.inference`, вызывает `inference`, пишет timing и итоговый отчёт через `mlflow_adapter`, завершает MLflow run.
