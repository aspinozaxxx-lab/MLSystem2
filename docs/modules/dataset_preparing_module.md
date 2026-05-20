# Модуль dataset_preparing

## Назначение

`dataset_preparing` проверяет наличие снимков датасета, читает список сцен и разметку, считает
количество объектов, строит train/val разбиение с балансировкой по числу объектов, готовит границы
сцен и возвращает `DatasetManifest` с подробным отчетом подготовки.

## Публичный интерфейс

`prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult`

Параметры:

- `request.images_uri`: базовый URI снимков.
- `request.scenes_file`: путь или URI файла со списком сцен.
- `request.annotation_file`: путь или URI файла разметки.
- `request.default_class_dir`: путь или URI директории классов.
- `request.val_fraction`: доля валидации.
- `request.split_strategy`: сейчас только `object_count_balanced`.
- `request.output_uri`: необязательное место для артефактов подготовки.

## Контракты

`DatasetPreparationRequest`, `DatasetPreparationResult`, `DatasetManifest`, `DatasetSplit`,
`SceneRecord`, `SceneFootprint`, `ObjectCountByScene`, `DatasetPreparationReport`,
`DatasetPreparationError`.

## Выходные артефакты

`DatasetManifest`, `DatasetPreparationReport`, ссылка на артефакт границ снимков, train-сцены,
val-сцены, количество объектов по сценам, список отсутствующих файлов, предупреждения и ошибки.

## Что модуль НЕ делает

Не нарезает тайлы, не обучает, не пишет в MLflow напрямую, не создает PyTorch dataloader и не
запускает конвейер.

## Разрешенные зависимости

`storage.api`, геометрические библиотеки в приватной реализации footprints и стандартные средства
разбора данных.

## Запрещенные пересечения

Не импортирует обучение, внутренности подготовки тайлов, внутренности конвейера обучения и внутренности
адаптера MLflow.

## MLflow

Напрямую в MLflow не пишет. Конвейер логирует отчет через `mlflow_adapter`.

## Временное профилирование

Не владеет замерами времени конвейера. Может включать счетчики подготовки в отчет.
