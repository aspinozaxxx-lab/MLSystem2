# Модуль dataset_preparing

## Назначение

`dataset_preparing` готовит локальный raster-датасет к нарезке тайлов: читает список сцен, находит снимки в `images_dir`, считает объекты по разметке, строит train/val split и возвращает VRT XML для train и val без записи VRT на диск.

## Публичный интерфейс

- `prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult` — выполняет подготовку датасета по локальным путям `images_dir`, `scenes_file`, `annotation_file` и доле `val_fraction`.

## Публичные контракты

- `DatasetPreparationError` — исключение невосстановимой ошибки.
- `DatasetPreparationRequest` — поля `images_dir`, `scenes_file`, `annotation_file`, `val_fraction`.
- `PreparedDataset` — поля `train_vrt_xml`, `val_vrt_xml`, `annotation_file`.
- `DatasetSceneReport` — поля `scene_id`, `image_path`, `object_count`, `split`.
- `DatasetPreparationReport` — поля `status`, `scenes_total`, `scenes_found`, `objects_total`, `train_scenes_count`, `train_objects_count`, `val_scenes_count`, `val_objects_count`, `scenes`, `missing_files`, `errors`.
- `DatasetPreparationResult` — поля `dataset`, `report`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. Входы текущей реализации принимаются как локальные пути и читаются через `Path`.

## Алгоритм работы и его особенности

Модуль читает `scenes_file`, индексирует `.tif/.tiff` в `images_dir`, сопоставляет сцены по имени, stem, casefold и нормализованному ключу. При отсутствующем или неоднозначном снимке возвращает `status="error"`. Затем считает объекты по `annotation_file`, детерминированно делит найденные сцены на train/val с `seed=42`, проверяет CRS, bands, dtype, resolution, nodata и grid alignment. При успехе собирает два GDAL VRT XML со ссылками на абсолютные пути снимков.
