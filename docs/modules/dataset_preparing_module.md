# Модуль dataset_preparing

## Назначение

`dataset_preparing` готовит локальный raster-датасет к нарезке тайлов: читает список сцен, находит подготовленные снимки в `images_dir`, считает объекты по разметке, строит train/val split и возвращает VRT XML для train и val без записи VRT на диск. Модуль принимает только локальные пути и не использует storage/S3.

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

Модуль не использует публичные API других модулей. Входы текущей реализации принимаются как локальные пути и читаются через `Path`; отдельный модуль доступа к хранилищу не используется.

## Алгоритм работы и его особенности

Модуль читает `scenes_file`, индексирует `.tif/.tiff` в `images_dir`, сопоставляет сцены по имени, stem, casefold и нормализованному ключу. Тяжелую нормализацию исходных снимков модуль не выполняет: одноразовая подготовка GeoTIFF в `EPSG:3857` с internal mask делается CLI-скриптом `mlsystem2.cli.prepare_images_for_vrt`. Ошибками считаются отсутствующие снимки, невозможность открыть raster, CRS не `EPSG:3857`, отсутствие usable mask или nodata, несовместимые bands или dtype, некорректный geotransform и ошибка `gdalbuildvrt`. При успехе модуль через `gdalbuildvrt` строит VRT из подготовленных снимков, возвращает XML string и дает GDAL учитывать source masks/nodata.
