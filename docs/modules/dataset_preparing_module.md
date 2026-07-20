# Модуль dataset_preparing

## Назначение

`dataset_preparing` готовит локальный raster-датасет к нарезке тайлов: читает списки сцен, находит подготовленные снимки в `images_dir`, считает объекты по разметке и возвращает общий VRT XML выбранного пула сцен без записи VRT на диск. Train/val split выполняется по тайлам в `tile_preparation`. Модуль принимает только локальные пути и не использует storage/S3.

## Публичный интерфейс

- `prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult` — выполняет подготовку датасета по локальным путям, binary или multiclass разметке и доле `val_fraction`.

## Публичные контракты

- `DatasetPreparationError` — исключение невосстановимой ошибки.
- `DatasetClassRequest` — поля `slug`, `name`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `DatasetPreparationRequest` — поля `images_dir`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `classes`, `val_fraction`, optional `expected_band_count` и `expected_dtype`. Валидация: либо binary `scenes_file` + `annotation_file`, либо multiclass `classes`; смешивать режимы нельзя.
- `DatasetClassAnnotation` — поля `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `PreparedDataset` — поля `train_vrt_xml`, `val_vrt_xml`, `pool_vrt_xml`, `annotation_file`, optional `hard_negative_annotation_file`, `class_annotations`. В binary режиме `annotation_file` заполнен, optional `hard_negative_annotation_file` содержит абсолютный resolved путь при наличии и `class_annotations=[]`; в multiclass режиме `annotation_file=None` и `class_annotations` заполнен id `1..N` вместе с class-level hard-negative путями. В tile-режиме `pool_vrt_xml` заполнен, а train/val VRT равны общему VRT.
- `DatasetSceneReport` — поля `scene_id`, `image_path`, `positive_objects`, `hard_negative_objects`, `object_count`; `object_count = positive_objects + hard_negative_objects`.
- `DatasetPreparationReport` — поля `status`, `scenes_total`, `scenes_found`, `positive_objects`, `hard_negative_objects`, `objects_total`, `band_count`, `dtypes`, `scenes`, `missing_files`, `errors`; `objects_total = positive_objects + hard_negative_objects`.
- `DatasetPreparationResult` — поля `dataset`, `report`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. Входы текущей реализации принимаются как локальные пути и читаются через `Path`; отдельный модуль доступа к хранилищу не используется.

## Алгоритм работы и его особенности

В binary режиме модуль читает один `scenes_file`, индексирует `.tif/.tiff` в `images_dir`, разворачивает записи-папки в относительные пути всех снимков внутри найденной папки, сопоставляет сцены по относительному пути, имени, stem, casefold и нормализованному ключу, считает positive objects по `annotation_file`, hard-negative objects по optional `hard_negative_annotation_file` и возвращает resolved пути в `PreparedDataset`. Если scene id неоднозначно найден в нескольких подпапках, модуль выбирает снимок по пересечению с геометрией обоих GeoJSON, а при отсутствии пересечения — ближайший к области разметки.

В multiclass режиме модуль читает `scenes_file` каждого класса, разворачивает записи-папки и использует результат только для сборки единого пула scene id с сохранением порядка первого появления. После этого снимки ищутся один раз по общему пулу. Positive objects считаются по каждому class `annotation_file` на всем общем пуле сцен, hard-negative objects считаются по optional class `hard_negative_annotation_file`. `PreparedDataset.class_annotations` возвращает список positive-разметок с `class_id` по порядку config: `1..N`, переносит `priority` без изменения и добавляет class-level hard-negative путь при наличии.

После сопоставления снимков модуль строит один общий VRT по всем найденным снимкам. Пул не делится по сценам: train/val split выполняется только в `tile_preparation` по тайлам, а scene report не содержит split-статус.

Тяжелую нормализацию исходных снимков модуль не выполняет: одноразовая подготовка GeoTIFF в `EPSG:3857` делается CLI-скриптом `mlsystem2.cli.prepare_images_for_vrt`. Ошибками считаются отсутствующие снимки, невозможность открыть raster, CRS не `EPSG:3857`, отсутствие mask flags или nodata, несовпадение с `expected_band_count`/`expected_dtype`, несовместимые снимки, некорректный geotransform и ошибка `gdalbuildvrt`. Rasterio-флаг `all_valid` является допустимой маской для ортофото без nodata. При успехе модуль через `gdalbuildvrt` строит VRT из подготовленных снимков, возвращает XML string и дает GDAL учитывать source masks/nodata.
