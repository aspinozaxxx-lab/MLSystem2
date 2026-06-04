# Модуль dataset_preparing

## Назначение

`dataset_preparing` готовит локальный raster-датасет к нарезке тайлов: читает списки сцен, находит подготовленные снимки в `images_dir`, считает объекты по разметке и возвращает общий VRT XML выбранного пула сцен без записи VRT на диск. Train/val split выполняется по тайлам в `tile_preparation`. Модуль принимает только локальные пути и не использует storage/S3.

## Публичный интерфейс

- `prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult` — выполняет подготовку датасета по локальным путям, binary или multiclass разметке, доле `val_fraction` и опциональному `negative_scene_limit`.

## Публичные контракты

- `DatasetPreparationError` — исключение невосстановимой ошибки.
- `DatasetClassRequest` — поля `slug`, `name`, `scenes_file`, `annotation_file`, `priority`.
- `DatasetPreparationRequest` — поля `images_dir`, `scenes_file`, `annotation_file`, `classes`, `val_fraction`, `negative_scene_limit`. Валидация: либо binary `scenes_file` + `annotation_file`, либо multiclass `classes`; смешивать режимы нельзя.
- `DatasetClassAnnotation` — поля `class_id`, `slug`, `name`, `annotation_file`, `priority`.
- `PreparedDataset` — поля `train_vrt_xml`, `val_vrt_xml`, `pool_vrt_xml`, `annotation_file`, `class_annotations`. В binary режиме `annotation_file` заполнен и `class_annotations=[]`; в multiclass режиме `annotation_file=None` и `class_annotations` заполнен id `1..N`. В tile-режиме `pool_vrt_xml` заполнен, а train/val VRT равны общему VRT.
- `DatasetSceneReport` — поля `scene_id`, `image_path`, `object_count`, `split` (`train`, `val`, `pool`, `excluded`, `missing`).
- `DatasetPreparationReport` — поля `status`, `negative_scene_limit`, `selected_positive_scenes_count`, `selected_negative_scenes_count`, `scenes_total`, `scenes_found`, `objects_total`, `train_scenes_count`, `train_objects_count`, `val_scenes_count`, `val_objects_count`, `scenes`, `missing_files`, `errors`.
- `DatasetPreparationResult` — поля `dataset`, `report`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. Входы текущей реализации принимаются как локальные пути и читаются через `Path`; отдельный модуль доступа к хранилищу не используется.

## Алгоритм работы и его особенности

В binary режиме модуль читает один `scenes_file`, индексирует `.tif/.tiff` в `images_dir`, разворачивает записи-папки в относительные пути всех снимков внутри найденной папки, сопоставляет сцены по относительному пути, имени, stem, casefold и нормализованному ключу, считает объекты по одному `annotation_file` и возвращает `PreparedDataset.annotation_file`. Если scene id неоднозначно найден в нескольких подпапках, модуль выбирает снимок по пересечению с геометрией GeoJSON, а при отсутствии пересечения — ближайший к области разметки.

В multiclass режиме модуль читает `scenes_file` каждого класса, разворачивает записи-папки и использует результат только для сборки единого пула scene id с сохранением порядка первого появления. После этого снимки ищутся один раз по общему пулу. Объекты считаются по каждому GeoJSON на всем общем пуле сцен, а не только на сценах, перечисленных в `scenes_file` этого класса. `PreparedDataset.class_annotations` возвращает список разметок с `class_id` по порядку config: `1..N`, и переносит `priority` без изменения.

После подсчета объектов сцены с `object_count > 0` считаются positive. Если задан `negative_scene_limit`, модуль сохраняет все positive scenes и добавляет до указанного числа zero-object scenes детерминированной случайной выборкой. Выбранный пул не делится по сценам: для него строится один общий VRT, а scene report помечает выбранные сцены как `pool`, остальные найденные zero-object сцены как `excluded`.

Тяжелую нормализацию исходных снимков модуль не выполняет: одноразовая подготовка GeoTIFF в `EPSG:3857` с internal mask делается CLI-скриптом `mlsystem2.cli.prepare_images_for_vrt`. Ошибками считаются отсутствующие снимки, невозможность открыть raster, CRS не `EPSG:3857`, отсутствие usable mask или nodata, несовместимые bands или dtype, некорректный geotransform и ошибка `gdalbuildvrt`. При успехе модуль через `gdalbuildvrt` строит VRT из подготовленных снимков, возвращает XML string и дает GDAL учитывать source masks/nodata.
