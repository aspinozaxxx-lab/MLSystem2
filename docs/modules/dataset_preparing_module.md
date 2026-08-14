# Модуль dataset_preparing

## Назначение

`dataset_preparing` проверяет локальный датасет, сопоставляет разметку с подготовленными TIFF и возвращает независимый список сцен для нарезки тайлов. Модуль не создаёт мозаики и не выполняет train/val split.

## Публичный интерфейс

- `prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult` — подготовить legacy binary, per-image binary, legacy multiclass или manifest-backed per-image multiclass датасет.
- `resolve_scene_images(request: SceneImageResolutionRequest) -> SceneImageResolution` — сопоставить legacy TXT либо per-image GeoJSON с TIFF.
- `per_image_annotation_name(image_path: str) -> str` — получить имя `<родительская_папка>_<stem>.geojson`.
- `per_image_footprint_name(image_path: str) -> str` и `footprint_name_for_annotation(annotation_file: str) -> str` — получить имя companion-футпринта `*_footprint.geojson`.
- `is_per_image_footprint_name(value: str) -> bool` и `per_image_annotation_files(annotations_dir: str) -> list[str]` — отличить companion-файлы от supervision-разметки.

## Публичные контракты

- `DatasetPreparationError` — невосстановимая ошибка подготовки.
- `DatasetClassRequest` — `slug`, `name`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `DatasetPreparationRequest` — `images_dir`, optional legacy-поля `scenes_file`, `annotation_file`, `hard_negative_annotation_file`, optional `annotations_dir`, optional `classes`, `val_fraction`, `expected_band_count`, `expected_dtype`; задаётся ровно один из трёх режимов.
- `DatasetClassAnnotation` — `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `PreparedScene` — `scene_id`, `image_path`, optional локальные `annotation_file` и `footprint_file`.
- `DatasetManifest`, `DatasetClassDefinition`, `DatasetSourceRevision` — строгая схема `.mlsystem2-dataset.json`, классы, ревизии исходных папок, идентификатор сборки и baseline-хеши.
- `PreparedDataset` — `format=legacy_binary|per_image_binary|legacy_multiclass|per_image_multiclass`, непустой `scenes`, optional общие `annotation_file`, `hard_negative_annotation_file`, `class_annotations` либо manifest-классы `classes`.
- `DatasetSceneReport` — `scene_id`, optional `image_path`, `positive_objects`, `hard_negative_objects`, `object_count`.
- `DatasetPreparationReport` — `status`, счётчики сцен и объектов, `band_count`, `dtypes`, `scenes`, `missing_files`, `errors`.
- `DatasetPreparationResult` — `dataset`, `report`.
- `SceneImageResolutionRequest` — `images_dir` и ровно одно из `scenes_file`/`annotations_dir`; `annotation_files` допустимы только для legacy.
- `ResolvedSceneImage` — `scene_id`, `image_path`, optional `annotation_file`, `footprint_file`, `request_scenes`.
- `SceneImageResolution` — `input_scene_count`, `images`, `missing_scenes`, `ambiguous_scenes`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. `rasterio` проверяет TIFF, `shapely` разбирает геометрию, локальные файлы читаются через `Path`.

## Алгоритм работы и его особенности

Legacy binary читает TXT, включая записи-папки и старые scene id, разрешает неоднозначность по геометрии и возвращает каждый TIFF отдельной сценой. Legacy multiclass объединяет сцены классов и назначает `class_id=1..N`. Per-image режим индексирует прямые файлы разметки в `annotations_dir` и TIFF рекурсивно; имя сопоставляется строго как `<parent>_<stem>.geojson`, а парный `<parent>_<stem>_footprint.geojson` содержит valid-data footprint и никогда не читается как supervision. Для обратной совместимости отсутствующий footprint не делает старый набор невалидным. Коллизия, отсутствующий или неоднозначный TIFF являются ошибкой. Если рядом находится `.mlsystem2-dataset.json`, формат становится `per_image_multiclass`: manifest и повторённые в каждом GeoJSON разметки `_mlsystem2_schema_version`, `_mlsystem2_task`, `_mlsystem2_classes` проверяются строго, positive требует известный `_mlsystem2_class`, hard negative не может иметь класс, а feature ID и `_mlsystem2_origin_key` обязательны и уникальны. Без manifest сохраняется прежняя binary-семантика, включая отсутствие роли как `positive`. GeoJSON разметки обязан быть `FeatureCollection` в CRS TIFF с валидными `Polygon/MultiPolygon`. Пустой `FeatureCollection` допустим, пустой датасет не готов к обучению. Все TIFF проверяются по CRS, каналам, dtype и nodata/mask; общие VRT не создаются.
