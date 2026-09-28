# Модуль dataset_preparing

## Назначение

`dataset_preparing` проверяет локальный датасет, сопоставляет разметку с подготовленными TIFF и возвращает независимый список сцен для нарезки тайлов. Модуль не создаёт мозаики и не выполняет train/val split.

## Публичный интерфейс

- `prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult` — подготовить legacy binary, per-image binary, legacy multiclass или manifest-backed per-image multiclass датасет.
- `annotation_regions(payload: dict, class_slugs: list[str] | None = None) -> list[AnnotationRegion] | None` — независимые зоны с учётом пересечения покрытий классов; `None` означает отсутствие зон, пустой список — отсутствие общей территории.
- `resolve_scene_images(request: SceneImageResolutionRequest) -> SceneImageResolution` — сопоставить legacy TXT либо per-image GeoJSON с TIFF.
- `per_image_annotation_name(image_path: str) -> str` — получить имя `<родительская_папка>_<stem>.geojson`.
- `per_image_footprint_name(image_path: str) -> str` и `footprint_name_for_annotation(annotation_file: str) -> str` — получить имя companion-футпринта `*_footprint.geojson`.
- `is_per_image_footprint_name(value: str) -> bool` и `per_image_annotation_files(annotations_dir: str) -> list[str]` — отличить companion-файлы от supervision-разметки.

## Публичные контракты

- `DatasetPreparationError` — невосстановимая ошибка подготовки.
- `DatasetClassRequest` — `slug`, `name`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `DatasetPreparationRequest` — `images_dir`, optional legacy-поля `scenes_file`, `annotation_file`, `hard_negative_annotation_file`, optional `annotations_dir`, optional `classes`, `val_fraction`, `expected_band_count`, `expected_dtype`, `expected_band_names`, `allow_rgb_alpha=false` (разрешить RGB+alpha при ожидаемых трёх каналах); задаётся ровно один из трёх режимов.
- `DatasetClassAnnotation` — `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `AnnotationRegion` — `zone_id`, полигональная GeoJSON `geometry` в CRS входного файла.
- `PreparedScene` — `scene_id`, `image_path`, optional локальные `annotation_file`, `footprint_file`; для зоны optional `parent_scene_id`, `zone_id`, `region_geometry`, `region_window=(x,y,width,height)` в исходной пиксельной сетке.
- `DatasetManifest`, `DatasetClassDefinition`, `DatasetSourceRevision` — строгая схема `.mlsystem2-dataset.json`, классы, ревизии исходных папок, идентификатор сборки и baseline-хеши.
- `PreparedDataset` — `format=legacy_binary|per_image_binary|legacy_multiclass|per_image_multiclass`, непустой `scenes`, optional общие `annotation_file`, `hard_negative_annotation_file`, `class_annotations` либо manifest-классы `classes`.
- `DatasetSceneReport` — `scene_id`, optional `image_path`, `positive_objects`, `hard_negative_objects`, `object_count`, `class_counts`, `annotation_zone_count=0`, `training_scene_count=1`.
- `DatasetPreparationReport` — `status`, счётчики сцен и объектов, `band_count`, `dtypes`, `scenes`, `missing_files`, `errors`, `warnings`.
- `DatasetPreparationResult` — `dataset`, `report`.
- `SceneImageResolutionRequest` — `images_dir` и ровно одно из `scenes_file`/`annotations_dir`; `annotation_files` допустимы только для legacy.
- `ResolvedSceneImage` — `scene_id`, `image_path`, optional `annotation_file`, `footprint_file`, `request_scenes`.
- `SceneImageResolution` — `input_scene_count`, `images`, `missing_scenes`, `ambiguous_scenes`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. `rasterio` проверяет TIFF, `shapely` разбирает геометрию, локальные файлы читаются через `Path`.

## Алгоритм работы и его особенности

Роль `annotation_zone` не участвует в счётчиках объектов. Подготовка per-image пересекает полный контур зоны с TIFF и разворачивает его в виртуальную сцену без новых растров; внешняя часть зоны не увеличивает пиксельное окно. Пересечения зон сохраняются отдельно. В multiclass территория ограничивается покрытием всех классов. Пустое пересечение пропускается с диагностикой, полностью пустой результат — ошибка. `resolve_scene_images` остаётся физическим поснимочным сопоставлением.

`DatasetPreparationRequest.allow_rgb_alpha=false` сохраняет прежний контракт; true разрешает ожидаемые три RGB-канала в четырёхканальном TIFF только при явном ColorInterp.alpha четвёртого канала. Счётчики и проверки типов/имён относятся к модельным RGB-каналам.

Legacy binary читает TXT, включая записи-папки и старые scene id, разрешает неоднозначность по геометрии и возвращает каждый TIFF отдельной сценой. Legacy multiclass объединяет сцены классов и назначает `class_id=1..N`. Per-image режим индексирует прямые файлы разметки в `annotations_dir` и TIFF рекурсивно; имя сопоставляется строго как `<parent>_<stem>.geojson`, а парный `<parent>_<stem>_footprint.geojson` содержит valid-data footprint и никогда не читается как supervision. Для обратной совместимости отсутствующий footprint не делает старый набор невалидным. Все TIFF проверяются по CRS, каналам, dtype и nodata/mask. Когда вызывающий next-gen задаёт `expected_band_names=RED,GRN,BLU,NIR`, явные описания каналов обязаны точно совпасть; полное отсутствие описаний принимает архитектурный порядок с предупреждением, частичный или противоречащий порядок является ошибкой. Общие VRT не создаются.

Коллизия, отсутствующий или неоднозначный TIFF являются ошибкой. Если рядом находится
`.mlsystem2-dataset.json`, формат становится `per_image_multiclass`: manifest и повторённые в каждом GeoJSON
поля схемы проверяются строго, positive требует известный класс, а feature ID и origin key обязательны и
уникальны. В обычном multiclass hard negative не может иметь класс; в управляемом manifest он может быть общим
либо ограниченным известным slug. Без manifest сохраняется прежняя binary-семантика, включая отсутствие роли
как `positive`. GeoJSON обязан быть `FeatureCollection` в CRS TIFF с валидными Polygon/MultiPolygon. Пустая
коллекция допустима, но полностью пустой датасет не готов к обучению.
