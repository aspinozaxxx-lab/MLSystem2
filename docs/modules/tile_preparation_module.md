# Модуль tile_preparation

## Назначение

`tile_preparation` создаёт train/val DataLoader по списку независимых TIFF. Тайлы каждого снимка строятся отдельно; перекрывающиеся снимки не объединяются, файлы тайлов на диск не записываются.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> object` — загрузить `settings.tile_preparation`, создать Dataset и PyTorch DataLoader.

Batch содержит `images: float32[B,C,H,W]`, binary `masks: float32[B,1,H,W]` либо multiclass `masks: long[B,H,W]` и `batch_meta` со счётчиками категорий/аугментаций. Значения mask: `-1` — hard negative, `0` — фон, `1` либо `1..N` — positive/class id.

## Публичные контракты

- `TilePreparationError` — ошибка подготовки loader.
- `HARD_NEGATIVE_LABEL=-1` — служебная метка hard-negative пикселя.
- `TileClassAnnotation` — `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `TileClassDefinition` — `class_id`, `slug`, `name`, `color`, `priority` для class-filtered чтения одного per-image GeoJSON.
- `TileSceneSource` — `scene_id`, `image_path`, optional per-image `annotation_file`.
- `TileSplitRequest` — `val_fraction`, `seed`.
- `TileDataloaderRequest` — непустой `scenes`, optional общие binary-файлы, legacy `class_annotations` либо per-image `classes`, `batch_size`, `mode`, optional `tile_split`, `max_batches_per_epoch`, `include_object_instances`. Допустим ровно один режим: общий binary, per-image binary, legacy multiclass либо per-image multiclass.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — параметры сетки, workers, sampling, augmentation и val-cache.
- `rasterio` — ленивое чтение окон отдельных TIFF с `boundless=True`.
- `shapely`, `rasterio.features` — пространственный индекс и rasterize разметки.
- `torch.utils.data` — sampler, DataLoader и фиксированный val subset.

## Алгоритм работы и его особенности

Для каждой сцены строится сетка `0,stride,...`; крайнее окно сохраняет `tile_size` и дополняется nodata. Coarse/sparse valid-footprint фильтр заранее удаляет полностью black/nodata окна без полного чтения всех тайлов, а частично невалидные окна остаются. Дескрипторы TIFF открываются лениво и удерживаются ограниченным LRU на worker. Разметка сцены индексируется отдельно; positive перекрывает hard negative. В multiclass слои растеризуются в порядке приоритета и дают `int64` mask `-1/0/1..N`. Nodata по значению, нулевая `dataset_mask` и boundless-padding объединяются в одну маску: изображение на ней получает исходное значение nodata, а supervision mask принудительно получает фон `0`, подрезая positive и hard negative по фактическому снимку. Геометрические аугментации преобразуют эту маску синхронно с изображением и target, после фотометрических изменений nodata восстанавливается. Split вычисляется хешем `seed+scene_id+x+y`, поэтому не зависит от порядка сцен. Train использует category-aware sampling и аугментации positive/hard-negative; при `class_balance=true` positive-веса выравниваются по типам, а дефицит отражается предупреждением. Val выбирает фиксированный balanced subset и кэширует его в RAM при безопасном лимите, иначе лениво читает те же индексы. Пересекающиеся TIFF дают независимые тайлы.
