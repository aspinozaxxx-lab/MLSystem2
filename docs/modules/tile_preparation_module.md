# Модуль tile_preparation

## Назначение

`tile_preparation` создаёт train/val DataLoader по списку независимых TIFF. Тайлы каждого снимка строятся отдельно; перекрывающиеся снимки не объединяются, файлы тайлов на диск не записываются.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> object` — загрузить `settings.tile_preparation`, создать Dataset и PyTorch DataLoader.

Batch содержит `images: float32[B,C,H,W]`, binary `masks: float32[B,1,H,W]` либо multiclass `masks: long[B,H,W]` и `batch_meta` со счётчиками категорий/аугментаций. Значения mask: `-2` — nodata/ignore, `-1` — hard negative, `0` — фон, `1` либо `1..N` — positive/class id.

## Публичные контракты

- `TilePreparationError` — ошибка подготовки loader.
- `HARD_NEGATIVE_LABEL=-1` — служебная метка hard-negative пикселя.
- `TileClassAnnotation` — `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `TileSceneSource` — `scene_id`, `image_path`, optional per-image `annotation_file`.
- `TileSplitRequest` — `val_fraction`, `seed`.
- `TileDataloaderRequest` — непустой `scenes`, optional общие binary-файлы, `class_annotations`, `batch_size`, `mode`, optional `tile_split`, `max_batches_per_epoch`, `include_object_instances`. Допустим ровно один режим: общий binary, per-image binary либо legacy multiclass; per-image multiclass запрещён.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — параметры сетки, workers, sampling, augmentation и val-cache.
- `rasterio` — ленивое чтение окон отдельных TIFF с `boundless=True`.
- `shapely`, `rasterio.features` — пространственный индекс и rasterize разметки.
- `torch.utils.data` — sampler, DataLoader и фиксированный val subset.

## Алгоритм работы и его особенности

Для каждой сцены строится сетка `0,stride,...`; крайнее окно сохраняет `tile_size` и дополняется nodata. Coarse/sparse valid-footprint фильтр заранее удаляет black/nodata окна без полного чтения всех тайлов. Дескрипторы TIFF открываются лениво и удерживаются ограниченным LRU на worker. Разметка сцены индексируется отдельно; positive перекрывает hard negative, а nodata по значению либо raster mask получает `NODATA_LABEL=-2` и не превращается в обучающий фон. Split вычисляется хешем `seed+scene_id+x+y`, поэтому не зависит от порядка сцен. Train использует category-aware sampling и аугментации positive/hard-negative. Val выбирает фиксированный balanced subset и кэширует его в RAM при безопасном лимите, иначе лениво читает те же индексы. Пересекающиеся TIFF дают независимые тайлы.
