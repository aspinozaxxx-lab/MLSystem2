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

Для каждой сцены строится сетка полезных центров `0,stride,...`. Полное входное окно начинается в `-context` и имеет размер `tile_size`; его полезный центр имеет размер `tile_size - 2 × context`. Поэтому первые и последние пиксели TIFF попадают в полезную область, а внешняя рамка читается через `boundless=True` как nodata. `context=0` полностью сохраняет прежнюю сетку. Sampling-категория определяется только по геометрии полезного центра. Настройки обязаны удовлетворять `tile_size > 2 × context`.

Coarse/sparse valid-footprint фильтр заранее удаляет полностью black/nodata окна без полного чтения всех тайлов, а частично невалидные окна остаются. Дескрипторы TIFF открываются лениво и удерживаются ограниченным LRU на worker. Разметка сцены индексируется отдельно; positive перекрывает hard negative. В multiclass слои растеризуются в порядке приоритета и дают `int64` mask `-1/0/1..N`. Для управляемого датасета class-specific hard negative дополнительно растеризуется в синхронный boolean-тензор `[N,H,W]`: канал задаёт только тот тип, для которого область является отрицательной, а hard negative без класса остаётся общим `-1`. Nodata по значению, нулевая `dataset_mask` и boundless-padding объединяются в одну маску: изображение на ней получает исходное значение nodata, а supervision и class-specific masks принудительно получают фон `0`, подрезая разметку по фактическому снимку. Геометрические аугментации преобразуют все маски синхронно с изображением и target, после фотометрических изменений nodata восстанавливается. Split вычисляется хешем `seed+scene_id+x+y`, поэтому не зависит от порядка сцен. Train использует category-aware sampling и аугментации positive/hard-negative; при `class_balance=true` positive-веса выравниваются по типам, а дефицит отражается предупреждением. Val выбирает фиксированный balanced subset и кэширует его в RAM при безопасном лимите, иначе лениво читает те же индексы. Диагностика содержит разрешение и числа candidate/valid/positive/hard-negative/background окон по каждому TIFF. Пересекающиеся TIFF дают независимые тайлы.

`prefetch_epochs` задаёт целевой объём готовых batch, а `prefetch_factor` рассчитывается обратно
пропорционально `num_workers`; уменьшение числа процессов не сокращает заданный префетч. В серверном профиле
используется восемь workers и один epoch префетча. Worker-процессы получают признак
`MLSYSTEM2_TILE_WORKER=1`; при наличии `pause.request` в `MLSYSTEM2_TRAINING_CONTROL_DIR` они ждут
возобновления до открытия TIFF и подготовки следующего тайла, не очищая уже готовую очередь.
