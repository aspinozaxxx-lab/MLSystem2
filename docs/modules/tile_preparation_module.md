# Модуль tile_preparation

## Назначение

`tile_preparation` создает `torch.utils.data.DataLoader` по одному VRT XML и binary или multiclass GeoJSON-разметке. Модуль отвечает только за формирование train/val loader для уже подготовленных VRT, не выполняет split и не готовит все tiles заранее.

`create_tile_dataloader` должен возвращаться быстро. Чтение raster data, определение nodata pixels и rasterize mask выполняются лениво в `Dataset.__getitem__`, то есть в основном процессе или в PyTorch DataLoader workers.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> torch.utils.data.DataLoader` - загружает текущие настройки `settings.tile_preparation`, создает `Dataset` и возвращает DataLoader.

Batch DataLoader:
- `images: torch.float32 [B, C, tile_size, tile_size]`;
- binary `masks: torch.float32 [B, 1, tile_size, tile_size]`, values `0/1`;
- multiclass `masks: torch.long [B, tile_size, tile_size]`, values `0=background`, `1..N=class id`;
- `batch_meta: dict` с полями `augmented_tile_count`, `positive_tile_count`, `class_positive_tile_counts`, `class_pixel_counts`, `tile_augmented`, `tile_positive`.

## Публичные контракты

- `TilePreparationError` - ошибка подготовки tile DataLoader.
- `TileClassAnnotation` - поля `class_id`, `slug`, `name`, `annotation_file`, `priority`.
- `TileDataloaderRequest` - поля `vrt_xml`, `annotation_file`, `class_annotations`, `batch_size`, `mode`. Валидация: либо задан `annotation_file` и `class_annotations=[]`, либо задан непустой `class_annotations` и `annotation_file=None`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` - получить `tile_size`, `stride`, `num_workers`, `prefetch_factor`, `seed`, `augmentation_level`, `smart_tiling`, `positive_factor`, `val_positive_factor`, `class_balance`.
- `rasterio` - открыть VRT и лениво читать image windows с `boundless=True`.
- `shapely` и `rasterio.features` - загрузить GeoJSON и rasterize mask в окно tile.
- `torch.utils.data` - создать Dataset/DataLoader и обеспечить prefetch через `num_workers` и `prefetch_factor`.

## Алгоритм работы и его особенности

При создании Dataset разрешено:
- открыть VRT;
- прочитать metadata: `width`, `height`, `count`, CRS, nodata;
- построить список окон по VRT/source rects;
- построить coarse valid-data footprint одним низкоразрешенным чтением VRT и отфильтровать black/nodata-only окна.

При создании Dataset запрещено:
- читать raster data по всем окнам;
- rasterize masks по всем окнам;
- заранее готовить все tiles или batches.

Окна строятся регулярной Geoalert-compatible сеткой: `0, stride, 2*stride, ...` до границы source rect или raster. Shifted last tile не добавляется. Окно всегда имеет размер `tile_size x tile_size`; выход за bounds закрывается `rasterio.read(boundless=True, fill_value=nodata)`.

После построения candidate windows модуль строит внутренний coarse valid-data footprint с фиксированным шагом `64` пикселя: сначала читает masks VRT в низком разрешении, затем низкоразрешенные raw values и считает valid cell только там, где mask valid и хотя бы один канал не равен нулю с eps `1e-6`. Candidate window должен пересекать valid cell. Затем выполняется точная sparse-проверка только по глобальным пикселям, которые соответствуют диагностической сетке tile (`0, 64, ..., center, last`). Это не читает каждый tile целиком и убирает black/nodata-only окна до DataLoader.

В `__getitem__` image читается через rasterio с `out_shape=(count, tile_size, tile_size)`, приводится только к `float32` и не нормализуется. Channel order сохраняет порядок каналов raster/VRT.

Binary mask rasterize выполняется в том же окне, возвращает shape `1,H,W`, dtype `float32`, значения `0/1`. Multiclass mask rasterize загружает spatial index для каждого класса, создает `int64 [H,W]`, а background оставляет `0`. Классы применяются по `(priority, class_id)`: меньший priority записывается раньше, больший priority перекрывает его в пересечениях; при равном priority более поздний `class_id` перекрывает более ранний. Nodata pixels зануляются в background. Это делает пересечения разметки детерминированными без падения DataLoader.

Аугментация сохраняет raw Geoalert tensor ABI подготовленных снимков: после photometric/cutout значения image зажимаются в диапазон `0..255`. Geometric flips/rotations применяются к image и mask, photometric только к image, cutout зануляет image patch и переводит тот же patch mask в background `0`. Sample возвращает `{"augmented": bool, "positive": bool}` и для multiclass дополнительно `class_positive` и `class_pixels`; collate собирает batch `(images, masks, batch_meta)` с aggregate-счетчиками и per-tile flags для диагностики.

При `smart_tiling=true` Dataset строит cheap-index для `train` и `val`: по bounds окна проверяет пересечение с GeoJSON geometry без чтения raster data и без rasterize. Для multiclass positive hint означает пересечение с геометрией любого класса. Поля `estimated_positive_tiles`, `estimated_negative_tiles` и `estimated_class_positive_tiles` являются geometry-intersection hint, а не точным rasterized mask count. Для `train`, если есть positive/negative hints, DataLoader использует `WeightedRandomSampler`: суммарный вес positive окон равен `positive_factor`, суммарный вес negative окон равен `1 - positive_factor`; иначе остается обычный sampler. Если `class_balance=true` и задан multiclass dataset, positive-доля делится между классами с positive windows, а окно с несколькими классами получает сумму class budgets. Классы без positive windows или с очень малым числом windows пишутся в warnings. Аугментация применяется только к positive tiles. Для `val` shuffle и augmentation выключены; если `val_positive_factor` задан, используется deterministic weighted sampler только для диагностической validation выборки, иначе val остается последовательным.

Полностью black/nodata-only tiles фильтруются заранее через coarse valid-data footprint и не попадают в DataLoader. Диагностика Dataset доступна как внутренние attributes: `candidate_window_count_before_valid_filter`, `black_filtered_window_count`, `valid_footprint_stride`, `valid_footprint_valid_cells`, `valid_footprint_total_cells`.
