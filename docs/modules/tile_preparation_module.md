# Модуль tile_preparation

## Назначение

`tile_preparation` создает `torch.utils.data.DataLoader` по одному VRT XML и одному GeoJSON-файлу разметки. Модуль отвечает только за формирование train/val loader для уже подготовленных VRT, не выполняет split и не готовит все tiles заранее.

`create_tile_dataloader` должен возвращаться быстро. Чтение raster data, определение nodata pixels и rasterize mask выполняются лениво в `Dataset.__getitem__`, то есть в основном процессе или в PyTorch DataLoader workers.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> torch.utils.data.DataLoader` - загружает текущие настройки `settings.tile_preparation`, создает `Dataset` и возвращает DataLoader.

Batch DataLoader:
- `images: torch.float32 [B, C, tile_size, tile_size]`;
- `masks: torch.float32 [B, 1, tile_size, tile_size]`;
- `batch_meta: dict` с полями `augmented_tile_count`, `positive_tile_count`, `tile_augmented`, `tile_positive`;
- mask binary `0/1`.

## Публичные контракты

- `TilePreparationError` - ошибка подготовки tile DataLoader.
- `TileDataloaderRequest` - поля `vrt_xml`, `annotation_file`, `batch_size`, `mode`.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` - получить `tile_size`, `stride`, `num_workers`, `prefetch_factor`, `seed`, `augmentation_level`, `smart_tiling`.
- `rasterio` - открыть VRT и лениво читать image windows с `boundless=True`.
- `shapely` и `rasterio.features` - загрузить GeoJSON и rasterize mask в окно tile.
- `torch.utils.data` - создать Dataset/DataLoader и обеспечить prefetch через `num_workers` и `prefetch_factor`.

## Алгоритм работы и его особенности

При создании Dataset разрешено:
- открыть VRT;
- прочитать metadata: `width`, `height`, `count`, CRS, nodata;
- построить список окон по VRT/source rects.

При создании Dataset запрещено:
- читать raster data по всем окнам;
- rasterize masks по всем окнам;
- заранее готовить все tiles или batches.

Окна строятся регулярной Geoalert-compatible сеткой: `0, stride, 2*stride, ...` до границы source rect или raster. Shifted last tile не добавляется. Окно всегда имеет размер `tile_size x tile_size`; выход за bounds закрывается `rasterio.read(boundless=True, fill_value=nodata)`.

В `__getitem__` image читается через rasterio с `out_shape=(count, tile_size, tile_size)`, приводится только к `float32` и не нормализуется. Channel order сохраняет порядок каналов raster/VRT. Mask rasterize выполняется в том же окне, возвращает shape `1,H,W`, dtype `float32`, значения `0/1`. Mask зануляется там, где все image channels равны nodata, чтобы padding/nodata не попадал в target. Sample возвращает `{"augmented": bool, "positive": bool}`, а collate собирает batch `(images, masks, batch_meta)` с aggregate-счетчиками и per-tile flags для диагностики.

При `smart_tiling=true` и `mode="train"` Dataset строит cheap-index: по bounds окна проверяет пересечение с GeoJSON geometry без чтения raster data и без rasterize. Если есть positive/negative hints, DataLoader использует `WeightedRandomSampler` с примерно равной суммарной вероятностью positive и negative окон; иначе остается обычный sampler. Аугментация применяется только к positive tiles. Для `val` sampler deterministic, shuffle и augmentation выключены.

Полностью nodata tiles не фильтруются заранее: они могут попасть в DataLoader как image с nodata-fill и zero mask. Это сохраняет быстрый старт loader и не меняет batch contract.
