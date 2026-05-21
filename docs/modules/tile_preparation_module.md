# Модуль tile_preparation

## Назначение

`tile_preparation` принимает `PreparedDataset` от `dataset_preparing` и готовит источники train/val батчей из `train_vrt_xml` и `val_vrt_xml`.

## Публичный интерфейс

- `build_tile_sources(request: TileSourceRequest) -> TileSourceBundle` — создает источники батчей по VRT XML и параметрам `tile_size`, `stride`, `batch_size`, `prefetch_workers`, `prefetch_batches`.

## Публичные контракты

- `TilePreparationError` — ошибка подготовки тайлов.
- `TileBatch` — поля `inputs`, `targets`, `scene_ids`, `metadata`.
- `TilePreparationReport` — поля `train_batches_prepared`, `val_batches_prepared`, `queue_capacity`, `worker_count`, `warnings`.
- `TileBatchSource` — протокол итератора батчей с `close` и `profile_snapshot`.
- `TileSourceRequest` — поля `dataset`, `tile_size`, `stride`, `batch_size`, `prefetch_workers`, `prefetch_batches`.
- `TileSourceBundle` — поля `train`, `val`, `report`.

## Список используемых данным модулем модулей и с какой целью

- `dataset_preparing.contracts` — публичный контракт `PreparedDataset`.

## Алгоритм работы и его особенности

Получает готовые VRT XML, открывает их in-memory через rasterio `MemoryFile` и возвращает источники батчей. Текущая нарезка батчей остается заглушкой: источники пустые, а реальное чтение окон из VRT будет следующим шагом.
