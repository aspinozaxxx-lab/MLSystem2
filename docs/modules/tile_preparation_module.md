# Модуль tile_preparation

## Назначение

`tile_preparation` создает один `torch.utils.data.DataLoader` по одному VRT XML и одному GeoJSON-файлу разметки.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> torch.utils.data.DataLoader` — загружает настройки, строит датасет тайлов для одного режима и возвращает DataLoader. Батч DataLoader содержит `images: torch.Tensor float32 [B, C, tile_size, tile_size]` и `masks: torch.Tensor float32 [B, 1, tile_size, tile_size]`, маска бинарная `0/1`.

## Публичные контракты

- `TilePreparationError` — ошибка подготовки tile DataLoader.
- `TileDataloaderRequest` — DTO запроса создания DataLoader:
  - `vrt_xml: str` — VRT XML одного набора данных;
  - `annotation_file: str | Path` — GeoJSON-разметка;
  - `batch_size: int` — размер батча для torch DataLoader; значение передает `train_pipeline` из `train.batch_size`;
  - `mode: Literal["train", "val"]` — режим одного loader; это не split. `train` включает shuffle и аугментации из настроек, `val` выключает shuffle и аугментации.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` — получить текущие настройки через `get_settings` и взять секцию `tile_preparation`.

## Алгоритм работы и его особенности

Получить настройки, открыть VRT через rasterio `MemoryFile`, загрузить GeoJSON, построить окна `tile_size`/`stride`, rasterize mask в `window_transform`, clip по valid/nodata mask, применить аугментации только в `train` mode при `augmentation_level > 0`, собрать torch DataLoader с `num_workers`, `prefetch_factor` и `seed`.
