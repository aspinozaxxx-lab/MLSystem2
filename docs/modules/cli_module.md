# Модуль cli

## Назначение

`cli` содержит точки входа командной строки и локальные диагностические скрипты MLSystem2.

## Публичный интерфейс

- `python -m mlsystem2.cli.prepare_images [--mode local|server]` — одноразово подготовить COG GeoTIFF.
- `python -m mlsystem2.cli.tiling_test_for_black --config <path>` — проверить train/val тайлы на чёрные и non-finite данные.
- `mlsystem2-train` — запустить обучение.
- `mlsystem2-infer` — запустить inference.

Устаревшей команды `prepare_images_for_vrt` нет.

## Публичные контракты

Модуль не объявляет DTO и не имеет `contracts.py`.

## Список используемых данным модулем модулей и с какой целью

- `dataset_preparing.api`, `tile_preparation.api`, `settings.api` — диагностика подготовки и тайлов.
- `settings.api`, `train_pipeline.api`, `inference_pipeline.api` — точки входа train/infer.

## Алгоритм работы и его особенности

`prepare_images` читает `.tif/.tiff`, перепроецирует в `EPSG:3857` с nearest resampling и сохраняет COG с исходными каналами, dtype, nodata, описаниями и тегами; alpha-интерпретация снимается без удаления канала. Относительная структура каталогов сохраняется, результат каждого файла попадает в JSON-отчёт. Локальный режим использует `D:\Projects\ImagesDeforestation` и отчёт `D:\Projects\test\prepare_images_report.json`; серверный — `s3://mlsystems/images/kanopus/`, `/data/mlsystem2/prepared_images/kanopus/` и `report/prepare_images_report.json`. `modules_test` пишет отчёты и до 100 batch, не создавая мозаик. `tiling_test_for_black` повторяет production split по независимым сценам и возвращает код `1` при пустом/non-finite тайле.
