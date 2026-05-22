# Модуль cli

## Назначение

`cli` содержит точки входа командной строки для запуска модулей через `python -m` и консольные scripts.

## Публичный интерфейс

- `python -m mlsystem2.cli.prepare_images_for_vrt` — одноразовая подготовка исходных GeoTIFF к построению VRT-мозаик.
- `mlsystem2-train` — запуск обучающего конвейера.
- `mlsystem2-infer` — запуск инференса.

## Публичные контракты

Модуль не объявляет DTO и не имеет `contracts.py`.

## Список используемых данным модулем модулей и с какой целью

- `dataset_preparing.api`, `tile_preparation.api`, `settings.api` — локальная диагностика модулей из `modules_test`.
- `settings.api`, `train_pipeline.api`, `inference_pipeline.api` — существующие точки входа train и infer.

## Алгоритм работы и его особенности

CLI разбирает аргументы, вызывает публичный API нужного модуля и завершает процесс с кодом, соответствующим результату. `modules_test` — служебный локальный диагностический скрипт, не публичный API и не основной CLI приложения; он пишет `preparation_report.json`, `modules_test_timing_report.json`, `train.vrt`, `val.vrt` и до 100 batch в `tile_batches` при успешной подготовке датасета и DataLoader.

Подготовка снимков для VRT

`src/mlsystem2/cli/prepare_images_for_vrt.py` — служебный CLI-скрипт для тяжелой одноразовой подготовки растров перед запуском `dataset_preparing`. Он не является публичным API модуля и не используется как библиотека.

Скрипт читает исходные `.tif` и `.tiff`, перепроецирует их в `EPSG:3857` с `nearest` resampling и записывает подготовленные Cloud Optimized GeoTIFF. При этом он сохраняет количество каналов, порядок каналов, dtype, nodata, описания каналов и теги каналов. Alpha band и internal mask не создаются, `.msk` sidecar не создается. Если в исходном снимке alpha указан только как `ColorInterp`, он заменяется на `undefined` на той же позиции, а сам канал остается обычным спектральным каналом.

Результат подготовки — набор COG GeoTIFF в целевой директории с сохранением относительной структуры папок и JSON-отчет со статусом по каждому файлу. Эти подготовленные снимки затем используются `dataset_preparing` для построения train/val VRT без повторного тяжелого warp исходных raw-снимков.

Запуск локального режима:

```bash
python -m mlsystem2.cli.prepare_images_for_vrt
python -m mlsystem2.cli.prepare_images_for_vrt --mode local
```

Локальный режим читает `D:\Projects\ImagesDeforestation`, пишет в `D:\Projects\ImagesDeforestationPrepared3857`, отчет пишет в `D:\Projects\test\prepare_images_for_vrt_report.json`.

Запуск серверного режима:

```bash
python -m mlsystem2.cli.prepare_images_for_vrt --mode server
```

Серверный режим читает снимки из `s3://mlsystems/images/kanopus/`, пишет подготовленные COG в `/data/mlsystem2/prepared_images/kanopus/`, отчет пишет в `/data/mlsystem2/prepared_images/report/prepare_images_for_vrt_report.json`.
