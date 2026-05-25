# Обучение DeepLabV3Plus ResNet50 по вырубкам

## Старт запуска

- Дата старта: 2026-05-25 19:18 UTC.
- Серверный commit: `09a12cd67c69e4dac717e7fd343cdae7827e9498`.
- Модель: `smp_deeplabv3plus_resnet50`.
- Конфиг: `/opt/mlsystem2/runtime/first_train/deforestation_deeplabv3plus_resnet50_10h.yaml`.
- Train PID: `1665725`.
- GPU monitor PID: `1665726`.
- Лог train: `/opt/mlsystem2/runtime/first_train/logs/deforestation_deeplabv3plus_resnet50_10h_20260525T191822Z.log`.
- Лог GPU: `/opt/mlsystem2/runtime/first_train/logs/deforestation_deeplabv3plus_resnet50_10h_20260525T191822Z_gpu.log`.
- MLflow run_id: `d83b4937aaec4f7fbb406ba2ff786018`.
- MLflow run_name: `deforestation_2505_7`.

## Preflight

- `tests/test_public_contracts.py`: passed.
- `tests`: `96 passed`.
- `ruff check src ./tests`: passed.
- `list_supported_models`: содержит `smp_deeplabv3plus_resnet50`.
- `create_model`: успешно создает `segmentation_models_pytorch.DeepLabV3Plus` с encoder `resnet50`.
- GPU перед стартом: RTX 5090, train-процессов не было.

## Данные

- `images_dir`: `/data/mlsystem2/prepared_images/`.
- `scenes_file`: `/data/MLMarkup/Вырубки/deforestation.txt`.
- `annotation_file`: `/data/MLMarkup/Вырубки/deforestation.geojson`.
- Объектов разметки: `532`.
- Train: `source_rect_count=29`, `uses_vrt_source_rects=True`, `candidate_window_count_before_valid_filter=28603`, `black_filtered_window_count=4374`, `len_dataset=24229`, `estimated_positive_tiles=1549`, `estimated_negative_tiles=22680`.
- Val: `source_rect_count=6`, `uses_vrt_source_rects=True`, `candidate_window_count_before_valid_filter=6248`, `black_filtered_window_count=1362`, `len_dataset=4886`, `estimated_positive_tiles=272`, `estimated_negative_tiles=4614`.
- Первый train batch: images `[4,4,1024,1024]`, range `0.0..242.2357`, masks `[4,1,1024,1024]`, range `0.0..1.0`, `positive_tile_count=4`, `augmented_tile_count=4`.
- Первый val batch: images `[4,4,1024,1024]`, range `18.0..255.0`, masks `[4,1,1024,1024]`, range `0.0..1.0`, `positive_tile_count=3`, `augmented_tile_count=0`.

## Первые 30 минут

- Статус MLflow: `RUNNING`.
- Эпох к 30-й минуте: 20 эпох, последние метрики на step `19`.
- GPU: около `14723 MiB`, загрузка около `68-71%`, температура около `59-60 C`.
- `train/skipped_optimizer_steps`: `0` на всех проверенных эпохах.
- Ошибок OOM, NaN и traceback за первые 30 минут не найдено.
- Запуск оставлен работать: да.

| step | train/loss | focal | tversky | val F1 @0.75 | best threshold | best-threshold F1 | precision @0.75 | recall @0.75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.9065 | 0.0370 | 0.8695 | 0.0030 | 0.50 | 0.0721 | 0.0263 | 0.0016 |
| 15 | 0.8678 | 0.0400 | 0.8278 | 0.0309 | 0.50 | 0.0896 | 0.0242 | 0.0426 |
| 19 | 0.8775 | 0.0374 | 0.8401 | 0.0441 | 0.50 | 0.1425 | 0.0790 | 0.0306 |

## Промежуточный вывод

DeepLabV3Plus ResNet50 стартовал устойчиво: память GPU достаточна, optimizer steps не пропускаются, loss конечный. Метрики в первые 30 минут растут, но пока ниже результата `smp_segformer_b2` continuation run. Окончательный вывод нужен после штатного завершения 10-часового запуска.
