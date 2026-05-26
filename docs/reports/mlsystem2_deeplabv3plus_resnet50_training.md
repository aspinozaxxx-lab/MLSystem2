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

## Итог обучения

- Статус MLflow: `FINISHED`.
- Завершение: по лимиту `epochs=300`, раньше `max_training_time_sec=36000`.
- Эпох завершено: `300`.
- `train/skipped_optimizer_steps`: `0` на всех эпохах.
- Best checkpoint: `/opt/mlsystem2/runtime/first_train/deeplabv3plus_resnet50_10h_scratch/checkpoints/best.pt`.
- Final checkpoint: `/opt/mlsystem2/runtime/first_train/deeplabv3plus_resnet50_10h_scratch/checkpoints/final.pt`.
- Best checkpoint epoch: `146`.
- Best checkpoint `val/pixel_f1` на threshold `0.75`: `0.37842743615953794`.
- Best checkpoint precision/recall на threshold `0.75`: `0.5169249755682588 / 0.2984618146631699`.
- Best checkpoint sweep-best F1: `0.4109503130224459` при threshold `0.5`.
- Лучший sweep-best F1 за весь run: `0.4294970138346798` на step `131`, threshold `0.3`.
- Лучший configured-threshold F1 за весь run: `0.37842743615953794` на step `146`.
- Последний step `300`: `val/pixel_f1=0.2847776173090043`, `val/best_threshold_pixel_f1=0.30895870852763135`, `train/loss=0.6576315227430314`.

| run | model | epochs | best F1 @configured threshold | best sweep F1 | notes |
| --- | --- | ---: | ---: | ---: | --- |
| `deforestation_2505_5` | `smp_segformer_b2` | 28 best epoch | 0.2627 | 0.2797 | первый 1h SMP B2 |
| `deforestation_2505_6` | `smp_segformer_b2` | 82 | 0.3324 | 0.3324 | 3h continuation |
| `deforestation_2505_7` | `smp_deeplabv3plus_resnet50` | 300 | 0.3784 | 0.4295 | текущий запуск |

DeepLabV3Plus ResNet50 оказался лучше предыдущего SMP B2 diagnostic path по pixel F1, но после пика метрики заметно колебались и к финальной эпохе снизились. Для экспорта выбран `best.pt`, а не `final.pt`.

## ONNX/Triton export

- Triton model name: `mlsystem2_deforestation_deeplabv3plus_resnet50`.
- Существующая модель `mlsystem2_deforestation` не затиралась.
- Threshold export: `0.5`, потому на best checkpoint sweep-best threshold был `0.5`.
- ONNX: `/opt/geoalert/triton_models/mlsystem2_deforestation_deeplabv3plus_resnet50/1/model.onnx`.
- ONNX external data: `/opt/geoalert/triton_models/mlsystem2_deforestation_deeplabv3plus_resnet50/1/model.onnx.data`.
- ONNX opset: `18`.
- Export metadata: `/opt/geoalert/triton_models/mlsystem2_deforestation_deeplabv3plus_resnet50/export_metadata.json`.
- Triton status после restart: `mlsystem2_deforestation=READY`, `mlsystem2_deforestation_deeplabv3plus_resnet50=READY`.

Для нового ONNX output shape был `[1,1,-1,-1]`, поэтому `config.pbtxt` для DeepLab использует `dims: [ 1, 1, -1, -1 ]`. Это отличается от старого экспорта `mlsystem2_deforestation`, где output batch dimension динамический.

## Geoalert inference

- Pipeline: `/opt/geoalert/pipelines/mlsystem2_deforestation_deeplabv3plus_resnet50_triton.yaml`.
- Server run root: `/opt/geoalert/runs/deforestation_deeplabv3plus_resnet50_10h`.
- Scenes: `35`.
- Processed: `35`.
- Failed: `0`.
- Missing inputs: `0`.
- Threshold: `0.5`.
- Total mask sum: `68123297`.
- Total features: `11198`.
- Elapsed: `255.152 sec`.
- Server merged GeoJSON: `/opt/geoalert/runs/deforestation_deeplabv3plus_resnet50_10h/pseudolabels_merged.geojson`.
- Local merged GeoJSON: `D:\Projects\razmetka\deforestation_pseudolabels_deeplabv3plus_resnet50_10h.geojson`.
- Local summary: `D:\Projects\razmetka\deforestation_pseudolabels_deeplabv3plus_resnet50_10h_summary.json`.
- Local export metadata: `D:\Projects\razmetka\deforestation_pseudolabels_deeplabv3plus_resnet50_10h_export_metadata.json`.

Сравнение объема псевдоразметки:

- Q3 accepted reference: `19070` features.
- SMP B2 3h threshold `0.70`: `37097` features.
- SMP B2 3h threshold `0.75`: `25118` features.
- DeepLabV3Plus ResNet50 10h threshold `0.5`: `11198` features.

## Вывод

`smp_deeplabv3plus_resnet50` обучается устойчиво и дал лучшую diagnostic validation-метрику, чем предыдущий `smp_segformer_b2`: `0.3784` F1 на рабочем threshold `0.75` и `0.4295` лучший sweep F1. При этом Geoalert pseudo-label export на threshold `0.5` получился существенно компактнее Q3 reference и SMP B2 вариантов. Для визуальной приемки стоит смотреть DeepLab pseudo-label рядом с Q3 accepted и SMP B2 `thr075`: DeepLab, вероятно, более precision-oriented, но может недобирать recall.
