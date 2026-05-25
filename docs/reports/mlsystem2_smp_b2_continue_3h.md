# MLSystem2 SMP B2 continuation 3h

Дата: 2026-05-25.

## Контур

- Server repo: `/opt/mlsystem2/repo`.
- Deployed commit: `9f4eb7c52ffdcc5e7d97573c06a9632b1b101e26`.
- MLMarkup: `/data/MLMarkup/Вырубки`.
- Prepared images: `/data/mlsystem2/prepared_images/`.
- MLflow experiment: `MLSystem2-First`.
- `/data/MLSystem2` не использовался.
- Symlink `/data/mlmarkup` не создавался.

## Reference Q3

- Run id: `a03eb71ad1f848d5891e2dbcd7837ab1`.
- Run name: `Q3-deforest-segformer-b1-t1024-aug-v1-allimages-triton-v11`.
- Model: `segformer_b1`.
- Tile size / stride: `1024 / 768`.
- Threshold: `0.65`.
- Accepted objects: `19070`.
- Local accepted GeoJSON: `D:\Projects\razmetka\Q3-deforest-segformer-b1-t1024-aug-v1-allimages-triton-v11.accepted.geojson`.

## 1h source run

- Run id: `d1d3700e500c45af8f3b6c2f6a33e465`.
- Run name: `deforestation_2505_5`.
- Model: `smp_segformer_b2`.
- Best checkpoint: `/opt/mlsystem2/runtime/first_train/smp_b2_posfactor_1h_scratch/checkpoints/best.pt`.
- Final checkpoint: `/opt/mlsystem2/runtime/first_train/smp_b2_posfactor_1h_scratch/checkpoints/final.pt`.
- Best epoch: `28`.
- Best `val/pixel_f1` at configured threshold `0.75`: `0.26272019266979324`.
- Best threshold F1 in history: `0.2797436834539595`.

## 3h continuation run

- Config: `/opt/mlsystem2/runtime/first_train/deforestation_smp_b2_continue_3h.yaml`.
- Run id: `fb6597864d5f4a86bab818bdf5c7534d`.
- Run name: `deforestation_2505_6`.
- Initial checkpoint: `/opt/mlsystem2/runtime/first_train/smp_b2_posfactor_1h_scratch/checkpoints/final.pt`.
- Model: `smp_segformer_b2`.
- LR: `0.000001`.
- Loss: `focal_tversky`.
- Tile size / stride: `1024 / 512`.
- `smart_tiling`: `true`.
- `positive_factor`: `0.8`.
- `val_positive_factor`: `0.5`.
- Max train batches per epoch: `128`.
- Max val batches per epoch: `1000`.
- Max training time: `10800 sec`.
- Status: `FINISHED`.
- Epochs completed: `82`.
- Skipped optimizer steps: `0`.

Best checkpoint:

- Path: `/opt/mlsystem2/runtime/first_train/smp_b2_continue_3h_scratch/checkpoints/best.pt`.
- Best checkpoint epoch: `38`.
- `val/pixel_f1`: `0.33239684598784325`.
- `val/loss`: `0.9244394881725311`.
- `train/loss`: `0.8698810841888189`.
- Precision at threshold `0.75`: `0.29314013538757333`.
- Recall at threshold `0.75`: `0.38379369002490155`.

Final checkpoint:

`/opt/mlsystem2/runtime/first_train/smp_b2_continue_3h_scratch/checkpoints/final.pt`

## Динамика

Сравнение с 1h run:

- 1h best configured-threshold F1: `0.2627`.
- 1h best-threshold F1: `0.2797`.
- 3h continuation best configured-threshold F1: `0.3324`.
- 3h continuation best-threshold F1: `0.3324`.

Метрики выросли после continuation. Лучшие эпохи 3h run чаще выбирали threshold `0.70` или `0.75`; высокие thresholds `0.90+` не стали основным рабочим режимом для текущей модели.

## Geoalert inference

Best checkpoint экспортировался в Triton model `mlsystem2_deforestation` через ONNX wrapper:

`sigmoid(logits) > threshold -> uint8 mask`

Проверены два threshold-варианта.

### Threshold 0.70

- Server run root: `/opt/geoalert/runs/deforestation_smp_b2_continue_3h_thr070`.
- Scenes: `35`.
- Processed: `35`.
- Failed: `0`.
- Missing inputs: `0`.
- Total mask sum: `152945716`.
- Features: `37097`.
- Elapsed: `272.769 sec`.
- Server merged GeoJSON: `/opt/geoalert/runs/deforestation_smp_b2_continue_3h_thr070/pseudolabels_merged.geojson`.
- Local merged GeoJSON: `D:\Projects\razmetka\deforestation_pseudolabels_smp_b2_continue_3h_thr070.geojson`.
- Local summary: `D:\Projects\razmetka\deforestation_pseudolabels_smp_b2_continue_3h_thr070_summary.json`.

### Threshold 0.75

- Server run root: `/opt/geoalert/runs/deforestation_smp_b2_continue_3h_thr075`.
- Scenes: `35`.
- Processed: `35`.
- Failed: `0`.
- Missing inputs: `0`.
- Total mask sum: `109221676`.
- Features: `25118`.
- Elapsed: `298.604 sec`.
- Server merged GeoJSON: `/opt/geoalert/runs/deforestation_smp_b2_continue_3h_thr075/pseudolabels_merged.geojson`.
- Local merged GeoJSON: `D:\Projects\razmetka\deforestation_pseudolabels_smp_b2_continue_3h_thr075.geojson`.
- Local summary: `D:\Projects\razmetka\deforestation_pseudolabels_smp_b2_continue_3h_thr075_summary.json`.

## Вывод

Continuation с `smp_segformer_b2` улучшил диагностический pixel F1 относительно 1h run и дал стабильное обучение без non-finite optimizer skips. По количеству объектов:

- Q3 reference: `19070`;
- MLSystem2 3h threshold `0.70`: `37097`;
- MLSystem2 3h threshold `0.75`: `25118`.

Threshold `0.75` ближе к Q3 по объему псевдоразметки, но все еще дает больше объектов. Для визуальной приемки разумнее сначала смотреть `thr075`, а `thr070` оставить как recall-oriented вариант.
