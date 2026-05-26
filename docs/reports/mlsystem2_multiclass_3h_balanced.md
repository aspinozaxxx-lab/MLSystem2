# MLSystem2 multiclass balanced 3h train

Дата: 2026-05-26.

## Почему прежний run остановился

Предыдущий run `14eab4b2bdbf4734935a2ab1e1debe24` остановился после 3 эпох из-за `early_stopping_patience=2`: `macro_f1` ухудшался после первой эпохи. Это был штатный early stopping, а не падение. Для нового запуска выставлены `epochs=300`, `early_stopping_patience=300`, `max_training_time_sec=10800`.

## Запуск

- Config: `/opt/mlsystem2/runtime/multiclass_b2_3h/multiclass_b2_3h_balanced.yaml`
- MLflow experiment: `MLSystem2-Multiclass`
- MLflow run: `69d502ed8d304c1ca03d991e79bc8a4d`
- Run name: `multiclass_2605_5`
- Model: `smp_segformer_b2`
- Loss: `cross_entropy_dice`
- Class balance: `true`
- Train time: `10807.83 sec`
- Total pipeline time: `11396.79 sec`
- Status: `FINISHED`

Best checkpoint:

```text
/opt/mlsystem2/runtime/multiclass_b2_3h/scratch_balanced/checkpoints/best.pt
```

Final checkpoint:

```text
/opt/mlsystem2/runtime/multiclass_b2_3h/scratch_balanced/checkpoints/final.pt
```

## Dataset

- Scenes total: 183
- Train scenes: 156
- Val scenes: 27
- Objects total: 1687
- Train objects: 1320
- Val objects: 367

## Tile report

Train split:

- Tile count: 105311
- Estimated positive tiles: 10045
- Estimated negative tiles: 95266
- Sampling mode: `weighted_class_balance`
- Target positive factor: 0.8
- Observed positive ratio: 0.800423
- Ratio abs error: 0.000423
- Warning: `landslide_scree` had no positive train windows.

Estimated class positive tiles:

| slug | estimated |
|---|---:|
| lakes | 1625 |
| abrasion | 30 |
| wind_erosion | 23 |
| water_erosion | 388 |
| deforestation | 2177 |
| fire | 176 |
| waterlogging | 140 |
| salinization | 348 |
| quarries | 269 |
| landslide_scree | 0 |
| desertification | 3311 |
| arable_lands | 1194 |
| rivers | 780 |

Observed train class-positive tile counts:

| slug | observed |
|---|---:|
| lakes | 6942 |
| abrasion | 4153 |
| wind_erosion | 4086 |
| water_erosion | 4254 |
| deforestation | 5933 |
| fire | 4242 |
| waterlogging | 4394 |
| salinization | 4404 |
| quarries | 4279 |
| landslide_scree | 0 |
| desertification | 4157 |
| arable_lands | 4315 |
| rivers | 5967 |

Val split used diagnostic weighted sampling:

- Target positive factor: 0.5
- Observed positive ratio: 0.494492
- Ratio abs error: 0.005508
- Warnings: `wind_erosion` and `salinization` had no positive val windows.

## Metrics

Best epoch: 119

- Best `val_macro_f1`: 0.081296
- Best `val_mean_iou`: 0.050902
- Best `val_pixel_accuracy`: 0.896797
- Best `val_loss`: 1.410364
- Best `train_loss`: 1.320127
- Final epoch: 120
- Final `val_macro_f1`: 0.069310
- Final `val_mean_iou`: 0.042523

Per-class validation metrics at best epoch:

| slug | f1 | iou | precision | recall | support_pixels |
|---|---:|---:|---:|---:|---:|
| abrasion | 0.059295 | 0.030553 | 0.057655 | 0.061031 | 463830 |
| wind_erosion | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| water_erosion | 0.000026 | 0.000013 | 0.000014 | 0.000265 | 3951602 |
| deforestation | 0.038037 | 0.019387 | 0.105981 | 0.023178 | 8304661 |
| fire | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4650297 |
| waterlogging | 0.033235 | 0.016898 | 0.035247 | 0.031441 | 5460201 |
| salinization | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| quarries | 0.001864 | 0.000933 | 0.000970 | 0.024067 | 270914 |
| landslide_scree | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 945403 |
| lakes | 0.227853 | 0.128574 | 0.154346 | 0.435038 | 4670753 |
| desertification | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3013964 |
| arable_lands | 0.001487 | 0.000744 | 0.079494 | 0.000750 | 20024416 |
| rivers | 0.532458 | 0.362823 | 0.430536 | 0.697604 | 19174450 |

## Geoalert inference after train

Inference was run with best checkpoint on the 156 train-split scenes.

- Triton model: `mlsystem2_multiclass_balanced_3h`
- Triton export path: `/opt/geoalert/triton_models/mlsystem2_multiclass_balanced_3h`
- Geoalert pipeline: `/opt/geoalert/pipelines/mlsystem2_multiclass_balanced_3h_triton.yaml`
- Server result root: `/opt/geoalert/runs/multiclass_balanced_3h_train`
- Local result path: `D:\Projects\razmetka\multiclass\2`
- Scenes processed: 156
- Failed: 0
- Missing images: 0
- Total features: 215681

Inference feature counts:

| slug | features |
|---|---:|
| abrasion | 4763 |
| wind_erosion | 53741 |
| water_erosion | 54808 |
| deforestation | 8786 |
| fire | 636 |
| waterlogging | 9126 |
| salinization | 33091 |
| quarries | 12853 |
| landslide_scree | 0 |
| lakes | 9463 |
| desertification | 154 |
| arable_lands | 7719 |
| rivers | 20541 |

Local structure:

```text
D:\Projects\razmetka\multiclass\2\
  multiclass_balanced_3h_train_summary.json
  merged_all_classes.geojson
  classes\
    abrasion.geojson
    ...
    rivers.geojson
  per_scene\
    <train_scene_id>\
      abrasion.geojson
      ...
      rivers.geojson
```

## Осталось для полноценного обучения

- `landslide_scree` не получил positive train windows в текущем split; нужны сцены, где его объекты реально пересекаются с prepared images.
- Для честной финальной оценки нужно прогнать sequential/full val без `val_positive_factor`.
- Классы с нулевой или почти нулевой validation поддержкой требуют отдельной проверки split и разметки.
