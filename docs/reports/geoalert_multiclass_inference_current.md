# Geoalert inference текущего multiclass checkpoint

Дата: 2026-05-26.

## Контур

- Checkpoint: `/opt/mlsystem2/runtime/multiclass_b2_3h/scratch_long/checkpoints/best.pt`
- MLflow run: `14eab4b2bdbf4734935a2ab1e1debe24`
- Triton model: `mlsystem2_multiclass`
- Triton export path: `/opt/geoalert/triton_models/mlsystem2_multiclass`
- Geoalert pipeline: `/opt/geoalert/pipelines/mlsystem2_multiclass_triton.yaml`
- Server result root: `/opt/geoalert/runs/multiclass_current_checkpoint`
- Local result path: `D:\Projects\razmetka\multiclass`

## Triton

Модель экспортирована как ONNX wrapper:

- input: `input`, `float32 [1,4,H,W]`;
- output: `masks`, `uint8 [1,13,H,W]`;
- wrapper делает `argmax(logits, dim=1)` и возвращает one-hot foreground masks без background;
- channel `0` соответствует `class_id=1 abrasion`, channel `12` соответствует `class_id=13 rivers`.

Triton container `geoalert-triton` был перезапущен, модель загружена в статусе `READY`. Тестовый HTTP inference вернул `uint8 (1,13,64,96)` со значениями `0/1`.

## Результат

- Scenes total: 183
- Scenes processed: 183
- Failed: 0
- Missing images: 0
- Total features: 4 566 086
- Archive: `/opt/geoalert/runs/multiclass_current_checkpoint/multiclass_current_result.zip`
- Archive size: 2.2 GiB

Feature count по классам:

| class_id | slug | имя | features |
|---:|---|---|---:|
| 1 | abrasion | Абразия | 38 938 |
| 2 | wind_erosion | Ветровая эрозия | 1 010 157 |
| 3 | water_erosion | Водная эрозия | 12 293 |
| 4 | deforestation | Вырубки | 769 310 |
| 5 | fire | Гари | 372 805 |
| 6 | waterlogging | Заболачивание | 203 362 |
| 7 | salinization | Засоления | 587 030 |
| 8 | quarries | Карьеры | 5 956 |
| 9 | landslide_scree | Обвально-оползневые и осыпные | 295 047 |
| 10 | lakes | Озера | 114 847 |
| 11 | desertification | Опустынивание | 834 036 |
| 12 | arable_lands | Пашни | 276 215 |
| 13 | rivers | Реки | 46 090 |

## Локальная структура

```text
D:\Projects\razmetka\multiclass\
  multiclass_current_summary.json
  merged_all_classes.geojson
  classes\
    abrasion.geojson
    wind_erosion.geojson
    ...
    rivers.geojson
  per_scene\
    <scene_id>\
      abrasion.geojson
      ...
      rivers.geojson
```

## Наблюдение

Текущий checkpoint дает очень много мелких полигонов почти по всем каналам. Это согласуется с низким `macro_f1` предыдущего run и является качеством модели, а не ошибкой Triton/Geoalert pipeline.
