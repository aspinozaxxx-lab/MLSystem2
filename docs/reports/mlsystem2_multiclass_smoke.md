# Multiclass segmentation smoke и B2 train

Дата: 2026-05-26.

Сервер: `ssh gpu-mlserver`.
Runtime config: `/opt/mlsystem2/runtime/multiclass_b2_3h/multiclass_b2_3h.yaml`.
MLflow experiment: `MLSystem2-Multiclass`.

## Найденные классы и class id

| class_id | slug | Русское имя | scenes_file | annotation_file | priority |
|---:|---|---|---|---|---:|
| 1 | abrasion | Абразия | `/data/MLMarkup/Абразия/abrasion.txt` | `/data/MLMarkup/Абразия/abrasion.geojson` | 0 |
| 2 | wind_erosion | Ветровая эрозия | `/data/MLMarkup/Ветровая эрозия/wind_erosion.txt` | `/data/MLMarkup/Ветровая эрозия/wind_erosion.geojson` | 0 |
| 3 | water_erosion | Водная эрозия | `/data/MLMarkup/Водная эрозия/water_erosion.txt` | `/data/MLMarkup/Водная эрозия/water_erosion.geojson` | 0 |
| 4 | deforestation | Вырубки | `/data/MLMarkup/Вырубки/deforestation.txt` | `/data/MLMarkup/Вырубки/deforestation.geojson` | 0 |
| 5 | fire | Гари | `/data/MLMarkup/Гари/burnt_forests.txt` | `/data/MLMarkup/Гари/burnt_forests.geojson` | 0 |
| 6 | waterlogging | Заболачивание | `/data/MLMarkup/Заболачивание/swampings.txt` | `/data/MLMarkup/Заболачивание/swampings.geojson` | 0 |
| 7 | salinization | Засоления | `/data/MLMarkup/Засоления/salty.txt` | `/data/MLMarkup/Засоления/salty.geojson` | 0 |
| 8 | quarries | Карьеры | `/data/MLMarkup/Карьеры/careers.txt` | `/data/MLMarkup/Карьеры/careers.geojson` | 0 |
| 9 | landslide_scree | Обвально-оползневые и осыпные | `/data/MLMarkup/Обвально-оползневые и осыпные/landslides.txt` | `/data/MLMarkup/Обвально-оползневые и осыпные/landslides.geojson` | 0 |
| 10 | lakes | Озера | `/data/MLMarkup/Озера/lakes.txt` | `/data/MLMarkup/Озера/lakes.geojson` | -100 |
| 11 | desertification | Опустынивание | `/data/MLMarkup/Опустынивание/desertification.txt` | `/data/MLMarkup/Опустынивание/desertification.geojson` | 0 |
| 12 | arable_lands | Пашни | `/data/MLMarkup/Пашни/areas_of_used_arable_land.txt` | `/data/MLMarkup/Пашни/areas_of_used_arable_land.geojson` | 0 |
| 13 | rivers | Реки | `/data/MLMarkup/Реки/rivers.txt` | `/data/MLMarkup/Реки/rivers.geojson` | 0 |

## Dataset preparation

Для рабочего runtime были созданы временные scene lists в `/opt/mlsystem2/runtime/multiclass_b2_3h/scenes/`: из исходных списков оставлены только сцены, которые однозначно сопоставляются с `/data/mlsystem2/prepared_images/`. MLMarkup не изменялся.

Итог подготовки:
- status: `ok`;
- scenes_total: `183`;
- scenes_found: `183`;
- objects_total: `1676`;
- train_scenes_count: `156`;
- val_scenes_count: `27`.

Классы с пустым исходным `scenes_file` не исключаются из масок: после сборки единого пула сцен их GeoJSON проверяется на всех снимках пула. В отчете long run это видно по support pixels для `waterlogging` и `rivers`.

## Tile preparation

Batch ABI подтвержден:
- image: `torch.float32 [B,4,1024,1024]`;
- mask: `torch.long [B,1024,1024]`;
- допустимые mask values: `0..13`; train loop проверял диапазон на каждом batch.

Tile report long run:
- train tiles: `105311`, batches: `52656`;
- val tiles: `17862`, batches: `8931`;
- train estimated positive/negative: `9924 / 95387`;
- val estimated positive/negative: `2520 / 15342`;
- observed train batches: `6000`;
- observed val batches: `1500`;
- observed class positive tile counts train: `lakes=1525`, `abrasion=23`, `wind_erosion=20`, `water_erosion=369`, `deforestation=2145`, `fire=193`, `waterlogging=120`, `salinization=311`, `quarries=245`, `landslide_scree=0`, `desertification=3209`, `arable_lands=1041`, `rivers=747`.

Пересечения классов разрешаются приоритетами: больший `priority` перекрывает меньший, при равном priority выигрывает больший `class_id`. Для `lakes` задан низший priority `-100`.

## Smoke и train runs

Короткий smoke:
- run_id: `a054bbf137a04bc7b661ae104ff1d873`;
- run_name: `multiclass_2605_4`;
- status: `succeeded`.

B2 train:
- model_name: `smp_segformer_b2`;
- run_id: `14eab4b2bdbf4734935a2ab1e1debe24`;
- run_name: `multiclass_b2_3h_2605_long`;
- status: `FINISHED`;
- total_pipeline_time_sec: `1439.42`;
- train_time_sec: `861.07`;
- epochs_total: `3`;
- остановка: early stopping после ухудшения `val_macro_f1` при `early_stopping_patience=2`;
- best checkpoint: `/opt/mlsystem2/runtime/multiclass_b2_3h/scratch_long/checkpoints/best.pt`;
- final checkpoint: `/opt/mlsystem2/runtime/multiclass_b2_3h/scratch_long/checkpoints/final.pt`.

Метрики по эпохам:

| epoch | train_loss | val_loss | macro_f1 | mean_iou | pixel_accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4239 | 0.9759 | 0.03987 | 0.02438 | 0.85448 |
| 2 | 0.7067 | 0.8051 | 0.02218 | 0.01214 | 0.85361 |
| 3 | 0.4971 | 0.6182 | 0.01691 | 0.00920 | 0.91234 |

Final per-class F1:
- abrasion: `0.0`;
- wind_erosion: `0.0`;
- water_erosion: `0.000009`;
- deforestation: `0.000488`;
- fire: `0.0`;
- waterlogging: `0.0`;
- salinization: `0.0`;
- quarries: `0.0`;
- landslide_scree: `0.003822`;
- lakes: `0.006196`;
- desertification: `0.000084`;
- arable_lands: `0.002132`;
- rivers: `0.173265`.

## Ошибки и исправления

- SSH по `root@31.192.104.147` не проходил; использован алиас `ssh gpu-mlserver`.
- MLflow artifacts пишутся в S3/MinIO; для CLI train на хосте нужны `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000`.
- В исходных списках были missing/ambiguous сцены. Для runtime train созданы временные очищенные scene lists, MLMarkup не менялся.
- Реальные GeoJSON имеют пересечения классов. Вместо падения DataLoader добавлен deterministic priority merge; `lakes` получил низший priority.
- Классы с пустыми scene lists теперь не теряются: их GeoJSON применяется на всем едином пуле сцен.

## Что осталось

- Уточнить или поправить исходные scene lists для `waterlogging`, `rivers`, `quarries` и ambiguous сцен, чтобы runtime-фильтрация не была нужна.
- Определить доменные приоритеты всех классов, а не только `lakes`.
- Для полноценного обучения настроить class balancing или sampler: текущий positive sampler foreground/background, без per-class баланса.
- Подобрать learning rate/epochs/patience и validation strategy; текущий запуск был инженерным smoke/train, метрики качества ожидаемо низкие.
