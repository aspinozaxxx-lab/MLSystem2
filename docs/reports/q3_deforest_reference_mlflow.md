# Q3 deforest reference MLflow

Дата: 2026-05-25.

## Цель

Зафиксировать reference для сравнения с MLSystem2 continuation training по вырубкам.

## Найденный run

- MLflow tracking URI: `http://127.0.0.1:5000`.
- Experiment id: `2`.
- Run id: `a03eb71ad1f848d5891e2dbcd7837ab1`.
- Run name: `Q3-deforest-segformer-b1-t1024-aug-v1-allimages-triton-v11`.
- Status: `FINISHED`.

Параметры из MLflow:

- `model.name`: `segformer_b1`.
- `preprocess.tile_size`: `1024`.
- `preprocess.stride`: `768`.
- `threshold_used`: `0.65`.
- `max_raw_features_for_candidate`: `200000`.

## Accepted pseudo-label

Server artifact storage:

`/data/mlsystem/minio/mlflow-artifacts/2/a03eb71ad1f848d5891e2dbcd7837ab1/artifacts/Q3-deforest-segformer-b1-t1024-aug-v1-allimages-triton-v11.accepted.geojson`

Локальная копия:

`D:\Projects\razmetka\Q3-deforest-segformer-b1-t1024-aug-v1-allimages-triton-v11.accepted.geojson`

Сводка из локального `pseudolabel_summary`:

- Accepted objects: `19070`.
- Accepted GeoJSON size: `27.64 MB`.
- Total area: `1339603620.234493 m2`.
- Total vertices: `608670`.
- Min object area: `10000.0 m2`.
- Simplify tolerance: `30.0 m`.
- Vectorization workers: `8`.

## Вывод

Q3 reference найден в старом artifact storage, а не среди тегов/параметров текущего experiment `MLSystem2-First`. Для текущего сравнения используем:

- object count: `19070`;
- threshold: `0.65`;
- model family: `segformer_b1`;
- tile/stride: `1024/768`.
