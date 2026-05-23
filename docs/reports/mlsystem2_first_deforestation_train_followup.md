# MLSystem2 first deforestation train follow-up

## Paths

- repo: `/opt/mlsystem2/repo`
- deployed commit: `f0e05164c04e8d607ca01e64dc45a83630978b69`
- MLMarkup: `/data/MLMarkup`
- `/data/mlmarkup` symlink: удален
- smoke config: `/opt/mlsystem2/runtime/first_train/deforestation_b0_smoke.yaml`
- logs: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b0_smoke_20260523T181234Z.log`
- scratch: `/opt/mlsystem2/runtime/first_train/b0_smoke_scratch`

## CUDA

- status: работает
- torch: `2.12.0+cu130`
- cuda available: `true`
- GPU: `NVIDIA GeForce RTX 5090`

## Tile Preparation

- before: первый запуск B2 тратил около 10 минут на upfront чтение окон до старта train.
- fix: upfront фильтрация fully-nodata windows удалена; `TileDataset` строит только список окон, а image/mask читает лениво в `__getitem__`.
- dataset_preparing_sec: `0.145`
- create_tile_dataloader_sec: `0.506`
- first_batch_sec: `0.667`
- first batch image shape: `[4, 4, 1024, 1024]`
- first batch image value range: `0.0..255.0`
- first batch mask shape: `[4, 1, 1024, 1024]`

## B0 Smoke

- model_name: `segformer_b0`
- epochs: `2`
- batch_size: `4`
- max_train_batches_per_epoch: `8`
- max_val_batches_per_epoch: `4`
- MLflow experiment: `MLSystem2-First`
- MLflow tracking_uri: `http://127.0.0.1:5000`
- run_id: `e4df0282fe764a44b00d27bbf2b04de0`
- run_name: `deforestation_2305_1`
- status: `FINISHED`
- CLI exit_code in log: `0`
- training_time_sec: `7.344`

## Live MLflow Metrics

Epoch metrics are present in MLflow metric history with epoch steps:

- `train/loss`: epoch 1 `0.9530088901519775`, epoch 2 `0.9830463901162148`
- `val/loss`: epoch 1 `0.999997615814209`, epoch 2 `0.9999976456165314`
- `val/pixel_f1`: epoch 1 `0.0`, epoch 2 `0.0`
- `train/epoch_time_sec`: epoch 1 `4.203153606000342`, epoch 2 `2.5607721639999`

Summary metrics:

- `train/epochs_total`: `2`
- `val/best_pixel_f1`: `0.0`
- `val/final_pixel_f1`: `0.0`

## Artifacts

- best checkpoint: `/opt/mlsystem2/runtime/first_train/b0_smoke_scratch/checkpoints/best.pt`
- final checkpoint: `/opt/mlsystem2/runtime/first_train/b0_smoke_scratch/checkpoints/final.pt`
- MLflow artifacts present: `reports/dataset_preparation.json`, `reports/training_history_full.json`, `reports/pipeline_timings.json`, `reports/pipeline_summary.json`, `checkpoints/best.pt`, `checkpoints/final.pt`

## Checks

- server `python -m pytest tests/test_public_contracts.py -q`: passed
- server `python -m pytest tests -q`: `72 passed`
- server `python -m ruff check src ./tests`: passed
- local affected tests: `33 passed, 1 skipped`
- local full tests: blocked by missing Windows `gdal_translate`, unrelated to this change

## Fixes

- `tile_preparation` no longer reads every window during DataLoader creation.
- Added live per-epoch MLflow logging via `log_training_epoch`.
- `train_pipeline` sends `epoch_finished` metrics to MLflow through progress sink.
- Added auto MLflow run naming by class and local date: `{class}_{DDMM}_{number}`.
- Added `segformer_b0` alongside existing `segformer_b2`.
- Added diagnostic train batch limits for short smoke runs.
- Updated architecture docs and public contract tests.

## Next Step

Run a longer B0 or B2 controlled training run now that loader startup, live MLflow metrics, run naming, CUDA and artifacts are verified.
