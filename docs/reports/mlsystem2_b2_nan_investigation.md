# MLSystem2 B2 NaN investigation

## Failed Run

- run_id: `79b587fec89d4851a1afaf061370ca55`
- run_name: `deforestation_2305_2`
- status: `FAILED`
- train log: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_long_20260523T184504Z.log`
- config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_1024_long.yaml`
- failure: `EpochMetrics` rejected `train_loss=nan` and `val_loss=nan`
- checkpoints: not created
- epoch metrics: not logged

## Diagnostics

DataLoader finite scan:

- script: `/opt/mlsystem2/runtime/first_train/check_batches_finite.py`
- log: `/opt/mlsystem2/runtime/first_train/logs/check_batches_finite.log`
- train batches checked: `2000`
- val batches checked: `1562`
- non-finite images: no
- non-finite masks: no
- image range observed: `0..255`
- mask range observed: `0..1`

Forward/loss diagnostic:

- script: `/opt/mlsystem2/runtime/first_train/check_b2_forward_loss.py`
- log: `/opt/mlsystem2/runtime/first_train/logs/check_b2_forward_loss.log`
- first batch images/masks/logits/loss: finite
- first non-finite source: gradient after `loss.backward`
- parameter: `segformer.stages.0.patch_embeddings.proj.weight`

Batch cases:

- script: `/opt/mlsystem2/runtime/first_train/check_b2_batch_cases.py`
- log: `/opt/mlsystem2/runtime/first_train/logs/check_b2_batch_cases.log`
- early background-only batch with batch size 4 reproduced non-finite gradient
- batches with positive masks produced finite gradients
- single-sample checks were finite

Conclusion: NaN did not originate from raster values or masks. The failure came from non-finite gradients during B2 backward before optimizer step. The first bad gradient poisoned model weights, and epoch loss aggregation later became NaN.

## Fix

Commit: `a91add91ac4222782c6506cd26f69eb98efbfc44`

Changed `src/mlsystem2/train/_trainer.py`:

- added finite checks for `images`, `masks`, `logits`, `loss`;
- added finite checks for epoch loss metrics before `EpochMetrics`;
- added fixed gradient clipping `max_norm=1.0`;
- if a batch produces non-finite gradients, its optimizer step is skipped and weights are not updated from that batch;
- if an epoch has zero optimizer steps, training fails with `TrainError`.

No tile normalization was added. Geoalert tensor ABI remains raw `float32` `C,H,W`.

## Tests

Local:

- `tests/test_train_loop.py tests/test_docs_exist.py tests/test_public_contracts.py`: passed
- `ruff check src tests`: passed

Server after deploy:

- `python -m pytest tests/test_public_contracts.py -q`: passed
- `python -m pytest tests -q`: `73 passed`
- `python -m ruff check src ./tests`: passed

## Diagnostic Train

- config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_nan_diag.yaml`
- run_id: `0b32bc6882b64d0eb2431da541ba0ba1`
- run_name: `deforestation_2305_3`
- status: `FINISHED`
- epochs: `2`
- batch limits: train `16`, val `8`
- learning_rate: `1e-6`
- warnings: first two epoch-1 background batches skipped due to non-finite gradients
- `train/loss`: epoch 1 `0.9673160500824451`, epoch 2 `0.9879662059247494`
- `val/loss`: epoch 1 `0.9999971091747284`, epoch 2 `0.9999962896108627`
- `val/pixel_f1`: `0.0`
- checkpoints: best and final created

## Long Rerun

- restarted: yes
- config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_1024_long_rerun.yaml`
- launcher: `/opt/mlsystem2/runtime/first_train/run_b2_long_rerun.sh`
- run_id: `94fd30adf1624c71b8a3ab231730b359`
- run_name: `deforestation_2305_4`
- status after 5 minute monitor: `RUNNING`
- learning_rate: `1e-6`
- batch limits: none
- timeout: `36000s`
- GPU: `81..100%` utilization, about `22346 MiB / 32607 MiB`
- action: left running

## Recommendation

Keep the long B2 rerun under observation until the first epoch finishes and MLflow receives finite epoch metrics. If many batches keep producing non-finite gradients, the next minimal step is to lower batch size to `2` for B2 while keeping tile size `1024` and raw Geoalert-compatible tensors unchanged.
