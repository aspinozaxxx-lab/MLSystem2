# MLSystem2 B2 deforestation long run

## Context

- repo: `/opt/mlsystem2/repo`
- commit: `dc21740b26acac4e6abf518b7eb9be3474fff22e`
- MLMarkup: `/data/MLMarkup`
- images_dir: `/data/mlsystem2/prepared_images/kanopus`
- config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_1024_long.yaml`
- launcher: `/opt/mlsystem2/runtime/first_train/run_b2_long.sh`
- timeout: `36000s`
- model: `segformer_b2`
- MLflow experiment: `MLSystem2-First`

## Hyperparameters

- tile_size: `1024`
- stride: `512`
- batch_size: `4`
- lr: `2e-6`
- weight_decay: `1e-4`
- loss: `focal_tversky(alpha=0.40,beta=0.60)`
- threshold: `0.75`
- epochs: `10`
- patience: `5`
- batch limits: none
- initial_checkpoint_uri: `null`

## Preflight

- `/data/mlmarkup` symlink: absent
- `/data/MLMarkup/Вырубки/deforestation.txt`: exists
- `/data/MLMarkup/Вырубки/deforestation.geojson`: exists
- CUDA: available
- GPU: `NVIDIA GeForce RTX 5090`
- server tests: `72 passed`
- server ruff: passed
- dataset_preparing_sec: `0.147`
- create_tile_dataloader_sec: `0.510`
- first_batch_sec: `0.728`
- first batch image shape: `[4, 4, 1024, 1024]`
- first batch image value range: `0.0..255.0`

## 5 Minute Monitor

- process alive: yes
- launcher PID: `335168`
- timeout PID: `335189`
- main train PID: `335193`
- GPU status: `88..100%` utilization, `22350 MiB / 32607 MiB`
- latest MLflow run_id: `79b587fec89d4851a1afaf061370ca55`
- latest MLflow run_name: `deforestation_2305_2`
- latest MLflow status: `RUNNING`
- epoch metrics after 5 minutes: not yet available, expected for full B2 epoch
- errors seen: no
- action: left running

## Logs

- train log: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_long_20260523T184504Z.log`
- gpu log: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_long_20260523T184504Z_gpu.log`
- nohup log: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_long_nohup.out`
- launcher pid file: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_long_launcher.pid`
- wrapper pid file: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_long_20260523T184504Z.pid`
