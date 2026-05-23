# Первый запуск обучения MLSystem2 по вырубкам

## Пути
- репозиторий: `/opt/mlsystem2/repo`
- тип развертывания: CI/CD export без `.git`
- deployed commit: `c9e5fee531d2441ad8d2b863a1506dbf7e580768`
- ручной clone `/data/MLSystem2`: удален
- MLMarkup: `/data/mlmarkup`
- фактический источник разметки: `/data/mlmarkup -> /data/MLMarkup`
- конфиг: `/opt/mlsystem2/runtime/first_train/deforestation_first_1024.yaml`
- логи: `/opt/mlsystem2/runtime/first_train/logs`
- scratch: `/opt/mlsystem2/runtime/first_train/scratch`

## CUDA
- before mismatch: loaded NVIDIA kernel module `580.142`, installed DKMS/userland `580.159.03`; `nvidia-smi` падал с `Driver/library version mismatch`.
- диагностика до исправления: `/opt/mlsystem2/reports/nvidia_cuda_diagnostic_before.txt`
- исправление: controlled reboot сервера.
- after fix: kernel `6.8.0-117-generic`, NVIDIA kernel module/userland `580.159.03`.
- диагностика после исправления: `/opt/mlsystem2/reports/nvidia_cuda_diagnostic_after.txt`
- `nvidia-smi` после исправления: `NVIDIA GeForce RTX 5090`, 32607 MiB, driver `580.159.03`, CUDA `13.0`.
- torch после исправления: `torch 2.12.0+cu130`, `torch.cuda.is_available() == True`, device `NVIDIA GeForce RTX 5090`.

## Датасет
- images_dir: `/data/mlsystem2/prepared_images/kanopus`
- подготовленные снимки: 995 GeoTIFF
- scenes_file: `/data/mlmarkup/Вырубки/deforestation.txt`
- annotation_file: `/data/mlmarkup/Вырубки/deforestation.geojson`
- scenes found: 35 из 35
- objects: 532
- train split: 29 сцен, 419 объектов
- val split: 6 сцен, 113 объектов
- first train loader size: 6072 batch при `batch_size=4`
- first batch image: `(4, 4, 1024, 1024)`, `torch.float32`, min `12.0`, max `255.0`
- first batch mask: `(4, 1, 1024, 1024)`, `torch.float32`, min `0.0`, max `0.0`

## Гиперпараметры
- source: 1024 no-noise threshold075 branch старого MLSystem и параметры запуска для вырубок
- model: `segformer_b2`
- tile_size: 1024
- stride: 512
- batch_size: 4
- learning_rate: `0.000002`
- weight_decay: `0.0001`
- loss: `focal_tversky`
- threshold: `0.75`
- epochs: 10
- early_stopping_patience: 5
- initial_checkpoint_uri: `null`
- checkpoint compatibility: старые checkpoints читаются как dict с `model_state_dict`, но state dict использует layout `encoder.patch_embed*/decoder.*`, несовместимый с текущей Hugging Face `SegformerForSemanticSegmentation`. Warm-start отключен.

## MLflow
- tracking_uri: `http://127.0.0.1:5000`
- experiment_name: `MLSystem2-First`
- experiment_id: `43`
- run_name: `mlsystem2_first_deforestation_1024_20260523T161829Z`
- run_id: `f6d425b888724ff180e8de24cf8d75f4`
- status: `KILLED`
- причина статуса: штатный `timeout 3600s` остановил процесс во второй эпохе, после сохранения `best.pt` за epoch 1.
- metrics:
  - `train/loss`: `0.9645444236828287`
  - `val/loss`: `0.9891750366588676`
  - `val/pixel_f1`: `0.018755369439880563`
  - `val/best_pixel_f1`: `0.018755369439880563`
  - `train/epochs_total`: `1`
  - `train/timeout_hit`: `1`
- MLflow artifacts:
  - `reports/dataset_preparation.json`
  - `checkpoints/best.pt`

## Лимит Времени
- команда: `timeout --preserve-status 3600s python -m mlsystem2.cli.train --config /opt/mlsystem2/runtime/first_train/deforestation_first_1024.yaml --run-name mlsystem2_first_deforestation_1024_20260523T161829Z`
- timeout: 3600s
- фактическое окно train process: `2026-05-23 19:18:29 MSK` - `2026-05-23 20:18:29 MSK`
- timeout hit: да
- MLflow run закрыт вручную после salvage logging: `2026-05-23 20:23:53 MSK`

## Артефакты
- best checkpoint: `/opt/mlsystem2/runtime/first_train/scratch/checkpoints/best.pt`
- final checkpoint: не создан, потому что timeout остановил процесс до штатного завершения `train_model`.
- train log: `/opt/mlsystem2/runtime/first_train/logs/mlsystem2_first_deforestation_1024_20260523T161829Z.log`
- gpu log: `/opt/mlsystem2/runtime/first_train/logs/mlsystem2_first_deforestation_1024_20260523T161829Z_gpu.log`
- checkpoint candidates: `/opt/mlsystem2/runtime/first_train/checkpoint_candidates.txt`

## Проверки
- `python -m pytest tests/test_public_contracts.py`: passed, 1 test
- `python -m pytest tests -q`: passed, 65 tests
- `python -m ruff check src tests`: passed
- dataset/tensor smoke: passed, raw image values confirmed above 1.0

## Проблемы И Исправления
- Первоначально был использован неправильный ручной clone `/data/MLSystem2`. Он проверен и удален; дальнейшая работа выполнена только из `/opt/mlsystem2/repo`.
- `/data/mlmarkup` отсутствовал, при этом CI/CD-разметка была в `/data/MLMarkup`. Создан symlink `/data/mlmarkup -> /data/MLMarkup`, и все дальнейшие обращения идут через `/data/mlmarkup`.
- CUDA была недоступна из-за mismatch NVIDIA kernel module/userland. Controlled reboot загрузил kernel module `580.159.03`; `nvidia-smi` и PyTorch CUDA после этого работают.
- После reboot контейнеры `mlsystem-gpu-minio`, `mlsystem-gpu-mlflow-postgres`, `mlsystem-gpu-mlflow` остались `Exited (0)`. Они подняты штатным `docker start`, MLflow health восстановлен.
- Первый train attempt `6b8f4554bfca4c80ab2ce65e108bf771` упал до обучения: host-side MLflow client не имел MinIO/S3 credentials для загрузки artifacts. Runtime launcher обновлен так, чтобы читать MinIO credentials из `/etc/mlsystem/gpu-platform.env` и выставлять `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000`.
- Часовой run `f6d425b888724ff180e8de24cf8d75f4` был остановлен timeout после epoch 1. Так как процесс был убит до штатного post-train logging, метрики epoch 1 и `best.pt` залогированы в тот же MLflow run из локального checkpoint metadata, после чего run закрыт статусом `KILLED`.
