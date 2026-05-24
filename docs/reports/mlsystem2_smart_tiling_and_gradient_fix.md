# Отчет: smart tiling и защита от non-finite gradients

Дата: 2026-05-24.

## Контекст

B2 diagnostic показал, что ранние background batch могли давать non-finite gradient, после чего optimizer step пропускался. Такой skip оставлен только как аварийная защита: если он повторяется чаще одного раза за эпоху, обучение падает с `TrainError`.

Рабочая причина нестабильности: raw Geoalert-compatible вход `0..255` подавался в SegFormer без внутреннего scaling. Для сохранения ABI tile preparation и Geoalert inference добавлен внутренний wrapper модели: `x.float() / 255.0` внутри `segformer_b0` и `segformer_b2`.

## Изменения

- В `TilePreparationSettings` добавлен единственный новый флаг `smart_tiling: bool = false`.
- При `smart_tiling=true` train loader использует positive-aware `WeightedRandomSampler` по cheap-index пересечений window bounds с геометриями разметки.
- При `smart_tiling=true` augmentation применяется только к positive tiles; negative/background tiles не аугментируются.
- `batch_meta` расширен счетчиком `positive_tile_count`.
- В tile report добавлены `smart_tiling_enabled`, source rect diagnostics, estimated positive/negative tiles и observed positive/augmented counters.
- MLflow пишет `train/optimizer_steps`, `train/skipped_optimizer_steps` и диагностические F1-счетчики `TP/FP/FN`.
- Config YAML запуска и `reports/tile_preparation.json` сохраняются в MLflow artifacts.

## Серверная проверка

Серверный repo: `/opt/mlsystem2/repo`.

Деплой-коммит: `4ce37549ac674077c15055c770e43206f29a0c25`.

Подготовленные снимки используются из `/data/mlsystem2/prepared_images/`.

Проверки на сервере:

- `python -m pytest tests/test_public_contracts.py -q`: `1 passed`.
- `python -m pytest tests -q`: `85 passed`.
- `python -m ruff check src ./tests`: `All checks passed`.

Примечание по доставке: GitHub Actions run `26364436311` завис на шаге установки GDAL CLI до копирования кода. Чтобы не блокировать диагностику, на сервер был развернут архив ровно этого коммита; `DEPLOYED_COMMIT` выставлен в тот же SHA.

## Diagnostic run

Config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_smart_diag.yaml`.

MLflow run: `5d25132534234422880f3dbc0a09706b` (`deforestation_2405_4`).

Статус: `FINISHED`.

Основные параметры:

- model: `segformer_b2`.
- epochs: `2`.
- batch size: `4`.
- learning rate: `2.0e-06`.
- threshold: `0.75`.
- `max_train_batches_per_epoch: 72`.
- `max_val_batches_per_epoch: 1000`.
- `smart_tiling: true`.

## Tile report

Train split:

- `tile_count`: `28603`.
- `batch_count`: `7151`.
- `source_rect_count`: `29`.
- `uses_vrt_source_rects`: `true`.
- `estimated_positive_tiles`: `1588`.
- `estimated_negative_tiles`: `27015`.
- `observed_batches`: `144`.
- `observed_tiles`: `576`.
- `observed_positive_tiles`: `274`.
- `observed_augmented_tiles`: `256`.
- `observed_real_tiles`: `320`.

Val split:

- `tile_count`: `6248`.
- `batch_count`: `1562`.
- `source_rect_count`: `6`.
- `uses_vrt_source_rects`: `true`.
- `observed_batches`: `2000`.
- `observed_tiles`: `8000`.
- `observed_positive_tiles`: `306`.
- `observed_augmented_tiles`: `0`.
- `observed_real_tiles`: `8000`.

Вывод: тысячи batch строятся по source rects VRT, не по fallback bbox grid. Для train `observed_augmented_tiles <= observed_positive_tiles`, значит negative/background tiles не раздуваются augmentation.

## Metrics

Epoch 1:

- `train/optimizer_steps`: `72`.
- `train/skipped_optimizer_steps`: `0`.
- `val/positive_pixels`: `7553992`.
- `val/pred_positive_pixels`: `474420232`.
- `TP`: `3861969`.
- `FP`: `470558263`.
- `FN`: `3692023`.
- `F1`: `0.016025624640042993`.

Epoch 2:

- `train/optimizer_steps`: `72`.
- `train/skipped_optimizer_steps`: `0`.
- `val/positive_pixels`: `7553992`.
- `val/pred_positive_pixels`: `730262093`.
- `TP`: `5217362`.
- `FP`: `725044731`.
- `FN`: `2336630`.
- `F1`: `0.014142716880454024`.

Вывод по F1: validation positives есть, predictions не нулевые, TP есть. Нулевой или странный F1 больше не объясняется пустыми масками. Низкий F1 в этом запуске объясняется большим числом false positive при threshold `0.75`.

## Решение по long B2

Diagnostic успешен для запуска long:

- skipped optimizer steps: `0`.
- val содержит positive pixels.
- source rects VRT используются.
- smart tiling включен.
- augmentation не применяется к negative/background tiles.
- F1 объясним через `TP/FP/FN`.

Long B2 запущен:

- Config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_smart_long.yaml`.
- MLflow run: `947d08be58684676907ad327ca3d4ce3` (`deforestation_2405_5`).
- Process PID: `2455101`.
- Log: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_smart_long_20260524T153436Z.log`.
- Timeout: `36000s`.
- `images_dir`: `/data/mlsystem2/prepared_images/`.
- `smart_tiling: true`.
- `max_train_batches_per_epoch: 72`.
- `max_val_batches_per_epoch: 1000`.

После long run нужно отдельно смотреть динамику FP и при необходимости подбирать threshold или loss/initialization, но блокеры по маскам, sampler size и non-finite gradients сняты.
