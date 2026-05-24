# Отчет: B2 aug3 smart training после фильтра черных tiles

Дата: 2026-05-24.

## Контур

- Код: `2dbe0c7873d51adc31852cfe50b5ba3f8e6aaf52` (`Filter black tiles using valid footprint`).
- Серверный repo: `/opt/mlsystem2/repo`.
- Датасет: prepared images из `/data/mlsystem2/prepared_images/`.
- Разметка: `/data/MLMarkup/Вырубки/deforestation.geojson`.
- Scenes: `/data/MLMarkup/Вырубки/deforestation.txt`.
- `/data/MLSystem2` не использовался.
- Symlink `/data/mlmarkup` не создавался.

## Исправление black tiles

`tile_preparation` теперь строит candidate windows по VRT source rects, а затем фильтрует их через внутренний coarse valid-data footprint. Фильтр не читает каждый tile целиком: он делает low-resolution read masks/data по VRT и дополнительную sparse-проверку по тем же позициям, которые использует локальный black detector.

Внутренние параметры:

- `valid_footprint_stride`: `64`.
- `valid_value_eps`: `1e-6`.
- Новые публичные настройки не добавлялись.

Локальный `modules_test.py` после исправления:

- `uses_vrt_source_rects`: `true`.
- `source_rect_count`: `29`.
- `candidate_window_count_before_valid_filter`: `28603`.
- `candidate_window_count`: `24229`.
- `black_filtered_window_count`: `4374`.
- `dataset_len`: `24229`.
- `loader_len`: `6058`.
- `black_tiles`: `0`.
- `positive_black_tiles`: `0`.
- `nonfinite_image_tiles`: `0`.
- `positive_tiles`: `12002`.
- `negative_tiles`: `12227`.
- `real_tiles`: `12227`.
- `augmented_tiles`: `12002`.
- `positive_augmented_tiles`: `12002`.
- `negative_augmented_tiles`: `0`.
- `positive_mask_pixels`: `892704972`.

Вывод: VRT режется по source `DstRect`, а не по всему bbox. Черные windows были внутри прямоугольных source rects и теперь отфильтрованы до выдачи из `Dataset`.

## Проверки

Локально:

- `python -m pytest tests/test_public_contracts.py -q`: `1 passed`.
- `python -m pytest tests -q`: `86 passed, 3 skipped`.
- `python -m ruff check src tests`: `All checks passed`.

На сервере после деплоя:

- `python -m pytest tests/test_public_contracts.py -q`: `1 passed`.
- `python -m pytest tests -q`: `89 passed`.
- `python -m ruff check src ./tests`: `All checks passed`.

## Train config

Config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_aug3_smart_30ep_2h.yaml`.

Основные параметры:

- `model_name`: `segformer_b2`.
- `input_channels`: `4`.
- `output_channels`: `1`.
- `pretrained`: `false`.
- `epochs`: `30`.
- `batch_size`: `4`.
- `learning_rate`: `0.000002`.
- `loss`: `focal_tversky`.
- `tversky_alpha`: `0.40`.
- `tversky_beta`: `0.60`.
- `threshold`: `0.75`.
- `augmentation_level`: `3`.
- `smart_tiling`: `true`.
- `max_train_batches_per_epoch`: `72`.
- `max_val_batches_per_epoch`: `1000`.
- `max_training_time_sec`: `7200`.

## MLflow run

- Experiment: `MLSystem2-First`.
- Run id: `36ebd154de454f06b7e8d5e728e34ff6`.
- Run name: `deforestation_2405_6`.
- Status: `FINISHED`.
- Artifact URI: `s3://mlflow-artifacts/43/36ebd154de454f06b7e8d5e728e34ff6/artifacts`.
- Epochs completed: `30`.
- Epoch time sum: `3173.79 sec`.
- Optimizer steps: `2160`.
- Skipped optimizer steps: `0`.

Best epoch:

- epoch: `14`.
- train loss: `0.776380887048112`.
- val loss: `0.9811905849277973`.
- val pixel precision: `0.030317419822745127`.
- val pixel recall: `0.37551029562395316`.
- val pixel F1: `0.05610510493431182`.
- val positive pixels: `12172356`.
- val pred positive pixels: `150766293`.
- TP: `4570845`.
- FP: `146195448`.
- FN: `7601511`.

Final epoch:

- epoch: `30`.
- train loss: `0.7813059141238531`.
- val loss: `0.977446864426136`.
- val pixel precision: `0.01579169631265955`.
- val pixel recall: `0.4717099138408374`.
- val pixel F1: `0.0305603081175459`.
- val pred positive pixels: `363597481`.
- TP: `5741821`.
- FP: `357855660`.
- FN: `6430535`.

Вывод по обучению: non-finite gradient проблема ушла на этом запуске (`skipped_optimizer_steps=0`). F1 остается низким, но теперь он объясним счетчиками: модель предсказывает много positive pixels, TP есть, основной вклад в низкую precision дают FP.

## Tile report из MLflow

Train split:

- `tile_count`: `24229`.
- `batch_count`: `6058`.
- `source_rect_count`: `29`.
- `uses_vrt_source_rects`: `true`.
- `candidate_window_count_before_valid_filter`: `28603`.
- `black_filtered_window_count`: `4374`.
- `valid_footprint_stride`: `64`.
- `estimated_positive_tiles`: `1549`.
- `estimated_negative_tiles`: `22680`.
- `observed_batches`: `2160`.
- `observed_tiles`: `8640`.
- `observed_positive_tiles`: `4292`.
- `observed_augmented_tiles`: `4292`.
- `observed_real_tiles`: `4348`.
- `warnings`: `[]`.

Val split:

- `tile_count`: `4886`.
- `batch_count`: `1222`.
- `source_rect_count`: `6`.
- `uses_vrt_source_rects`: `true`.
- `candidate_window_count_before_valid_filter`: `6248`.
- `black_filtered_window_count`: `1362`.
- `valid_footprint_stride`: `64`.
- `observed_batches`: `30000`.
- `observed_tiles`: `120000`.
- `observed_positive_tiles`: `7230`.
- `observed_augmented_tiles`: `0`.
- `observed_real_tiles`: `120000`.
- `warnings`: `[]`.

## Checkpoints

- Best checkpoint: `/opt/mlsystem2/runtime/first_train/b2_aug3_smart_30ep_scratch/checkpoints/best.pt`.
- Final checkpoint: `/opt/mlsystem2/runtime/first_train/b2_aug3_smart_30ep_scratch/checkpoints/final.pt`.

## Export и Geoalert inference

Best checkpoint epoch 14 экспортирован в Triton model repository:

- ONNX: `/opt/geoalert/triton_models/mlsystem2_deforestation/1/model.onnx`.
- ONNX size: `112660349` bytes.
- Opset: `18`.
- Threshold: `0.75`.
- Output: `uint8 mask after sigmoid + threshold`.
- Triton model status: `READY`.
- `config.pbtxt` output dims обновлены до `[ 1, 1, -1, -1 ]`, потому новый ONNX экспорт явно возвращает `N,C,H,W`.

Geoalert inference по датасету вырубок:

- Pipeline: `/opt/geoalert/pipelines/mlsystem2_deforestation_triton.yaml`.
- Output root: `/opt/geoalert/runs/deforestation_after_training`.
- Scenes processed: `35`.
- Failed scenes: `0`.
- Missing inputs: `0`.
- Total mask sum: `353604222`.
- Total vector features: `18669`.

Подробный per-scene отчет записан в `docs/reports/geoalert_inference_after_training.md`.
