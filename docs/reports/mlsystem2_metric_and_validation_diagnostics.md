# Диагностика метрик и validation sampling MLSystem2

Дата: 2026-05-25.

## Контекст

После black-tile filter, `smart_tiling=true` и 30 эпох B2 pixel F1 остался низким: precision очень низкая, recall растет, модель предсказывает слишком много positive. Поэтому новые long-run не запускались вслепую; сначала проверены validation sampling, tensor integrity, диапазон аугментаций и tiny-overfit.

Текущий код: `3d80d2b8a5b0c20254f5138c900d51b978d890cd`.

## Что исправлено

- `val_positive_factor` добавлен в `TilePreparationSettings` как диагностический sampler только для `mode=val`, только при `smart_tiling=true`.
- Positive/negative hints теперь строятся для `train` и `val`, если включен `smart_tiling`.
- Validation threshold sweep расширен до `[0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]`.
- В MLflow добавлено логирование `val/sweep_*`, `val/prob_p999`, `val/prob_positive_*`, `val/prob_negative_*`.
- После аугментаций image зажимается в raw диапазон `0..255`, mask остается `0/1`.
- `modules_test.py` сохраняет overlay для mask и отдельный tensor-integrity пример.
- `train_overfit_test.py` стал строгим: 300 шагов по умолчанию и success threshold `best-threshold F1 >= 0.80`.

## Локальный modules_test

Запуск:

```powershell
$env:PYTHONPATH="src"
python src\mlsystem2\cli\modules_test.py
```

Отчет: `D:\Projects\test\modules_test_timing_report.json`.
Overlay examples: `D:\Projects\test\tile_batches\batch_0000\*_preview_mask_overlay.png`.
Tensor integrity examples: `D:\Projects\test\tensor_integrity\`.

Итог scan:

- `total_batches`: 6058
- `total_tiles`: 24229
- `black_tiles`: 0
- `positive_black_tiles`: 0
- `positive_tiles`: 19198
- `negative_tiles`: 5031
- `positive_augmented_tiles`: 19198
- `negative_augmented_tiles`: 0
- `image_min`: 0.0
- `image_max`: 255.0
- `source_rect_count`: 29
- `candidate_window_count`: 24229
- `candidate_window_count_before_valid_filter`: 28603
- `black_filtered_window_count`: 4374
- `uses_vrt_source_rects`: true
- `augmentation_level`: 3
- `smart_tiling`: true
- `positive_factor`: 0.8
- `val_positive_factor`: 0.5

Важно: `positive_tiles` и `negative_tiles` здесь являются observed samples одного полного прохода train DataLoader с weighted sampler и replacement, а не количеством уникальных окон.

Tensor integrity для первого positive tile:

- `channels`: 4
- `channel_min`: `[0.0, 0.0, 0.0, 0.0]`
- `channel_max`: `[86.0780, 92.3394, 67.2939, 231.1332]`
- `channel_mean`: `[47.2064, 65.0125, 47.0846, 144.2647]`
- `all_channels_equal`: false
- `nonzero_channel_count`: 4

Вывод: массовых black tiles больше нет, negative tiles не аугментируются, raw range после аугментаций не выходит за `0..255`, 4 канала присутствуют и не являются одинаковыми.

## Val hints

Серверный preflight на `/opt/mlsystem2/repo`:

- train:
  - `source_rect_count`: 29
  - `candidate_window_count`: 24229
  - `candidate_window_count_before_valid_filter`: 28603
  - `black_filtered_window_count`: 4374
  - `uses_vrt_source_rects`: true
  - `estimated_positive_tiles`: 1549
  - `estimated_negative_tiles`: 22680
- val:
  - `source_rect_count`: 6
  - `candidate_window_count`: 4886
  - `candidate_window_count_before_valid_filter`: 6248
  - `black_filtered_window_count`: 1362
  - `uses_vrt_source_rects`: true
  - `estimated_positive_tiles`: 272
  - `estimated_negative_tiles`: 4614

Вывод: прежний `val.estimated_positive_tiles=null` закрыт. Значения являются cheap geometry-intersection hints, не результатом полной rasterize materialization.

## Tiny-overfit

Запуск на сервере:

```bash
python -m mlsystem2.cli.train_overfit_test \
  --config /opt/mlsystem2/runtime/first_train/deforestation_b2_posfactor_diag.yaml \
  --model segformer_b0 \
  --steps 300 \
  --learning-rate 0.0001 \
  --threshold-sweep \
  --report /opt/mlsystem2/runtime/first_train/overfit_test_report.json
```

Результат: `failed`.

- `positive_tiles_used`: 16
- `negative_tiles_used`: 16
- `actual_optimizer_steps`: 304
- `initial_f1`: 0.0
- `final_f1`: 0.2596
- `final_precision`: 0.1602
- `final_recall`: 0.6842
- `final_best_threshold`: 0.995
- `final_best_threshold_f1`: 0.3328
- `best_epoch`: 36
- `best_threshold`: 0.99
- `best_threshold_f1`: 0.3330
- `best_threshold_precision`: 0.2559
- `best_threshold_recall`: 0.4765
- `final_prob_positive_mean`: 0.7950
- `final_prob_negative_mean`: 0.3001

Sample overlays сохранены в `/opt/mlsystem2/runtime/first_train/overfit_samples/`.

Вывод: сеть начала отделять GT-positive pixels от GT-negative pixels по вероятностям, но не смогла переобучиться на 32 tiles до требуемого `best-threshold F1 >= 0.80`. Это блокер для новых long-run.

## Diagnostic B2

Diagnostic B2 run с `val_positive_factor=0.5` не запускался, потому strict tiny-overfit не пройден. Это намеренное решение: пока модель/loss/mask/metric не переобучаются на маленьком наборе, long или medium B2 run не является доказательным.

## Вывод

Проверки закрыли явные ошибки data path:

- VRT режется по source rects, не по общему bbox.
- Black/nodata-only tiles отфильтрованы.
- Negative tiles при `smart_tiling=true` не аугментируются.
- Аугментации не выводят image за raw диапазон `0..255`.
- Val positive/negative hints теперь есть.
- Tensor integrity по локальному scan выглядит корректно: 4 канала, не все нулевые, не все одинаковые.

Оставшийся блокер находится глубже в train path: loss/model/output resizing/mask alignment/threshold calibration. Следующий шаг перед любым новым long-run - добиться tiny-overfit `best-threshold F1 >= 0.80` на 16 positive + 16 negative tiles или найти конкретную причину, почему это невозможно.

## Проверки

- Локально: `python -m pytest tests/test_public_contracts.py -q` - passed.
- Локально: `python -m pytest tests -q` - `90 passed, 3 skipped`.
- Локально: `python -m ruff check src tests` - passed.
- Сервер: `python -m pytest tests/test_public_contracts.py -q` - passed.
- Сервер: `python -m pytest tests -q` - `93 passed`.
- Сервер: `python -m ruff check src ./tests` - passed.
