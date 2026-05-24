# Отчет: диагностика низкого pixel F1 в MLSystem2

Дата: 2026-05-25.

## Контур

- Код: `f2617ece4a660b1ec53ff6bffa29b91c3834a731`.
- Серверный repo: `/opt/mlsystem2/repo`.
- Prepared images: `/data/mlsystem2/prepared_images/`.
- MLMarkup: `/data/MLMarkup/Вырубки`.
- `/data/MLSystem2` не использовался.
- Symlink `/data/mlmarkup` не создавался.

## Найденная ошибка

В `tile_preparation._augmentations._cutout` cutout/dropout занулял только `image`, но оставлял `mask` неизменной. Для segmentation это создавало противоречивую train pair: в вырезанном черном прямоугольнике target мог оставаться positive.

Исправление:

- `_cutout(image, mask, rng)` зануляет один и тот же rectangle в `image` и `mask`;
- `apply_augmentations` возвращает согласованные `image, mask`;
- добавлен unit test, который проверяет, что cutout region занулен в обоих массивах.

Это критичный data-path фикс. Он не меняет публичный API.

## Positive sampler

Добавлено поле `tile_preparation.positive_factor`.

- Default: `0.5`.
- Для diagnostic/нового обучения: `0.8`.
- Используется только при `smart_tiling=true` и `mode=train`.
- Суммарный вес positive windows: `positive_factor`.
- Суммарный вес negative windows: `1 - positive_factor`.
- Val loader не меняется.
- Synthetic labels не создаются, masks не меняются.

Unit test проверяет, что при 2 positive и 8 negative hints суммарный вес positive равен `0.8`, negative равен `0.2`.

## Preview diagnostics

`modules_test.py` теперь сохраняет дополнительный preview:

- `tile_XXXX_preview_mask_overlay.png`.

Overlay рисует только красный контур mask без заливки. Это нужно, чтобы визуально проверить попадание разметки на снимок, не закрывая RGB preview.

## Threshold sweep и probability stats

Validation теперь считает основные метрики по `train.threshold`, как раньше, и дополнительно фиксированный sweep:

- thresholds: `[0.3, 0.5, 0.7, 0.75, 0.8, 0.9]`;
- `val_best_threshold`;
- `val_best_threshold_pixel_f1`;
- `val_best_threshold_precision`;
- `val_best_threshold_recall`.

Также добавлена потоковая histogram-оценка вероятностей:

- `val_prob_mean`;
- `val_prob_min`;
- `val_prob_max`;
- `val_prob_p50`;
- `val_prob_p90`;
- `val_prob_p99`.

Эти метрики пишутся в MLflow live на каждой эпохе.

## Проверки

Локально:

- `python -m pytest tests/test_public_contracts.py -q`: `1 passed`.
- `python -m pytest tests -q`: `89 passed, 3 skipped`.
- `python -m ruff check src tests`: `All checks passed`.

На сервере после deploy:

- `python -m pytest tests/test_public_contracts.py -q`: `1 passed`.
- `python -m pytest tests -q`: `92 passed`.
- `python -m ruff check src ./tests`: `All checks passed`.

## Tiny-overfit diagnostic

Команда:

```bash
python -m mlsystem2.cli.train_overfit_test \
  --config /opt/mlsystem2/runtime/first_train/deforestation_b2_posfactor_diag.yaml \
  --model segformer_b0
```

Отчет:

- `/opt/mlsystem2/runtime/first_train/overfit_test_report.json`.

Результат:

- status: `ok`;
- model: `segformer_b0`;
- device: `cuda`;
- positive tiles: `16`;
- negative tiles: `16`;
- initial F1: `0.0`;
- final F1: `0.21040757526528722`;
- final precision: `0.12364453191433455`;
- final recall: `0.7053876393898187`;
- final best threshold: `0.9`;
- final best threshold F1: `0.27629584167088833`;
- epochs: `20`;
- training time: `23.47 sec`;
- elapsed including dataset collection/model setup: `176.48 sec`.

Вывод: model/loss/optimizer/masks/metric в принципе работают. Если бы train path был полностью сломан, tiny dataset не начал бы переобучаться. Это снимает главный блокер перед новым controlled long-run.

## B2 positive_factor diagnostic

Config:

- `/opt/mlsystem2/runtime/first_train/deforestation_b2_posfactor_diag.yaml`.

MLflow:

- run id: `82e33d7de7fd410b8eedf936ff826340`;
- run name: `deforestation_2505_2`;
- status: `FINISHED`.

Параметры:

- `segformer_b2`;
- `epochs`: `2`;
- `batch_size`: `4`;
- `augmentation_level`: `3`;
- `smart_tiling`: `true`;
- `positive_factor`: `0.8`;
- `max_train_batches_per_epoch`: `72`;
- `max_val_batches_per_epoch`: `1000`;
- `threshold`: `0.75`.

Tile report, train:

- `tile_count`: `24229`;
- `candidate_window_count_before_valid_filter`: `28603`;
- `black_filtered_window_count`: `4374`;
- `uses_vrt_source_rects`: `true`;
- `estimated_positive_tiles`: `1549`;
- `estimated_negative_tiles`: `22680`;
- `observed_batches`: `144`;
- `observed_tiles`: `576`;
- `observed_positive_tiles`: `465`;
- `observed_augmented_tiles`: `465`;
- `observed_real_tiles`: `111`;
- observed positive ratio: `80.7%`;
- negative augmented tiles: `0`;
- warnings: `[]`.

Tile report, val:

- `tile_count`: `4886`;
- `black_filtered_window_count`: `1362`;
- `observed_batches`: `2000`;
- `observed_tiles`: `8000`;
- `observed_positive_tiles`: `482`;
- `observed_augmented_tiles`: `0`;
- warnings: `[]`.

Epoch 1:

- train loss: `0.7863860875368118`;
- skipped optimizer steps: `0`;
- val F1 at threshold 0.75: `0.006618169910841665`;
- precision: `0.003892070716254108`;
- recall: `0.02209177910997674`;
- TP: `268909`;
- FP: `68822590`;
- FN: `11903447`;
- best threshold: `0.7`;
- best threshold F1: `0.01996392270594786`;
- prob mean: `0.44620293846726417`;
- prob p50/p90/p99: `0.4975 / 0.6775 / 0.7785`.

Epoch 2:

- train loss: `0.7754155728552077`;
- skipped optimizer steps: `0`;
- val F1 at threshold 0.75: `0.006883967208650018`;
- precision: `0.004145957211588336`;
- recall: `0.02027109624463826`;
- TP: `246747`;
- FP: `59268339`;
- FN: `11925609`;
- best threshold: `0.7`;
- best threshold F1: `0.017153583984599933`;
- prob mean: `0.3773736119884998`;
- prob p50/p90/p99: `0.4095 / 0.6445 / 0.7725`.

Вывод по diagnostic: sampler теперь реально дает примерно 80% positive samples, negative tiles не аугментируются, skipped optimizer steps нет, validation содержит positive pixels и threshold/probability диагностику. Низкий F1 на 2 эпохах объясняется сочетанием малого числа optimizer steps и большого числа FP/FN; это уже не выглядит как нулевые masks, black tiles или non-finite train failure.

## Новый controlled long-run

Так как tiny-overfit прошел, запущен новый controlled long-run, но уже не вслепую:

- config: `/opt/mlsystem2/runtime/first_train/deforestation_b2_posfactor_aug3_128b_30ep_2h.yaml`;
- scratch: `/opt/mlsystem2/runtime/first_train/b2_posfactor_aug3_128b_30ep_scratch`;
- `epochs`: `30`;
- `max_train_batches_per_epoch`: `128`;
- `max_val_batches_per_epoch`: `1000`;
- `max_training_time_sec`: `7200`;
- MLflow run id: `c0ca29edc8254c35a4d1b8e18b67b13d`;
- run name: `deforestation_2505_3`;
- server pid: `3297509`;
- log: `/opt/mlsystem2/runtime/first_train/logs/deforestation_b2_posfactor_aug3_128b_30ep_20260524T214627Z.log`.

На момент записи отчета run был `RUNNING`, GPU utilization около `90-98%`, memory около `28193 / 32607 MiB`. Первая видимая MLflow epoch metric:

- `val/pixel_f1`: `0.026808005494856242`;
- `val/best_threshold`: `0.75`;
- `val/best_threshold_pixel_f1`: `0.026808005494856242`;
- `train/loss`: `0.7517089534085244`;
- `train/skipped_optimizer_steps`: `0`.

## Итог

Явные ошибки training data path исправлены:

- cutout больше не создает positive mask на зануленном image patch;
- train sampler теперь использует заданную positive долю, а не жесткие 50/50;
- negative tiles при `smart_tiling=true` не аугментируются;
- black/nodata tiles остаются отфильтрованными.

Tiny-overfit показывает, что базовая связка model/loss/optimizer/masks/metric работоспособна. Следующий главный вопрос уже не “почему все сломано”, а насколько B2 успевает сойтись на полном train distribution при 128 batch/epoch и `positive_factor=0.8`, и какой threshold даст лучшую precision/recall точку.
