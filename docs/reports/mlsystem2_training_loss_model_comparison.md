# Сравнение loss и model path для обучения MLSystem2

Дата: 2026-05-25.

## Причина проверки

После фильтра black tiles, `smart_tiling`, balanced train sampling и расширенной validation диагностики strict tiny-overfit на реальных 32 tiles не проходил: `best-threshold F1=0.333`, при требуемом `>=0.80`. Новые long-run не запускались.

Новая гипотеза: MLSystem2 не повторял старый train path:

- старый MLSystem использовал `segmentation_models_pytorch.Segformer` с encoder `mit_b0/mit_b2`;
- старый `focal_tversky` был суммой `focal + tversky`;
- MLSystem2 использовал Hugging Face `SegformerForSemanticSegmentation`;
- MLSystem2 ошибочно считал `focal_tversky` как `(tversky_loss)^2`.

## Изменения в коде

Кодовый commit: `2d744e5c6d5acc5f8be44dbd001dd28d53546842`.

- `focal_tversky` исправлен на `focal_loss + tversky_loss`.
- В `EpochMetrics` добавлены компоненты train loss:
  - `train_loss_focal`;
  - `train_loss_tversky`;
  - `train_loss_bce`;
  - `train_loss_dice`.
- В MLflow добавлено логирование:
  - `train/loss_focal`;
  - `train/loss_tversky`;
  - `train/loss_bce`;
  - `train/loss_dice`.
- В `models` добавлены отдельные диагностические имена:
  - `smp_segformer_b0`;
  - `smp_segformer_b2`.
- Hugging Face модели `segformer_b0/b2` оставлены без переименования и с wrapper `x / 255.0`.
- SMP модели строятся как старый MLSystem path: `segmentation_models_pytorch.Segformer`, `encoder_weights=None`, без wrapper `x / 255.0`.
- `train_overfit_test.py` расширен:
  - single overfit;
  - real/synthetic mode;
  - matrix mode;
  - `tiny_unet_4ch`;
  - сохранение 16 overlays и per-sample stats.

На сервере в venv установлен пакет `segmentation-models-pytorch==0.5.0`.

## Проверки

Локально:

- `python -m pytest tests/test_public_contracts.py -q` - passed.
- `python -m pytest tests -q` - `91 passed, 4 skipped`.
- `python -m ruff check src tests` - passed.

Сервер:

- `python -m pytest tests/test_public_contracts.py -q` - passed.
- `python -m pytest tests -q` - `94 passed, 1 skipped`.
- `python -m ruff check src ./tests` - passed.
- После установки SMP: `tests/test_models_segformer.py tests/test_train_loop.py` - `15 passed`.

## Synthetic overfit matrix

Команда:

```bash
python -m mlsystem2.cli.train_overfit_test \
  --config /opt/mlsystem2/runtime/first_train/deforestation_b2_posfactor_diag.yaml \
  --synthetic \
  --matrix \
  --steps 500 \
  --device cuda \
  --report /opt/mlsystem2/runtime/first_train/synthetic_overfit_matrix_report.json
```

Итог:

- status: `ok`
- `smp_focal_tversky_passed`: `true`
- sample overlays: `/opt/mlsystem2/runtime/first_train/synthetic_overfit_samples/`

Cases:

| model | loss | status | best-threshold F1 | final F1 | вывод |
|---|---|---:|---:|---:|---|
| `segformer_b0` | `bce_dice` | error | - | - | non-finite gradients |
| `segformer_b0` | `focal_tversky` | error | - | - | non-finite `grad_norm` |
| `smp_segformer_b0` | `bce_dice` | ok | 0.9831 | 0.9755 | проходит |
| `smp_segformer_b0` | `focal_tversky` | ok | 0.9728 | 0.9719 | проходит |
| `tiny_unet_4ch` | `bce_dice` | ok | 1.0000 | 1.0000 | проходит |
| `tiny_unet_4ch` | `focal_tversky` | ok | 1.0000 | 1.0000 | проходит |

Вывод по synthetic: train loop, metric, corrected `focal_tversky`, SMP path и tiny model способны переобучиться на простой задаче. Hugging Face `segformer_b0` нестабилен даже на synthetic input, что отдельно подтверждает риск текущего HF model path.

## Real overfit matrix

Команда:

```bash
python -m mlsystem2.cli.train_overfit_test \
  --config /opt/mlsystem2/runtime/first_train/deforestation_b2_posfactor_diag.yaml \
  --matrix \
  --steps 500 \
  --device cuda \
  --report /opt/mlsystem2/runtime/first_train/overfit_matrix_report.json
```

Итог:

- status: `failed`
- `smp_focal_tversky_passed`: `false`
- sample overlays: `/opt/mlsystem2/runtime/first_train/overfit_samples/`

Cases:

| model | loss | status | best-threshold F1 | final F1 | prob positive mean | prob negative mean |
|---|---|---:|---:|---:|---:|---:|
| `segformer_b0` | `bce_dice` | failed | 0.3768 | 0.0774 | 0.4028 | 0.1259 |
| `segformer_b0` | `focal_tversky` | failed | 0.5259 | 0.5251 | 0.6553 | 0.1466 |
| `smp_segformer_b0` | `bce_dice` | failed | 0.4083 | 0.3089 | 0.4084 | 0.0560 |
| `smp_segformer_b0` | `focal_tversky` | failed | 0.6790 | 0.6742 | 0.7186 | 0.0452 |
| `tiny_unet_4ch` | `bce_dice` | failed | 0.2447 | 0.0010 | 0.1968 | 0.0606 |
| `tiny_unet_4ch` | `focal_tversky` | failed | 0.2567 | 0.1477 | 0.3445 | 0.1130 |

Лучший real case - `smp_segformer_b0 + focal_tversky`, но `best-threshold F1=0.6790`, ниже обязательного gate `0.80`.

## Alignment stats по real samples

Для 16 positive samples сохранены:

- `sample_XXXX_preview_rgb.png`;
- `sample_XXXX_mask.png`;
- `sample_XXXX_overlay.png`;
- `sample_XXXX_stats.json`.

Путь: `/opt/mlsystem2/runtime/first_train/overfit_samples/`.

Численный sanity по разнице средних значений внутри/снаружи mask показывает неоднородную картину. Примеры `inside_outside_abs_diff` по каналам:

- sample 0: `[12.896, 13.756, 8.461, 20.432]`
- sample 3: `[0.314, 0.203, 0.662, 3.176]`
- sample 5: `[0.074, 0.518, 0.061, 7.465]`
- sample 12: `[15.309, 10.145, 8.398, 24.999]`

Это не доказывает ошибку разметки само по себе, но объясняет, почему real tiny-overfit намного сложнее synthetic: часть positive masks почти не отличается от surrounding pixels по простым каналам. Нужна визуальная проверка overlays и, вероятно, проверка соответствия старому MLSystem preprocessing/mask generation.

## Вывод

Найден и исправлен реальный bug: `focal_tversky` в MLSystem2 не соответствовал старому MLSystem. Добавлен старый SMP model path для прямого сравнения.

После фикса:

- synthetic overfit проходит на SMP и tiny model;
- real overfit не проходит ни на HF, ни на SMP, ни на tiny model;
- лучший real result дает именно old-style path `smp_segformer_b0 + focal_tversky`, но он не достигает gate `0.80`;
- HF SegFormer остается подозрительным: на synthetic matrix он дает non-finite gradients.

Следующий long-run запускать нельзя. Следующий диагностический шаг - визуально и численно сравнить real overlays с исходной разметкой и старым MLSystem data path: rasterize transform, CRS, mask alignment, preprocessing prepared images, channel order и то, какие именно positive tiles выбирал старый sampler.
