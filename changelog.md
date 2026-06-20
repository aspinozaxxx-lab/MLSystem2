# Changelog

## 2026-06-20

- Dobavlena edinaya supervision mask dlya hard negative: `-1` hard negative, `0` background, `1..N` positive.
- Train loss pereveden na pixel-level `hard_negative_weight` bez ispolzovaniya `tile_hard_negative` dlya vesov loss.
- Obnovleny UI tooltip, dokumentatsiya i testy dlya hard-negative pixel weighting.

## 2026-06-19

- Dobavlena podderzhka hard negative annotation file dlya binary i multiclass datasetov.
- Razdeleny positive, hard_negative i background tile kategorii v sampler, batch meta i tile preparation report.
- Obnovleny dataset discovery MLMarkup, UI schema/templates, worker run.yml i public docs pod novye tile factors.
