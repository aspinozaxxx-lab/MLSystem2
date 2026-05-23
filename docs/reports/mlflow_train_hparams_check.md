# Проверка train hyperparams в MLflow

Дата проверки: 2026-05-23.

Источник: SSH `root@31.192.104.147`.

## Доступность

- `/data/mlsystem/mlruns` отсутствует.
- `/data/mlsystem/runs` существует и содержит локальные run-директории.
- `/data/mlsystem/airflow/status` существует и содержит stage JSON.
- Внешний URL `http://31.192.104.147/mlflow/...` закрыт login-формой.
- Через SSH доступен backend MLflow: `http://127.0.0.1:5000`.
- Experiment id `38` доступен как `mlsystem-class-training`, а не как `mlsystem-cuttings-tuning`.
- Указанные run id найдены в MLflow backend и/или artifact layout.

## Проверенные runs

| run id | experiment | run name | model | lr | wd | loss | pos_weight | batch | epochs | tile | stride | scheduler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `b97303dcad8b4f368696513f193b995c` | 40 | `cuttings_ft_20260515_0748_ckpt1_lr1e5_bcedice_512_b8` | `segformer_b2` | `1e-05` | `0.0001` | `bce_dice` | `1.0` | `8` | `3` | `512` | `512` | `cosine` |
| `0ac0e246a7ef429aa60cf2b29d038d06` | 40 | `cuttings_ft_20260515_0649_ckpt1_lr1e5_bcedice_512_b4` | `segformer_b2` | `1e-05` | `0.0001` | `bce_dice` | `1.0` | `4` | `3` | `512` | `512` | `cosine` |
| `ef174cae4b764e66832b2dd09451ecaa` | 40 | `cuttings_ft_20260515_0648_ckpt1_lr2e5_focaldice_512_b4` | `segformer_b2` | `2e-05` | `0.0001` | `focal_dice` | `1.0` | `4` | `3` | `512` | `512` | `cosine` |
| `9d23e90adb15494bbb2987b5e24962fd` | 38 | `manual_deforestation_20260514_002020_posweight_recall` | `segformer_b2` | `5e-05` | `0.0001` | `focal_dice` | `2.0` | `2` | `18` | `1024` | `512` | `none` |
| `f0b8f5c089ef4ffabda9742ac76540d4` | 38 | `manual_deforestation_20260514_000042_fulltiles_nocache` | `segformer_b2` | `5e-05` | `0.0001` | `focal_dice` | не задан | `2` | `25` | `1024` | `512` | `none` |

## Вывод для MLSystem2

В settings переносится только минимальный набор train-параметров, который найден в runs или нужен
для цикла обучения SegFormer B2: `learning_rate`, `weight_decay`, `loss`, `pos_weight`, `threshold`,
`batch_size`, `epochs`, `device`, `pretrained`, `initial_checkpoint_uri`,
`early_stopping_patience`, а также параметры вариантов loss `focal_alpha`, `tversky_alpha`,
`tversky_beta`.

Optimizer фиксируется как AdamW, scheduler фиксируется как cosine. Эти значения не выносятся в
settings, потому что текущая задача требует только SegFormer B2 и нет необходимости делать их
публичными гиперпараметрами.
