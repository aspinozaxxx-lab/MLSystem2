# HPO SegFormer B2 по вырубкам, итог 26.05

## Статус

- HPO остановлен вручную по запросу пользователя.
- STOP-файл создан, активных `mlsystem2.cli.train` процессов нет.
- MLflow experiment: [Segformer-b2-HPO-deforest-2605](http://127.0.0.1:5000/#/experiments/45).
- Baseline до HPO: `val/best_threshold_pixel_f1 ~= 0.28`.

## Лучший результат

Лучший подтверждённый run: [trial_0008 / Stage C](http://127.0.0.1:5000/#/experiments/45/runs/59b45400260c4e4da5d6f753244339b1).

| Метрика | Значение |
|---|---:|
| best F1 | `0.4687` |
| best threshold | `0.30` |
| best epoch | `58` |
| средняя эпоха | `128.8` с |
| рост к baseline | `+0.1887` |

Параметры победителя: `smp_segformer_b2`, `lr=1e-5`, `loss=focal_tversky`, `tversky=0.4/0.6`, `focal_alpha=0.6`, `positive_factor=0.8`, `augmentation_level=3`, `weight_decay=1e-4`, `batch_size=4`, `128` train batches.

Checkpoint: `/opt/mlsystem2/runtime/hpo/segformer_b2_deforest_2605/scratch/trial_0008/checkpoints/best.pt`.

## Важные runs

| Trial | Stage | Run | Best F1 | Комментарий |
|---:|---:|---|---:|---|
| 0008 | C | [59b454...](http://127.0.0.1:5000/#/experiments/45/runs/59b45400260c4e4da5d6f753244339b1) | `0.4687` | лучший подтверждённый |
| 0007 | B | [53126c...](http://127.0.0.1:5000/#/experiments/45/runs/53126c7c57e74032931e46e89cfb61e4) | `0.4086` | первый сильный Stage B |
| 0022 | B | [804851...](http://127.0.0.1:5000/#/experiments/45/runs/80485145fca943c8a177c796bb0129aa) | `0.3306` | `focal_alpha=0.75`, упал из-за MLflow/Postgres |
| 0006 | A | [713c3c...](http://127.0.0.1:5000/#/experiments/45/runs/713c3cf712e141bb8d7080453fba4a00) | `0.2883` | быстрый trial, дал направление |
| 0016 | A | [30e5d8...](http://127.0.0.1:5000/#/experiments/45/runs/30e5d86cfb37447f8b7bebf88a129267) | `0.2845` | `focal_alpha=0.75`, слабее champion |
| 0023 | B | [9229cb...](http://127.0.0.1:5000/#/experiments/45/runs/9229cb1d9cfe486ba4170c45ca92b94e) | `0.1936` | остановлен при закрытии HPO |

## Выводы

- Главная находка: `lr=1e-5` резко лучше стартового `2e-6/5e-6`.
- Лучший loss: `focal_tversky`; `bce_dice` и `focal_dice` заметно слабее.
- Лучший баланс: `positive_factor=0.8`; `0.85/0.9` ухудшали стабильность.
- `augmentation_level=3` лучше `2`.
- `weight_decay=1e-4` лучше `1e-5` и `3e-4`.
- `focal_alpha=0.75` не побил `0.6`; в Stage B успел дойти до `0.3306`, но отставал от champion.

## Инфраструктура

- `trial_0022` оборвался из-за заполненного root filesystem: MLflow Postgres ушёл в recovery и вернул 503 на логирование метрик.
- Освобождено место удалением старых регенерируемых Geoalert outputs; MLflow/Postgres снова `healthy`.
- На момент остановки доступно около `42G` на `/`.

## Что использовать

Для дальнейшего обучения или инференса брать checkpoint `trial_0008`. Следующий разумный шаг, если HPO продолжать позже: не расширять поиск, а подтверждать champion на другом seed и/или полном validation.
