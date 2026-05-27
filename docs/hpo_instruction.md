# Инструкция для Codex-оркестратора HPO

Этот документ описывает, как внешний Codex CLI должен вести HPO-сессию для MLSystem2. Оркестратор работает действиями агента: подключается к серверу, читает MLflow, запускает trials, принимает решения, при необходимости вносит минимальные изменения в код и пишет отчеты.

Не создавай отдельный HPO runner script. Не добавляй новый CLI. Не меняй код обучения без явной причины.

## 1. Назначение

Codex-оркестратор должен:

- вести HPO-сессию без заранее написанного HPO-скрипта;
- запускать один trial за другим;
- не ждать окончания явно плохих trials;
- анализировать предыдущие результаты MLflow;
- выбирать следующую стратегию на основе фактов;
- при необходимости вносить минимальные изменения в код MLSystem2;
- прогонять тесты после изменений;
- фиксировать выводы в отчетах.

Оркестратор не должен превращать HPO в жесткий grid search. Его задача - исследовать пространство параметров итеративно: гипотеза, trial, наблюдение, решение, следующая гипотеза.

## 2. Входные параметры сессии

Оператор в стартовом prompt должен указать:

- `model_name`;
- dataset/class;
- MLflow experiment name;
- primary metric;
- max wall time per trial или session policy;
- report workspace;
- можно ли менять код;
- можно ли запускать Geoalert inference после удачных trials.

Если оператор не указал значения, используй defaults:

- report workspace: `/opt/hpo/report`;
- repo on GPU server: `/opt/mlsystem2/repo`;
- tracking URI: `http://127.0.0.1:5000`;
- GPU server: `gpu-mlserver`, fallback `root@31.192.104.147`.

Перед первым trial запиши фактические параметры сессии в `session_state.json` и краткую стартовую стратегию в `current_strategy.md`.

## 3. Безопасность и инфраструктура

Неизменные правила:

- не использовать `/data/MLSystem2`;
- не создавать `/data/mlmarkup` и symlink на этот путь;
- MLMarkup source of truth: `/data/MLMarkup`;
- prepared images: `/data/mlsystem2/prepared_images/`;
- не коммитить secrets, keys, токены и содержимое env-файлов;
- MinIO/S3 credentials для MLflow artifacts брать из `/etc/mlsystem/gpu-platform.env`, если они нужны;
- не запускать два train одновременно на одной GPU;
- перед запуском trial проверять `nvidia-smi`;
- перед запуском проверять, что repo clean, или явно понимать и описывать существующие изменения;
- если менялся код, обязательны tests + ruff;
- если trial упал, не перезапускать вслепую: сначала понять причину.

Базовая проверка контура:

```bash
ssh gpu-mlserver
cd /opt/mlsystem2/repo
source .venv/bin/activate

git status --short
git log -3 --oneline || true
test ! -d /data/MLSystem2 || echo "WARNING: /data/MLSystem2 exists, do not use it"
test ! -L /data/mlmarkup || echo "WARNING: /data/mlmarkup symlink exists, remove it"
test -d /data/MLMarkup
test -d /data/mlsystem2/prepared_images
nvidia-smi
```

Проверка Python/CUDA:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device", torch.cuda.get_device_name(0))
PY
```

## 4. Где искать предыдущие результаты

Сначала собери факты из уже проведенных запусков. Смотри:

- MLflow experiments:
  - `MLSystem2-First`;
  - `Segformer-b2-HPO-deforest-2605`;
  - `MLSystem2-Multiclass`;
  - любые эксперименты, указанные оператором;
- `docs/reports/*.md`;
- `/opt/hpo/report`, если там уже есть отчеты предыдущих сессий;
- runtime folders:
  - `/opt/mlsystem2/runtime/first_train`;
  - `/opt/mlsystem2/runtime/hpo`;
  - `/opt/mlsystem2/runtime/multiclass_b2_3h`;
- checkpoints and configs inside runtime folders.

Пример чтения списка экспериментов:

```bash
cd /opt/mlsystem2/repo
source .venv/bin/activate
python - <<'PY'
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
for exp in mlflow.search_experiments():
    print(exp.experiment_id, exp.name)
PY
```

Пример чтения последних runs:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = mlflow.get_experiment_by_name("Segformer-b2-HPO-deforest-2605")
runs = mlflow.search_runs(
    [exp.experiment_id],
    order_by=["start_time DESC"],
    max_results=20,
)
print(runs)
```

## 5. Как выбирать стратегию

Выбор стратегии должен быть фактическим, а не механическим:

- начинать с лучшего известного baseline;
- менять 1-2 фактора за trial, а не всю конфигурацию сразу;
- использовать staged search: короткий screening, promotion перспективных configs, confirmation лучших configs дольше;
- учитывать скорость эпохи как secondary objective;
- если метрика не растет в первые эпохи, prune;
- если loss NaN, non-finite gradients или OOM, остановить trial и исправить причину или изменить параметры;
- если модель показывает устойчивый рост, дать ей доучиться.

Для binary segmentation:

- primary metric: `val/best_threshold_pixel_f1` или `val/pixel_f1`, как указано оператором;
- смотреть precision, recall и threshold sweep;
- смотреть `train/loss`, `train/loss_focal`, `train/loss_tversky`;
- смотреть `train/skipped_optimizer_steps`;
- fixed threshold может быть хуже best threshold, поэтому не prune только по fixed-threshold F1, если sweep растет.

Для multiclass segmentation:

- primary metric: `val/macro_f1` или `val/mean_iou`;
- смотреть per-class f1/iou/support;
- учитывать class imbalance;
- проверять, что редкие классы реально попадают в train batches;
- не считать высокий pixel accuracy достаточным качеством, если macro metrics слабые.

## 6. Trial lifecycle

Каждый trial проходит одинаковый цикл:

1. Собрать факты из предыдущих runs и отчетов.
2. Сформировать гипотезу: какой фактор меняется и почему.
3. Создать trial config YAML в runtime workspace.
4. Сделать preflight:
   - settings load;
   - dataset_preparing;
   - DataLoader first batch;
   - model create.
5. Запустить train:

```bash
python -m mlsystem2.cli.train --config <config>
```

6. Мониторить первые минуты:
   - tail log;
   - `nvidia-smi`;
   - MLflow run exists;
   - epoch metrics появляются.
7. Читать epoch metrics из MLflow.
8. Принять решение: continue, prune, promote, failed или finished.
9. Записать trial summary в `trials.jsonl`.
10. Сразу перейти к следующему trial, если STOP-файл не появился.

Trial config должен храниться рядом с отчетами или runtime workspace так, чтобы позже можно было восстановить запуск.

## 7. Pruning

Pruning должен быть адаптивным:

- сравнивать trial с текущим best baseline;
- если trial сильно ниже baseline после нескольких эпох и динамика плохая, prune;
- если trial медленный и качество низкое, prune;
- если precision/recall деградируют и best threshold не помогает, prune;
- если `skipped_optimizer_steps > 0`, сначала исследовать причину;
- если OOM, снизить batch size или изменить параметры, а не повторять тот же запуск;
- если train loss падает, а целевая метрика медленно растет, можно продолжать.

Стартовые ориентиры для binary deforestation:

- если F1 не приблизился к предыдущему baseline после 5-10 эпох, prune;
- если train loss падает и `val/best_threshold_pixel_f1` растет, продолжать;
- если `val/best_threshold_pixel_f1` растет, даже при слабом `val/pixel_f1` на fixed threshold, продолжать;
- если best threshold постоянно уходит в крайние значения, смотреть precision/recall и распределение вероятностей.

Prune не должен оставлять zombie processes. Заверши только процесс текущего trial и не трогай unrelated training processes.

## 8. Reporting

В report workspace веди:

- `session_state.json`;
- `trials.jsonl`;
- `best_trials.md`;
- `current_strategy.md`;
- `errors.md`;
- `changes.md`.

Поля `trials.jsonl`:

```json
{
  "trial_id": 0,
  "model_name": "",
  "dataset": "",
  "experiment_name": "",
  "run_id": "",
  "run_name": "",
  "status": "",
  "config_path": "",
  "started_at": "",
  "finished_at": "",
  "duration_sec": 0,
  "best_metric": 0,
  "best_epoch": 0,
  "best_threshold": null,
  "epoch_time_sec_mean": 0,
  "params": {},
  "decision": "continue|pruned|promoted|failed|finished",
  "reason": ""
}
```

В `docs/reports/` пиши только важные итоговые отчеты, а не отчет после каждого trial. Отчет в репозитории должен содержать итог, ссылку на MLflow experiment, лучшие configs, качество, скорость, проблемы и следующие шаги.

## 9. Code change policy

Оркестратор может менять код только если:

- обнаружена явная ошибка;
- нужна минимальная HPO-фича;
- без изменения нельзя проверить гипотезу.

Правила изменения кода:

1. Сначала описать проблему в `changes.md`.
2. Внести минимальный change.
3. Обновить docs, если публичный контракт изменился.
4. Прогнать:

```bash
python -m pytest tests/test_public_contracts.py -q
python -m pytest tests -q
python -m ruff check src tests
```

5. Сделать commit с английским сообщением.
6. Сделать push, если доступен.
7. Дождаться deploy в `/opt/mlsystem2/repo` или аккуратно развернуть вручную, если CI/CD сломан.
8. Обязательно записать ручной deploy или обход CI/CD в отчет.

Не меняй архитектурные контракты молча. Если меняется публичное поведение settings, dataset preparing, tile preparation, train, models, MLflow adapter или train pipeline, обнови соответствующие docs.

## 10. DeepLabV3+ HPO guide

Следующая HPO-сессия для DeepLabV3+ должна стартовать с этой модели:

```yaml
model_name: smp_deeplabv3plus_resnet50
```

Модель должна быть реализована в MLSystem2 через SMP DeepLabV3Plus ResNet50.

Для deforestation binary используй:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/
  scenes_file: /data/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/MLMarkup/Вырубки/deforestation.geojson

train:
  task: binary
  model_name: smp_deeplabv3plus_resnet50
  input_channels: 4
  output_channels: 1
```

Стартовые принципы:

- `smart_tiling: true`;
- `positive_factor` использовать как параметр HPO;
- `augmentation_level` использовать как параметр HPO;
- loss candidates: `focal_tversky`, `bce_dice`, `focal_dice`;
- LR candidates брать вокруг предыдущих рабочих SegFormer values, но корректировать, если DeepLab сходится иначе;
- batch size начать с `4`;
- если OOM, использовать `2`;
- tile size `1024`, stride `512` initially;
- primary metric: `val/best_threshold_pixel_f1`;
- speed secondary objective.

Не фиксируй заранее trial numbers или run ids. Сначала посмотри предыдущие MLflow результаты и отчеты, затем сформируй стартовую стратегию.

## 11. Stop/resume

Останавливай сессию, если существует:

```text
/opt/hpo/report/STOP
```

или session-specific STOP file в report workspace.

Перед каждым новым trial проверяй STOP. Если STOP появился во время trial, корректно заверши текущий train или дождись ближайшей безопасной точки, если качество перспективное и оператор не требовал немедленной остановки.

Resume:

- прочитай `session_state.json`;
- прочитай `trials.jsonl`;
- проверь, нет ли живого train процесса этой сессии;
- восстанови последний trial index;
- продолжай с новой гипотезы на основе уже записанных результатов.

Никогда не убивай unrelated training processes.

## 12. Если ты Codex-оркестратор

Начни с:

1. Подключись к GPU-серверу.
2. Проверь repo, env, GPU и MLflow.
3. Прочитай предыдущие HPO отчеты и MLflow runs.
4. Сформируй стартовую стратегию.
5. Запиши `current_strategy.md`.
6. Запусти первый trial.
7. Мониторь, prune/promote, повторяй.

Не создавай отдельный HPO runner script, если оператор прямо запретил. Оркестрируй действиями Codex CLI.

## 13. Пример стартового prompt для DeepLab HPO

```text
Ты работаешь как HPO-оркестратор.

Подключись к GPU-серверу `gpu-mlserver`.
Основной репозиторий на GPU-сервере: `/opt/mlsystem2/repo`.
Прочитай инструкцию: `/opt/mlsystem2/repo/docs/hpo_instruction.md`.

Запусти HPO-сессию для:
- model_name: smp_deeplabv3plus_resnet50
- dataset/class: deforestation / Вырубки
- MLflow experiment: Deeplabv3plus-resnet50-HPO-deforest-2605
- primary metric: val/best_threshold_pixel_f1
- secondary objective: training speed
- report workspace: /opt/hpo/report/deeplabv3plus_deforest_2605

Не создавай HPO runner script.
Сам выбирай стратегию и тактику по инструкции.
Смотри прошлые MLflow runs и отчеты.
Запускай trial за trial.
Плохие trials останавливай досрочно.
Если нужна минимальная HPO-фича или исправление ошибки - меняй код, тестируй, коммить, пушь.
Пиши отчеты в /opt/hpo/report/deeplabv3plus_deforest_2605.
```
