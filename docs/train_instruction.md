# Инструкция для обучения модели

Этот документ нужен Codex-агенту, который запускает обычное обучение MLSystem2 на заданном датасете и с заданными параметрами. Цель - не исследовать систему заново при каждом старте, а быстро проверить входные данные, собрать конфиг, запустить обучение, проконтролировать MLflow и зафиксировать результат.

Документ не заменяет архитектуру. Перед изменением кода соблюдай `docs/architecture.md` и модульные документы из `docs/modules/`.

## 1. Что должен указать оператор

Перед запуском обучения должны быть известны:

- GPU-сервер: обычно `gpu-mlserver`.
- Репозиторий на сервере: обычно `/opt/mlsystem2/repo`.
- Датасет: двоичный или многоклассовый.
- Пути к снимкам и разметке.
- `model_name`.
- Основные параметры обучения: размер тайла, шаг тайла, размер батча, эпохи, learning rate, loss и threshold.
- Нужно ли стартовать с существующего чекпойнта.
- Имя эксперимента MLflow.
- Рабочая runtime-папка для scratch и logs.

Если оператор дал неполный набор параметров, сначала возьми ближайший успешный конфиг из отчетов и MLflow, затем явно запиши, какие значения были приняты по умолчанию.

## 2. Канонические пути

Основные пути на GPU-сервере:

```text
/opt/mlsystem2/repo
/data/mlsystem2/prepared_images/
/data/MLMarkup
/opt/mlsystem2/runtime
/opt/mlsystem2/runtime/hpo
/opt/hpo/report
```

Не используй `/data/MLSystem2`. Не создавай `/data/mlmarkup` и symlink на него. Источник истины для разметки - `/data/MLMarkup`.

Локальные примеры конфигов в репозитории:

```text
configs/settings.server.yaml
configs/run.example.server.yaml
```

Основной CLI по архитектуре:

```bash
mlsystem2-train --settings <settings.yml> --run <run.yml>
```

Допустимый эквивалент при работе из venv:

```bash
python -m mlsystem2.cli.train --settings <settings.yml> --run <run.yml>
```

Legacy-режим `--config <config>` оставлен для старых полных YAML, но новый штатный запуск использует два файла:
`settings.yml` с параметрами приложения и `run.yml` с заданием конкретного обучения.

## 3. Где искать гиперпараметры лучших чекпойнтов

Сначала ищи не сам файл `best.pt`, а связку: чекпойнт, MLflow run id, параметры запуска, метрики и решение оркестратора.

Главные источники:

- `/opt/hpo/report/<session>/best_trials.md` - короткая таблица лучших trials, параметры, run id, лучший F1, threshold, epoch и формальный лучший чекпойнт.
- `/opt/hpo/report/<session>/trials_journal.md` - полный список перебранных сочетаний параметров и решений.
- `/opt/hpo/report/<session>/session_state.json` - состояние HPO-сессии и текущий champion.
- `/opt/hpo/report/<session>/current_strategy.md` - логика выбора следующих запусков.
- `docs/reports/*.md` - итоговые отчеты, которые уже попали в репозиторий.
- MLflow `http://127.0.0.1:5000` - параметры run, история метрик по эпохам, артефакты config/report.
- Чекпойнты в runtime: `/opt/mlsystem2/runtime/hpo/<session>/scratch/trial_<NNNN>/checkpoints/best.pt`.

Текущие важные ориентиры:

```text
SegFormer B2, вырубки, tiles 512/768:
  отчет: /opt/hpo/report/segformer_b2_tiles_512_768_2805/best_trials.md
  лучший trial: 0050
  run id: 6b1d18d8ac1249088b8577d75777a5a2
  чекпойнт: /opt/mlsystem2/runtime/hpo/segformer_b2_tiles_512_768_2805/scratch/trial_0050/checkpoints/best.pt
  параметры: smp_segformer_b2, tile 512, stride 256, batch 8, lr 1.5e-7, focal_tversky, tversky 0.4/0.6, focal_alpha 0.6, augmentation_level 2, positive_factor 0.9, weight_decay 1e-4.

DeepLabV3+ ResNet50, вырубки:
  отчет: /opt/hpo/report/deeplab_v3_2705/best_trials.md
  лучший trial: 0015
  run id: f1b795dcdacc4ef1a24d95d0afea5c19
  чекпойнт: /opt/mlsystem2/runtime/hpo/deeplabv3plus_resnet50_deforest_2705/scratch/trial_0015/checkpoints/best.pt
  параметры: smp_deeplabv3plus_resnet50, tile 768, stride 384, batch 8, дообучение от trial 0014, lr 2e-6, focal_tversky, threshold 0.7, positive_factor 0.8, weight_decay 1e-4.

Старый SegFormer B2 HPO от 26.05:
  отчет: docs/reports/segformer_b2_hpo_deforest_2605.md
  лучший trial: 0008
  run id: 59b45400260c4e4da5d6f753244339b1
  чекпойнт: /opt/mlsystem2/runtime/hpo/segformer_b2_deforest_2605/scratch/trial_0008/checkpoints/best.pt
```

Если значения в отчете и текущих MLflow metrics выглядят разными, смотри историю метрик по эпохам, а не только последние значения `run.data.metrics`: MLflow хранит последние значения метрик отдельно от лучшей эпохи.

Минимальный способ прочитать MLflow:

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

Параметры и история метрик конкретного run:

```bash
cd /opt/mlsystem2/repo
source .venv/bin/activate
python - <<'PY'
import mlflow

run_id = "<run_id>"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

print("status:", run.info.status)
print("run_name:", run.data.tags.get("mlflow.runName"))
print("params:")
for key, value in sorted(run.data.params.items()):
    print(key, "=", value)

metric = "val/best_threshold_pixel_f1"
history = client.get_metric_history(run_id, metric)
best = max(history, key=lambda item: item.value)
print("best", metric, best.value, "epoch", best.step)
PY
```

Если нужно читать MLflow artifacts из S3/MinIO, сначала подгрузи окружение:

```bash
set -a
source /etc/mlsystem/gpu-platform.env
set +a
```

## 4. Проверка перед обучением

Выполни на GPU-сервере:

```bash
ssh gpu-mlserver
cd /opt/mlsystem2/repo
source .venv/bin/activate

git status --short
test -d /data/MLMarkup
test -d /data/mlsystem2/prepared_images
nvidia-smi
```

Проверь, что на GPU нет чужого процесса обучения, который нельзя трогать:

```bash
ps -ef | grep -E "mlsystem2.cli.train|mlsystem2-train" | grep -v grep || true
```

Проверь CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY
```

## 5. Конфиг двоичного датасета

Поддерживаются два взаимоисключающих формата. В legacy-формате используй `dataset.scenes_file` и `dataset.annotation_file`. Если есть hard negative объекты, добавь `dataset.hard_negative_annotation_file`.
Hard-negative GeoJSON содержит области, которые модель должна считать фоном: внутри tile supervision mask они
получают служебное значение `-1`, перед loss превращаются в target background `0` и получают pixel weight из
произведения `train.background_weight × train.hard_negative_weight`. Это не отдельный выходной класс модели.
Nodata по значению, невалидные пиксели `dataset_mask` и padding за границей TIFF получают target background `0`.
Ложный прогноз целевого класса на них увеличивает loss и учитывается validation-метриками как false positive.

Пример для вырубок:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/
  scenes_file: /data/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/MLMarkup/Вырубки/deforestation.geojson
  hard_negative_annotation_file: /data/MLMarkup/Вырубки/hard_negative.geojson
  val_fraction: 0.2

train:
  model_name: smp_segformer_b2
```

В per-image формате укажи только каталог `dataset.annotations_dir`. В нём каждому TIFF соответствует один
`FeatureCollection` в CRS снимка с именем `<родительская_папка>_<имя_TIFF_без_расширения>.geojson`.
Свойство `_mlsystem2_role` равно `positive` или `hard_negative`; отсутствующее свойство совместимо с
`positive`. Снимки берутся из `images_dir`, сопоставляются строго по имени, а каждый снимок нарезается
независимо.

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/
  annotations_dir: /data/MLMarkup/Реки/test
  val_fraction: 0.2

train:
  model_name: smp_segformer_b2
```

Не смешивай `annotations_dir` с legacy-полями или `dataset.classes`. Пустой per-image датасет можно наполнять
в редакторе, но запуск обучения отклонит его до добавления хотя бы одного снимка.

Для дообучения укажи:

```yaml
train:
  initial_checkpoint_uri: /opt/mlsystem2/runtime/hpo/<session>/scratch/trial_<NNNN>/checkpoints/best.pt
```

## 6. Конфиг многоклассового датасета

Для legacy multiclass используй `dataset.classes`. Для нового per-image multiclass укажи только `dataset.annotations_dir`; schema читается из `.mlsystem2-dataset.json` и строго сверяется со всеми GeoJSON.

Принципы:

- `train.task: multiclass`.
- `train.loss: cross_entropy` или `cross_entropy_dice`.
- `train.output_channels = N + 1`, где `N` берётся из schema dataset.
- `background = 0`, class ID приходят из YAML для legacy или из manifest для per-image.
- `priority` влияет на перекрытия: больший приоритет перекрывает меньший.

Training UI автоматически выбирает multiclass task, три выхода для двух типов, class balance и показывает только `cross_entropy|cross_entropy_dice`.

## 7. Параметры, которые обычно задаются оператором

Основные поля:

```yaml
tile_preparation:
  tile_size: 512
  stride: 256
  augmentation_level: 2
  positive_factor: 0.8
  hard_negative_factor: 0.0
  background_factor: 0.2

train:
  epochs: 30
  batch_size: 4
  learning_rate: 0.00001
  weight_decay: 0.0001
  loss: focal_tversky
  focal_alpha: 0.6
  pos_weight: 1.0
  background_weight: 1.0
  hard_negative_weight: 1.0
  tversky_alpha: 0.4
  tversky_beta: 0.6
  threshold: 0.8
  early_stopping_patience: 10
  max_train_batches_per_epoch: 72
  max_val_batches_per_epoch: 1000
  max_training_time_sec: 1800
```

`tile_preparation.hard_negative_factor` управляет частотой hard-negative tiles в train sampler.
`train.background_weight` задаёт базовый вес всего фона, а `train.hard_negative_weight` дополнительно умножает
штраф loss на pixels внутри hard-negative геометрии.

Воркеры, prefetch, seed, device, binary task и каналы модели задаются defaults модулей.
`max_train_batches_per_epoch`, `max_val_batches_per_epoch` и `max_training_time_sec` остаются параметрами запуска,
потому что они управляют длительностью конкретного обучения. Лимит val также ограничивает число тайлов,
выбираемых до оценки и построения RAM cache.

## 8. Создание runtime-конфига

Не меняй `configs/settings.server.yaml` ради одного запуска. Это параметры приложения: их меняет администратор
при настройке сервера. Для конкретного обучения создай отдельный `run.yml` в runtime-папке:

```bash
RUN_ROOT=/opt/mlsystem2/runtime/train/<name>_<DDMM>
mkdir -p "$RUN_ROOT"/{scratch,logs,configs}
cp configs/run.example.server.yaml "$RUN_ROOT/configs/run.yml"
```

В `run.yml` укажи runtime текущего запуска:

```yaml
runtime:
  project_root: /opt/mlsystem2/repo
  scratch_root: /opt/mlsystem2/runtime/train/<name>_<DDMM>/scratch
  logs_root: /opt/mlsystem2/runtime/train/<name>_<DDMM>/logs
  cleanup_scratch_after_mlflow_log: false
```

Для длительных и важных запусков лучше `cleanup_scratch_after_mlflow_log: false`, чтобы локальный `best.pt` остался доступен даже при проблемах с MLflow artifacts.

Размер эпохи задается в `run.yml`, потому что это параметр конкретного запуска:

```yaml
train:
  max_train_batches_per_epoch: 72
  max_val_batches_per_epoch: 1000
```

Пустое значение (`null`) означает полный train или полный balanced validation subset и может резко увеличить
длительность эпохи и оценочный объём val cache. Небольшой val кэшируется в RAM, а при превышении 50% доступной
памяти либо невозможности определить её объём читается лениво по тем же фиксированным индексам. Для крупных тайлов
`768 × 768` разумно начинать с `max_val_batches_per_epoch: 256` и увеличивать лимит только после проверки отчёта.

Проверка загрузки конфига:

```bash
python - <<'PY'
from mlsystem2.settings.api import load_settings
settings = load_settings("configs/settings.server.yaml", "<run.yml>")
print(settings.train.model_name)
print(settings.dataset.images_dir)
print(settings.train.max_train_batches_per_epoch)
PY
```

## 9. Запуск

Запускай один процесс обучения на одну GPU:

```bash
cd /opt/mlsystem2/repo
source .venv/bin/activate
SETTINGS=/opt/mlsystem2/repo/configs/settings.server.yaml
RUN=/opt/mlsystem2/runtime/train/<name>_<DDMM>/configs/run.yml
LOG=/opt/mlsystem2/runtime/train/<name>_<DDMM>/logs/train.log

nohup mlsystem2-train --settings "$SETTINGS" --run "$RUN" > "$LOG" 2>&1 &
echo $! > /opt/mlsystem2/runtime/train/<name>_<DDMM>/train.pid
```

Мониторинг:

```bash
tail -f "$LOG"
nvidia-smi
```

Параллельно проверь, что run появился в MLflow experiment и что пишутся epoch metrics.

## 10. Что считать результатом обучения

После завершения зафиксируй:

- путь к config;
- MLflow experiment и run id;
- статус run;
- путь к `best.pt` и `final.pt`, если есть;
- лучшую метрику и эпоху;
- threshold из sweep;
- основные гиперпараметры;
- длительность обучения;
- ошибки или отклонения.

Основная метрика обучения и выбора checkpoint - `val/best_threshold_pixel_f1`. Для HPO-сравнения завершенных запусков используй итоговую `train/best_threshold_pixel_f1`. Также смотри `val/best_threshold_precision`, `val/best_threshold_recall`, `val/best_threshold`, `train/loss`, `val/loss`, `train/epoch_time_sec` и итоговое `train/training_time_sec`.

## 11. Отчетность

Каноническое место результата - MLflow. Локальный runtime нужен для логов, scratch и восстановления запуска.

Если запуск важный, добавь короткий отчет в `docs/reports/` или обнови существующий итоговый отчет. Не пиши новый отчет после каждого технического запуска.

В отчете должны быть:

- задача и датасет;
- путь к config;
- MLflow experiment и run id;
- путь к чекпойнту;
- параметры запуска;
- лучшие метрики;
- вывод: использовать, доучивать, отклонить или проверить дополнительно.

## 12. Когда можно менять код

Код меняй только если:

- найдена явная ошибка;
- без минимальной доработки нельзя запустить заданный датасет или параметр;
- нужно поддержать уже согласованный публичный контракт.

После изменения кода выполни:

```bash
python -m pytest tests/test_public_contracts.py -q
python -m pytest tests -q
python -m ruff check src tests
```

Если изменился публичный контракт settings, dataset preparing, tile preparation, train, models, MLflow adapter или train pipeline, обнови соответствующий документ в `docs/modules/`.

## 13. Быстрый шаблон запроса для Codex

```text
Подключись к `gpu-mlserver`.
Репозиторий: `/opt/mlsystem2/repo`.
Прочитай `docs/train_instruction.md`.

Запусти обучение:
- dataset: <двоичный или многоклассовый>
- images_dir: <путь>
- scenes_file / annotation_file или список classes: <пути>
- model_name: <модель>
- initial_checkpoint_uri: <чекпойнт или null>
- tile_size / stride: <значения>
- batch_size: <значение>
- epochs: <значение>
- learning_rate: <значение>
- loss: <значение>
- threshold: <значение>
- MLflow experiment: <имя>
- runtime: <путь>

Не создавай новый CLI и не меняй архитектуру.
Перед стартом проверь GPU, repo, пути датасета и MLflow.
После завершения дай run id, чекпойнт, лучшие метрики и путь к config.
```
