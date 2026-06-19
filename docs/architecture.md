# Архитектура

MLSystem 2.0 — простая система командной строки с явными границами модулей. 
Снимки для обучения и инференса берутся из локальной папки заранее подготовленных изображений.
Основной принцип реализации системы и построения архитектуры - необходимо и достаточно. Это касается документов, интефейсов, модулей, функционала и всего остального что есть в проекте.

## Ахритекутурные правила обязательные к соблюдению зафиксированы тут: 
- Данный документ - главный источник правды для всего проекта. 
- Архитекутурные правила в архитектурных документах должны абсолютно соблюдаться
- Любые изменения в архитектурных документах - согласуются.
- В репозитории не должно быть функционала повторяющиего функционал модулей.
- Недопустимо оставлять deprecated wrapper в репозитории.
- Новые модули и код не принадлежащий к какому либо модулю создавать запрещено без отдельного согласования

## Архитекутурные документы
- Этот файл
- Правила оформление модулей в docs\module_rules.md
- Описание модулей в папке docs\modules



Основное приложение обучения:

```bash
mlsystem2-train --settings configs/settings.server.yaml --run configs/run.example.server.yaml
```

Основное приложение инференса:

```bash
mlsystem2-infer --config configs/example.server.yaml
```

Веб-интерфейс обучения:

```bash
mlsystem2-training-ui-api
python frontend/build.py
```

`training_ui_api` — отдельный модуль MLSystem2 и единственная серверная точка доступа frontend к
данным UI обучения. Frontend не обращается к Postgres напрямую. UI-данные обучения, шаблоны,
очереди, custom datasets и результаты хранятся в отдельной Postgres БД/схеме. Инфраструктура
Postgres разворачивается вручную по runbook и не входит в CI/CD. UI-сервис может читать публичные
данные MLflow для списка experiments, best checkpoint и скачивания checkpoint, но не открывает
training runs и не пишет MLflow-метрики; запись метрик выполняет `train_pipeline`.

Автоматизация обучения и псевдоразметки является частью `training_ui_api`. Она хранит правила
`датасет × модель` в Postgres, отслеживает версию конкретного варианта MLMarkup по git-коммиту
папки варианта или filesystem mtime fallback и создает auto jobs только через те же очереди, что
и ручной запуск. Jobs диспетчеризуются общей очередью: ручная псевдоразметка выше ручного обучения,
ручное обучение выше auto псевдоразметки, auto псевдоразметка выше auto обучения. Auto jobs отменяются
изменением правила автоматизации или заменяются при новой версии конкретного датасета; frontend не
имеет прямого доступа к БД и не запускает процессы напрямую.
Глобальное отключение автоматизации отменяет все active auto jobs, останавливает running automatic
process и помечает известный MLflow run как `KILLED`; повторное включение создает jobs заново по
текущим правилам и версиям датасетов.
Шаблоны обучения в `training_ui_api` бывают базовыми для сети и привязанными к конкретному варианту
датасета. При создании job сервис сначала ищет шаблон `(architecture, dataset_key)`, затем использует
базовый `(architecture, null)`, поэтому frontend не дублирует эту бизнес-логику.
Псевдоразметка Training UI запускается отдельным процессом с backend `pytorch_one_off`: worker пишет
`pseudo_config.yaml`, runner загружает checkpoint через `models.load_checkpoint`, выполняет локальный
PyTorch-инференс и после обработки освобождает CUDA cache. Этот путь не экспортирует модель в Triton, не
создает Geoalert pipeline YAML и не добавляет запись в Triton model repository; Triton остается ручным
production-инференсом и явным экспортом.
Страница экспорта модели в `training_ui_api` принимает локальный MLSystem2 checkpoint `.pt`, берет threshold и
sample_size из metadata checkpoint, собирает временный zip-архив для `models-serving-service` и Triton CPU,
отдает его пользователю и не пишет данные в Postgres, MLflow, S3 или рабочий каталог сервиса инференса.
На странице результатов обучения тот же временный zip-экспорт доступен для успешного training result: сервис
скачивает `checkpoints/best.pt` из MLflow по сохраненному run id результата и передает checkpoint в общий сборщик
экспорта.

Каноническое место хранения результатов работы — MLflow. Локальные директории используются только
как временная рабочая область: в них могут лежать временные файлы, кэши, журналы и промежуточные
отчеты до записи в артефакты MLflow.

## Конвейер Обучения

1. CLI получает стабильный `settings.yml` через `--settings` и задание конкретного обучения через `--run`, вызывает `settings.api.load_settings(settings_path, run_path)` и инициализирует текущие настройки процесса. Совместимый legacy-режим `--config` остается для старых полных YAML.
2. Создать или открыть запуск MLflow через `mlflow_adapter` и записать YAML задания запуска в артефакты.
3. `dataset_preparing` принимает локальные пути, проверяет наличие подготовленных снимков в `dataset.images_dir`, готовит датасет и возвращает общий VRT XML найденного пула снимков. Поддерживаются два режима датасета: binary через `dataset.scenes_file` + `dataset.annotation_file` + optional `dataset.hard_negative_annotation_file` и multiclass через `dataset.classes` с optional class-level hard negative GeoJSON. Train/val split всегда выполняется по тайлам; scene-based отбор negative-снимков не используется.
4. Если `dataset_preparing` вернул ошибки, `train_pipeline` записывает отчет подготовки в MLflow и
   завершает конвейер с ошибкой.
5. После успешной подготовки `train_pipeline` сохраняет исходные txt/geojson файлы датасета в MLflow artifacts `dataset/`.
6. `train_pipeline` вызывает `tile_preparation.create_tile_dataloader` отдельно для train и val. Оба loader получают общий VRT и одинаковый `tile_split`, а `tile_preparation` детерминированно делит список окон на непересекающиеся train/val subsets. Train loader читает raster и rasterize лениво в `Dataset.__getitem__`, классифицирует tiles как positive, hard_negative или background, использует weighted sampling с общим marked-бюджетом `positive_factor + hard_negative_factor` и отдельным `background_factor`, а также применяет `tile_preparation.prefetch_epochs` как расчетный PyTorch `prefetch_factor`; если задан `train.max_train_batches_per_epoch`, расчет prefetch использует ограниченную длину train-эпохи. Val loader выбирает фиксированный balanced subset без replacement по positive/non-positive hints, один раз собирает CPU batch tensors в RAM cache и переиспользует их на каждой эпохе; train factors и `prefetch_epochs` к val не применяются. Image tensors уже соответствуют Geoalert ABI: raw `float32`, `C,H,W` на sample и `B,C,H,W` в batch. Hard negative не является выходным классом и остается background `0`. В binary режиме mask имеет `torch.float32 [B,1,H,W]`, в multiclass режиме mask имеет `torch.long [B,H,W]`, где `0` - background, `1..N` - class id по порядку `dataset.classes`.
7. `train_pipeline` создает поддерживаемую segmentation-модель (`segformer_b0`, `segformer_b2`, диагностический SMP-совместимый `smp_segformer_b0`/`smp_segformer_b2`/`smp_segformer_b3` или `smp_deeplabv3plus_resnet50`) через `models.create_model` или загружает checkpoint через `models.load_checkpoint`, если `train.initial_checkpoint_uri` задан. Для multiclass `train.output_channels` должен быть равен `len(dataset.classes) + 1`.
8. `train` выполняет PyTorch обучение segmentation-модели: AdamW, cosine scheduler, binary BCE/Dice-family loss или multiclass cross entropy, validation metrics, early stopping и best/final checkpoints. Background имеет фиксированный вес `1`, `pos_weight` усиливает positive pixels в binary loss, а `hard_negative_weight` усиливает штраф за false positive на hard-negative тайлах в binary и multiclass train; hard negative при этом остается target background `0`. Checkpoint metadata содержит validation threshold и `sample_size`, необходимый для экспорта в Triton.
9. `train_pipeline` передает в `train` progress sink, который пишет метрики каждой завершенной эпохи в MLflow сразу через `mlflow_adapter.log_training_epoch`.
10. `mlflow_adapter` записывает итоговые train/val метрики, артефакты, модель или чекпойнт, отчет tile preparation, отчет времени и итоговый
   отчет.

## Конвейер Инференса

1. CLI получает путь `--config`, вызывает `settings.api.load_settings` и инициализирует текущие настройки процесса.
2. Загрузить модель или чекпойнт.
3. Выполнить инференс напрямую в Python/PyTorch по локальной папке `dataset.images_dir`.
4. Записать результаты и отчеты в MLflow.
