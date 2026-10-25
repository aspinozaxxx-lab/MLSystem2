# Модуль settings

## Назначение

`settings` загружает YAML и является единой точкой типизации и валидации настроек процесса.

## Публичный интерфейс

- `load_settings(path: str | Path, run_path: str | Path | None = None) -> SystemSettings` — прочитать основной YAML и optional overlay запуска, сохранить текущие настройки.
- `get_settings() -> SystemSettings` — вернуть загруженные настройки либо бросить `SettingsError`.
- `get_settings_path() -> Path` — вернуть путь активного YAML либо бросить `SettingsError`.

## Публичные контракты

- `SettingsError` — ошибка чтения или валидации.
- `RuntimeSettings` — `project_root`, `scratch_root`, `logs_root`, `cleanup_scratch_after_mlflow_log`.
- `DatasetClassSettings` — `slug`, `name`, `scenes_file`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `DatasetSettings` — `images_dir`, optional legacy-поля `scenes_file`, `annotation_file`, `hard_negative_annotation_file`, optional `annotations_dir`, `classes`, `val_fraction`; свойство `is_multiclass`.
- `TilePreparationSettings` — `tile_size`, `stride`, `num_workers`, `prefetch_epochs`, `seed`, `augmentation_level`, три sampling factor, `val_positive_factor`, `class_balance`.
- `TrainSettings` — task/metric/model/channels/checkpoint, `pipeline_variant=legacy|next_gen|next_gen2|next_gen3|next_gen4|next_gen4_global_norm|next_gen4_augmentation_v3|next_gen4_augmentation_v4|next_gen4_consistency_v1|next_gen4_consistency_v2|next_gen4_consistency_v3|next_gen4_consistency_v4|object_f1|object_f1_v2`, epochs/batch/device, optimizer/loss параметры, threshold, patience и optional batch/time limits. Next-gen3 наследует ограничения архитектур/тайлов next-gen2; автоматическая политика фона версионируется в сэмплере, новых ручных коэффициентов нет.
- `NextGenSettings` — validation fold, normalization, validation interval, threshold mode и optional Gaussian A/B.
- `InferenceSettings` — `checkpoint_uri`, `threshold`, `batch_size`, `device`.
- `MLflowSettings` — `enabled`, `tracking_uri`, `experiment_name`.
- `SystemSettings` — `runtime`, `dataset`, `tile_preparation`, `train`, `next_gen`, `inference`, `mlflow`.

Next-gen4 наследует фиксированный профиль и автоматический фон next-gen3, но требует stride=tile_size, context=0. Размер 512/768/1024/1536 выбирается пользователем; ненулевые лимиты train/validation batch не допускаются. Старые варианты сохраняют собственные ограничения.

`next_gen4_global_norm` имеет те же ограничения и отдельное имя варианта. Способ нормализации фиксирован профилем: границы по train вычисляет `train_pipeline`, сохраняет модель; поля ручных границ в settings не добавляются.

`next_gen4_augmentation_v3` наследует эти ограничения и общую train-нормализацию; защита valid-mask аугментаций фиксирована профилем и не добавляет ручных параметров. Старые варианты не переключаются.

`next_gen4_augmentation_v4` — дополнительное значение `TrainSettings.pipeline_variant` с теми же ограничениями. Защищённые квадратные вырезки внутри train-окна фиксированы профилем; отдельных пользовательских коэффициентов нет.

`TrainSettings.pipeline_variant` также принимает `next_gen4_consistency_v1`, наследующий
фиксированные параметры v4. Optional `diagnostic_checkpoint_interval_sec` задаёт период
сохранения текущих диагностических весов на границе эпох; по умолчанию `None`, прежние
профили ничего дополнительно не сохраняют. Это не настройка точного продолжения обучения.

`next_gen4_consistency_v2` наследует эти параметры и допускает только SMP SegFormer
B0/B1/B2/B3; способ нормализации декодера фиксирован версией, ручного переключателя нет.

`next_gen4_consistency_v3` — отдельное допустимое значение `TrainSettings.pipeline_variant`
с теми же ограничениями SMP B0/B1/B2/B3, размеров, seed и optimizer, что у v2.
Параметры двунаправленного штрафа фиксированы профилем, новых ручных настроек нет.

`next_gen4_consistency_v4` — отдельное значение `TrainSettings.pipeline_variant`,
наследующее фиксированные настройки SMP-профиля v3. Надзор неохваченных пикселей
по train-эталону задан версией; новых ручных коэффициентов нет.

## Список используемых данным модулем модулей и с какой целью

Публичные API других модулей не используются. YAML читает `PyYAML`, DTO валидирует Pydantic.

## Алгоритм работы и его особенности

`object_f1_v2` — самостоятельный бинарный профиль SMP B0/B1/B2/B3 с объектовой метрикой и прежними ограничениями размеров/эпох объектного профиля. Отсутствующий вариант по-прежнему означает legacy; старые варианты не переводятся в v2 автоматически.

`train.pipeline_variant` дополнен object_f1: binary SegFormer B0/B1/B2/B3, quality_metric=objects, фиксированные параметры next-gen2 и отдельная ветвь подготовки/оценки границ. Batch size для тайлов 512/768/1024/1536: B0 — 16/8/4/2, B1 — 8/4/2/1, B2 и B3 — 4/2/1/1. Размер тайла, pretrained и условия остановки изменяемы.

`next_gen2` требует профиль ноутбука: binary или multiclass, HF B0 или SMP SegFormer B0/B1/B2/B3, 3 или 4 входных канала,
один логический выход для binary либо N+1 по схеме для multiclass,
тайлы 512/768/1024/1536, шаг в половину тайла, context 0, augmentation_level 3, seed 42,
batch по архитектуре: B0 — 16/8/4/2, B1 — 8/4/2/1, B2/B3 — 4/2/1/1 соответственно, LR 1e-4,
weight decay 0.01, loss `cross_entropy_tversky`, alpha/beta 0.75/0.25 и threshold 0.5.
В UI меняются pretrained, tile_size, epochs, early_stopping_patience и max_training_time_sec. Batch limits запрещены.
Для продолжения завершённого обучения разрешён `train.initial_checkpoint_uri` также в next-gen2 и object_f1.
Фиксированные параметры профиля сохраняются; `train_pipeline` проверяет архитектуру, число каналов,
вариант конвейера и нормализацию загружаемого checkpoint перед обучением. HTTP-диалог выбирает best/last
исходного результата, прямой путь к файлу через этот диалог не принимается.
Число workers — параметр исполнения из серверного settings.yml (8), а не ограничение профиля ноутбука.

`load_settings` проверяет файлы, рекурсивно накладывает `run.yml` на стабильный `settings.yml`, запрещает лишние поля и сохраняет результат. Отсутствующий `train.pipeline_variant` означает `legacy`; старые YAML остаются совместимыми. `next_gen` v1 разрешён только для binary, четырёх каналов, одного выхода и `segformer_b0|smp_segformer_b0`; ненулевой `max_val_batches_per_epoch` отклоняется. Pretrained разрешён только HF B0. Gaussian A/B дополнительно требует tile `512` и stride `256`. Остальные проверки и поведение `legacy` не изменены.

Пример per-image binary:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/kanopus
  annotations_dir: /data/MLMarkup/Реки/test
  val_fraction: 0.2
```
