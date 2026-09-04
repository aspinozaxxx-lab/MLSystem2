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
- `TrainSettings` — task/metric/model/channels/checkpoint, `pipeline_variant=legacy|next_gen`, epochs/batch/device, optimizer/loss параметры, threshold, patience и optional batch/time limits.
- `NextGenSettings` — validation fold, normalization, validation interval, threshold mode и optional Gaussian A/B.
- `InferenceSettings` — `checkpoint_uri`, `threshold`, `batch_size`, `device`.
- `MLflowSettings` — `enabled`, `tracking_uri`, `experiment_name`.
- `SystemSettings` — `runtime`, `dataset`, `tile_preparation`, `train`, `next_gen`, `inference`, `mlflow`.

## Список используемых данным модулем модулей и с какой целью

Публичные API других модулей не используются. YAML читает `PyYAML`, DTO валидирует Pydantic.

## Алгоритм работы и его особенности

`load_settings` проверяет файлы, рекурсивно накладывает `run.yml` на стабильный `settings.yml`, запрещает лишние поля и сохраняет результат. Отсутствующий `train.pipeline_variant` означает `legacy`; старые YAML остаются совместимыми. `next_gen` v1 разрешён только для binary, четырёх каналов, одного выхода и `segformer_b0|smp_segformer_b0`; ненулевой `max_val_batches_per_epoch` отклоняется. Pretrained разрешён только HF B0. Gaussian A/B дополнительно требует tile `512` и stride `256`. Остальные проверки и поведение `legacy` не изменены.

Пример per-image binary:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/kanopus
  annotations_dir: /data/MLMarkup/Реки/test
  val_fraction: 0.2
```
