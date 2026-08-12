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
- `TrainSettings` — task/metric/model/channels/checkpoint, epochs/batch/device, optimizer/loss параметры, threshold, patience и optional batch/time limits.
- `InferenceSettings` — `checkpoint_uri`, `threshold`, `batch_size`, `device`.
- `MLflowSettings` — `enabled`, `tracking_uri`, `experiment_name`.
- `SystemSettings` — `runtime`, `dataset`, `tile_preparation`, `train`, `inference`, `mlflow`.

## Список используемых данным модулем модулей и с какой целью

Публичные API других модулей не используются. YAML читает `PyYAML`, DTO валидирует Pydantic.

## Алгоритм работы и его особенности

`load_settings` проверяет файлы, рекурсивно накладывает `run.yml` на стабильный `settings.yml`, запрещает лишние поля и сохраняет результат. Dataset задаёт ровно один режим: legacy binary (`scenes_file+annotation_file`, optional hard negative), per-image (`annotations_dir`) либо legacy multiclass (`classes`). Наличие `.mlsystem2-dataset.json` внутри `annotations_dir` автоматически определяет per-image multiclass; схема классов берётся из manifest, поэтому дублировать её в YAML не нужно. Binary требует `train.task=binary`, multiclass — `task=multiclass`, `cross_entropy|cross_entropy_dice` и `output_channels=N+1`. Проверяются размеры/stride, sampling factors с суммой `1`, диапазоны loss/threshold и лимиты. `prefetch_epochs` относится к train; val использует фиксированный balanced subset. `hard_negative_weight` влияет на loss, а `hard_negative_factor` — на sampler.

Пример per-image binary:

```yaml
dataset:
  images_dir: /data/mlsystem2/prepared_images/kanopus
  annotations_dir: /data/MLMarkup/Реки/test
  val_fraction: 0.2
```
