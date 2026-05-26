from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import mlsystem2.settings.api as settings_api
from mlsystem2.settings.contracts import SettingsError


def test_get_settings_before_load_raises() -> None:
    api = importlib.reload(settings_api)

    with pytest.raises(SettingsError):
        api.get_settings()
    with pytest.raises(SettingsError):
        api.get_settings_path()


def test_load_settings_without_storage_section(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(_minimal_config(), encoding="utf-8")

    settings = api.load_settings(settings_path)

    assert settings.dataset.images_dir == "/data/mlsystem2/prepared_images/"
    assert api.get_settings() is settings
    assert api.get_settings_path() == settings_path.resolve()


def test_load_settings_accepts_multiclass_dataset(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(_multiclass_config(), encoding="utf-8")

    settings = api.load_settings(settings_path)

    assert settings.dataset.is_multiclass is True
    assert [item.slug for item in settings.dataset.classes] == ["abrasion", "rivers"]
    assert settings.dataset.scenes_file is None
    assert settings.dataset.annotation_file is None
    assert settings.train.task == "multiclass"
    assert settings.train.loss == "cross_entropy"
    assert settings.train.output_channels == 3


def test_load_settings_accepts_multiclass_class_balance_and_ce_dice(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        _multiclass_config()
        .replace("  smart_tiling: false", "  smart_tiling: true\n  class_balance: true")
        .replace("  loss: cross_entropy", "  loss: cross_entropy_dice"),
        encoding="utf-8",
    )

    settings = api.load_settings(settings_path)

    assert settings.tile_preparation.class_balance is True
    assert settings.train.loss == "cross_entropy_dice"


def test_load_settings_rejects_mixed_binary_and_multiclass_dataset(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        _multiclass_config().replace(
            "  val_fraction: 0.2",
            "  scenes_file: /data/MLMarkup/Вырубки/deforestation.txt\n"
            "  annotation_file: /data/MLMarkup/Вырубки/deforestation.geojson\n"
            "  val_fraction: 0.2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError):
        api.load_settings(settings_path)


def test_load_settings_rejects_duplicate_multiclass_slugs(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        _multiclass_config().replace("  - slug: rivers", "  - slug: abrasion"),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError):
        api.load_settings(settings_path)


def test_repeated_load_settings_replaces_current_settings(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    first_config.write_text(_minimal_config(images_dir="/first"), encoding="utf-8")
    second_config.write_text(_minimal_config(images_dir="/second"), encoding="utf-8")

    first_settings = api.load_settings(first_config)
    second_settings = api.load_settings(second_config)

    assert first_settings.dataset.images_dir == "/first"
    assert second_settings.dataset.images_dir == "/second"
    assert api.get_settings() is second_settings
    assert api.get_settings_path() == second_config.resolve()


def test_load_settings_rejects_storage_section(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        _minimal_config()
        + f"\n{'stor' + 'age'}:\n  enabled: false\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError):
        api.load_settings(settings_path)


def test_load_settings_rejects_stride_larger_than_tile_size(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(_minimal_config(tile_size=128, stride=256), encoding="utf-8")

    with pytest.raises(SettingsError):
        api.load_settings(settings_path)


def test_load_settings_accepts_segformer_train_settings(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(_minimal_config(), encoding="utf-8")

    settings = api.load_settings(settings_path)

    assert settings.train.model_name == "segformer_b2"
    assert settings.train.pretrained is False
    assert settings.train.initial_checkpoint_uri is None
    assert settings.train.learning_rate == 0.00001
    assert settings.train.weight_decay == 0.0001
    assert settings.train.loss == "bce_dice"
    assert settings.train.focal_alpha == 0.6
    assert settings.train.pos_weight == 1.0
    assert settings.train.tversky_alpha == 0.4
    assert settings.train.tversky_beta == 0.6
    assert settings.train.threshold == 0.5
    assert settings.train.early_stopping_patience == 5
    assert settings.train.max_train_batches_per_epoch is None
    assert settings.train.max_val_batches_per_epoch is None
    assert settings.train.max_training_time_sec is None
    assert settings.tile_preparation.smart_tiling is False
    assert settings.tile_preparation.positive_factor == 0.5
    assert settings.tile_preparation.val_positive_factor is None
    assert settings.tile_preparation.class_balance is False


def test_load_settings_rejects_invalid_train_loss(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        _minimal_config().replace("  loss: bce_dice", "  loss: dice_only"),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError):
        api.load_settings(settings_path)


def _minimal_config(
    *,
    images_dir: str = "/data/mlsystem2/prepared_images/",
    tile_size: int = 512,
    stride: int = 256,
) -> str:
    return """
runtime:
  project_root: /opt/mlsystem2/repo
  scratch_root: /opt/mlsystem2/scratch
  logs_root: /opt/mlsystem2/logs
  cleanup_scratch_after_mlflow_log: true

dataset:
  images_dir: {images_dir}
  scenes_file: /data/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/MLMarkup/Вырубки/deforestation.geojson
  val_fraction: 0.2

tile_preparation:
  tile_size: {tile_size}
  stride: {stride}
  num_workers: 16
  prefetch_factor: 2
  seed: 42
  augmentation_level: 0
  smart_tiling: false

train:
  model_name: segformer_b2
  input_channels: 4
  output_channels: 1
  pretrained: false
  initial_checkpoint_uri: null
  epochs: 50
  batch_size: 8
  device: cuda
  learning_rate: 0.00001
  weight_decay: 0.0001
  loss: bce_dice
  focal_alpha: 0.6
  pos_weight: 1.0
  tversky_alpha: 0.4
  tversky_beta: 0.6
  threshold: 0.5
  early_stopping_patience: 5

inference:
  checkpoint_uri: /data/mlsystem2/models/latest.pt
  threshold: 0.5
  batch_size: 8
  device: cuda

mlflow:
  enabled: true
  tracking_uri: http://mlflow.example.local:5000
  experiment_name: MLSystem2
""".format(images_dir=images_dir, tile_size=tile_size, stride=stride)


def _multiclass_config() -> str:
    return """
runtime:
  project_root: /opt/mlsystem2/repo
  scratch_root: /opt/mlsystem2/scratch
  logs_root: /opt/mlsystem2/logs
  cleanup_scratch_after_mlflow_log: true

dataset:
  images_dir: /data/mlsystem2/prepared_images/
  val_fraction: 0.2
  classes:
    - slug: abrasion
      name: Абразия
      scenes_file: /data/MLMarkup/Абразия/abrasion.txt
      annotation_file: /data/MLMarkup/Абразия/abrasion.geojson
    - slug: rivers
      name: Реки
      scenes_file: /data/MLMarkup/Реки/rivers.txt
      annotation_file: /data/MLMarkup/Реки/rivers.geojson

tile_preparation:
  tile_size: 512
  stride: 256
  num_workers: 16
  prefetch_factor: 2
  seed: 42
  augmentation_level: 0
  smart_tiling: false

train:
  task: multiclass
  model_name: segformer_b2
  input_channels: 4
  output_channels: 3
  pretrained: false
  initial_checkpoint_uri: null
  epochs: 50
  batch_size: 8
  device: cuda
  learning_rate: 0.00001
  weight_decay: 0.0001
  loss: cross_entropy
  focal_alpha: 0.6
  pos_weight: 1.0
  tversky_alpha: 0.4
  tversky_beta: 0.6
  threshold: 0.5
  early_stopping_patience: 5

inference:
  checkpoint_uri: /data/mlsystem2/models/latest.pt
  threshold: 0.5
  batch_size: 8
  device: cuda

mlflow:
  enabled: true
  tracking_uri: http://mlflow.example.local:5000
  experiment_name: MLSystem2
"""
