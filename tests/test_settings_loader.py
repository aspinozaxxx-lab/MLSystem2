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


def test_load_settings_without_storage_section(tmp_path: Path) -> None:
    api = importlib.reload(settings_api)
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(_minimal_config(), encoding="utf-8")

    settings = api.load_settings(settings_path)

    assert settings.dataset.images_dir == "/data/mlsystem2/prepared_images/kanopus"
    assert api.get_settings() is settings


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


def _minimal_config(
    *,
    images_dir: str = "/data/mlsystem2/prepared_images/kanopus",
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
  scenes_file: /data/mlsystem/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/mlsystem/MLMarkup/Вырубки/deforestation.geojson
  val_fraction: 0.2

tile_preparation:
  tile_size: {tile_size}
  stride: {stride}
  num_workers: 16
  prefetch_factor: 2
  seed: 42
  augmentation_level: 0

train:
  model_name: unet
  input_channels: 3
  output_channels: 1
  epochs: 50
  batch_size: 8
  device: cuda

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
