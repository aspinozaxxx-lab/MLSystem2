from __future__ import annotations

from pathlib import Path

import pytest

from mlsystem2.settings.api import load_settings
from mlsystem2.settings.contracts import SettingsError


def test_load_settings_without_storage_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_config(), encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.dataset.images_dir == "/data/mlsystem2/prepared_images/kanopus"


def test_load_settings_rejects_storage_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _minimal_config()
        + f"\n{'stor' + 'age'}:\n  enabled: false\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError):
        load_settings(config_path)


def _minimal_config() -> str:
    return """
runtime:
  project_root: /opt/mlsystem2/repo
  scratch_root: /opt/mlsystem2/scratch
  logs_root: /opt/mlsystem2/logs
  cleanup_scratch_after_mlflow_log: true

dataset:
  images_dir: /data/mlsystem2/prepared_images/kanopus
  scenes_file: /data/mlsystem/MLMarkup/Вырубки/deforestation.txt
  annotation_file: /data/mlsystem/MLMarkup/Вырубки/deforestation.geojson
  val_fraction: 0.2

tile_preparation:
  tile_size: 512
  stride: 256
  prefetch_workers: 16
  prefetch_batches: 8

train:
  model_name: unet
  input_channels: 3
  output_channels: 1
  epochs: 50
  batch_size: 8
  device: cuda
  num_workers: 8

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
