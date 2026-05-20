"""Публичные контракты настроек."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SettingsError(RuntimeError):
    """Ошибка загрузки или валидации настроек."""


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    scratch_root: str
    logs_root: str
    cleanup_scratch_after_mlflow_log: bool


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_uri: str
    mlmarkup_repo: str
    local_cache_root: str
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None


class DatasetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_dir: str
    scenes_file: str
    annotation_file: str
    val_fraction: float = Field(gt=0.0, lt=1.0)


class TilePreparationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: int = Field(gt=0)
    stride: int = Field(gt=0)
    prefetch_workers: int = Field(default=16, gt=0)
    prefetch_batches: int = Field(gt=0)


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    input_channels: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: str
    num_workers: int = Field(ge=0)


class InferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_uri: str
    threshold: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(gt=0)
    device: str


class MLflowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    tracking_uri: str
    experiment_name: str


class SystemSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeSettings
    storage: StorageSettings
    dataset: DatasetSettings
    tile_preparation: TilePreparationSettings
    train: TrainSettings
    inference: InferenceSettings
    mlflow: MLflowSettings


__all__ = [
    "DatasetSettings",
    "InferenceSettings",
    "MLflowSettings",
    "RuntimeSettings",
    "SettingsError",
    "StorageSettings",
    "SystemSettings",
    "TilePreparationSettings",
    "TrainSettings",
]
