"""Публичные контракты настроек."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsError(RuntimeError):
    """Ошибка загрузки или валидации настроек."""


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    scratch_root: str
    logs_root: str
    cleanup_scratch_after_mlflow_log: bool


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
    num_workers: int = Field(default=16, ge=0)
    prefetch_factor: int = Field(default=2, gt=0)
    seed: int = 42
    augmentation_level: int = Field(default=0, ge=0, le=3)
    smart_tiling: bool = False
    positive_factor: float = Field(default=0.5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_stride(self) -> Self:
        if self.stride > self.tile_size:
            raise ValueError("stride должен быть меньше или равен tile_size")
        return self


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    input_channels: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    pretrained: bool = False
    initial_checkpoint_uri: str | None = None
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: str
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss: Literal["bce_dice", "focal_dice", "focal_tversky"]
    focal_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    pos_weight: float = Field(default=1.0, gt=0.0)
    tversky_alpha: float = Field(default=0.4, gt=0.0)
    tversky_beta: float = Field(default=0.6, gt=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(gt=0)
    max_train_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_val_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_training_time_sec: int | None = Field(default=None, gt=0)


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
    "SystemSettings",
    "TilePreparationSettings",
    "TrainSettings",
]
