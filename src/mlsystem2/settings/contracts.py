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


class DatasetClassSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    scenes_file: str
    annotation_file: str
    priority: int = 0


class DatasetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_dir: str
    scenes_file: str | None = None
    annotation_file: str | None = None
    classes: list[DatasetClassSettings] = Field(default_factory=list)
    val_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_dataset_mode(self) -> Self:
        has_binary_paths = self.scenes_file is not None or self.annotation_file is not None
        has_classes = bool(self.classes)
        if has_binary_paths and has_classes:
            raise ValueError(
                "dataset должен задавать либо classes, либо scenes_file + annotation_file"
            )
        if has_classes:
            _validate_unique_values([item.slug for item in self.classes], "slug")
            _validate_unique_values([item.name for item in self.classes], "name")
            return self
        if not self.scenes_file or not self.annotation_file:
            raise ValueError(
                "binary dataset должен задавать scenes_file и annotation_file"
            )
        return self

    @property
    def is_multiclass(self) -> bool:
        return bool(self.classes)


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
    val_positive_factor: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_stride(self) -> Self:
        if self.stride > self.tile_size:
            raise ValueError("stride должен быть меньше или равен tile_size")
        return self


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["binary", "multiclass"] = "binary"
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
    loss: Literal["bce_dice", "focal_dice", "focal_tversky", "cross_entropy"]
    focal_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    pos_weight: float = Field(default=1.0, gt=0.0)
    tversky_alpha: float = Field(default=0.4, gt=0.0)
    tversky_beta: float = Field(default=0.6, gt=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(gt=0)
    max_train_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_val_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_training_time_sec: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_task_loss(self) -> Self:
        if self.task == "multiclass" and self.loss != "cross_entropy":
            raise ValueError("multiclass train требует loss=cross_entropy")
        if self.task == "binary" and self.loss == "cross_entropy":
            raise ValueError("binary train не поддерживает loss=cross_entropy")
        return self


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

    @model_validator(mode="after")
    def validate_dataset_train_consistency(self) -> Self:
        if self.dataset.is_multiclass:
            if self.train.task != "multiclass":
                raise ValueError("dataset.classes требует train.task=multiclass")
            expected_channels = len(self.dataset.classes) + 1
            if self.train.output_channels != expected_channels:
                raise ValueError(
                    "multiclass output_channels должен быть равен "
                    f"len(dataset.classes) + 1: ожидается {expected_channels}"
                )
        elif self.train.task != "binary":
            raise ValueError("binary dataset требует train.task=binary")
        return self


def _validate_unique_values(values: list[str], field_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"dataset.classes должен иметь уникальные {field_name}: {joined}")


__all__ = [
    "DatasetClassSettings",
    "DatasetSettings",
    "InferenceSettings",
    "MLflowSettings",
    "RuntimeSettings",
    "SettingsError",
    "SystemSettings",
    "TilePreparationSettings",
    "TrainSettings",
]
