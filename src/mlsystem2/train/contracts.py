"""Публичные контракты обучения."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlsystem2.models.contracts import ModelHandle


class TrainError(RuntimeError):
    """Ошибка обучения."""


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["binary", "multiclass"] = "binary"
    quality_metric: Literal["pixel", "objects"] = "pixel"
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: str
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss: Literal["bce_dice", "focal_dice", "focal_tversky", "cross_entropy", "cross_entropy_dice"]
    focal_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    pos_weight: float = Field(default=1.0, gt=0.0)
    hard_negative_weight: float = Field(default=1.0, gt=0.0)
    tversky_alpha: float = Field(default=0.4, gt=0.0)
    tversky_beta: float = Field(default=0.6, gt=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(gt=0)
    max_train_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_val_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_training_time_sec: int | None = Field(default=None, gt=0)
    class_slugs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_loss(self) -> Self:
        multiclass_losses = {"cross_entropy", "cross_entropy_dice"}
        if self.task == "multiclass" and self.loss not in multiclass_losses:
            raise ValueError("multiclass train требует loss=cross_entropy или cross_entropy_dice")
        if self.task == "binary" and self.loss in multiclass_losses:
            raise ValueError("binary train не поддерживает multiclass loss")
        if self.task != "binary" and self.quality_metric == "objects":
            raise ValueError("Объектовая метрика качества поддерживается только для binary train")
        return self


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def fill_quality_metric_compatibility(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        resolved = dict(data)
        resolved.setdefault(
            "val_quality_f1",
            resolved.get("val_best_threshold_pixel_f1", 0.0),
        )
        resolved.setdefault(
            "val_quality_precision",
            resolved.get("val_best_threshold_precision", 0.0),
        )
        resolved.setdefault(
            "val_quality_recall",
            resolved.get("val_best_threshold_recall", 0.0),
        )
        resolved.setdefault(
            "val_best_pixel_threshold",
            resolved.get("val_best_threshold", 0.0),
        )
        resolved.setdefault(
            "val_best_threshold_pixel_precision",
            resolved.get("val_best_threshold_precision", 0.0),
        )
        resolved.setdefault(
            "val_best_threshold_pixel_recall",
            resolved.get("val_best_threshold_recall", 0.0),
        )
        return resolved

    epoch: int = Field(ge=0)
    train_loss: float = Field(ge=0.0)
    val_loss: float = Field(ge=0.0)
    quality_metric: Literal["pixel", "objects"] = "pixel"
    val_quality_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    val_quality_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_quality_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_pixel_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_object_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_object_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_object_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    epoch_time_sec: float = Field(ge=0.0)


class CheckpointArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    label: str


class TrainProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)
    message: str
    metrics: EpochMetrics | None = None


@runtime_checkable
class TrainProgressSink(Protocol):
    def __call__(self, event: TrainProgressEvent) -> None: ...


class TrainRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: ModelHandle
    train_loader: object
    val_loader: object
    config: TrainConfig
    checkpoint_dir: str
    sample_size: int | None = Field(default=None, gt=0)


class TrainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[EpochMetrics]
    epochs_total: int = Field(ge=0)
    training_time_sec: float = Field(ge=0.0)
    best_checkpoint_path: str | None = None
    final_checkpoint_path: str | None = None
    artifacts: list[CheckpointArtifact] = Field(default_factory=list)


__all__ = [
    "CheckpointArtifact",
    "EpochMetrics",
    "TrainConfig",
    "TrainError",
    "TrainProgressEvent",
    "TrainProgressSink",
    "TrainRequest",
    "TrainResult",
]
