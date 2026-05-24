"""Публичные контракты обучения."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mlsystem2.models.contracts import ModelHandle


class TrainError(RuntimeError):
    """Ошибка обучения."""


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)
    train_loss: float = Field(ge=0.0)
    train_optimizer_steps: int = Field(ge=0)
    train_skipped_optimizer_steps: int = Field(ge=0)
    val_loss: float = Field(ge=0.0)
    val_pixel_precision: float = Field(ge=0.0, le=1.0)
    val_pixel_recall: float = Field(ge=0.0, le=1.0)
    val_pixel_f1: float = Field(ge=0.0, le=1.0)
    val_positive_pixels: int = Field(ge=0)
    val_pred_positive_pixels: int = Field(ge=0)
    val_true_positive: int = Field(ge=0)
    val_false_positive: int = Field(ge=0)
    val_false_negative: int = Field(ge=0)
    val_best_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    val_prob_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    val_prob_min: float = Field(default=0.0, ge=0.0, le=1.0)
    val_prob_max: float = Field(default=0.0, ge=0.0, le=1.0)
    val_prob_p50: float = Field(default=0.0, ge=0.0, le=1.0)
    val_prob_p90: float = Field(default=0.0, ge=0.0, le=1.0)
    val_prob_p99: float = Field(default=0.0, ge=0.0, le=1.0)
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
