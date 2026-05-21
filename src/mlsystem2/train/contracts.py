"""Публичные контракты обучения."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mlsystem2.models.contracts import ModelHandle


class TrainError(RuntimeError):
    """Ошибка обучения."""


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: str


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)
    f1_pixel: float = Field(ge=0.0, le=1.0)
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
