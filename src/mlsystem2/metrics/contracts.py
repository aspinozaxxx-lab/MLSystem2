"""Публичные контракты метрик."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MetricsError(RuntimeError):
    """Ошибка некорректных входов метрик."""


class PixelF1Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    y_true: list[float]
    y_pred: list[float]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PixelF1Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)
    f1_pixel: float = Field(ge=0.0, le=1.0)
    epoch_time_sec: float = Field(ge=0.0)


class MetricsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs_total: int = Field(ge=0)
    best_f1_pixel: float | None = Field(default=None, ge=0.0, le=1.0)
    final_f1_pixel: float | None = Field(default=None, ge=0.0, le=1.0)
    total_epoch_time_sec: float = Field(ge=0.0)


__all__ = [
    "EpochMetrics",
    "MetricsError",
    "MetricsSummary",
    "PixelF1Request",
    "PixelF1Result",
]
