"""Публичный фасад обучения."""

from __future__ import annotations

from ._trainer import train_model as _train_model
from .contracts import TrainProgressSink, TrainRequest, TrainResult


def train_model(
    request: TrainRequest,
    progress_sink: TrainProgressSink | None = None,
) -> TrainResult:
    return _train_model(request, progress_sink=progress_sink)


__all__ = ["train_model"]
