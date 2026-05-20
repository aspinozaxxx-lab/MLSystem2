"""Публичный фасад конвейера обучения."""

from __future__ import annotations

from ._runner import run_train_pipeline as _run_train_pipeline
from .contracts import TrainPipelineRequest, TrainPipelineResult


def run_train_pipeline(request: TrainPipelineRequest) -> TrainPipelineResult:
    return _run_train_pipeline(request)


__all__ = ["run_train_pipeline"]
