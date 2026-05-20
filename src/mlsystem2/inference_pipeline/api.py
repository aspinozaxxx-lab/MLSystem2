"""Публичный фасад конвейера инференса."""

from __future__ import annotations

from ._runner import run_inference_pipeline as _run_inference_pipeline
from .contracts import InferencePipelineRequest, InferencePipelineResult


def run_inference_pipeline(request: InferencePipelineRequest) -> InferencePipelineResult:
    return _run_inference_pipeline(request)


__all__ = ["run_inference_pipeline"]
