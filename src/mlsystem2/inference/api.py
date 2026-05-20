"""Публичный фасад инференса."""

from __future__ import annotations

from ._runner import run_inference as _run_inference
from .contracts import InferenceRequest, InferenceResult


def run_inference(request: InferenceRequest) -> InferenceResult:
    return _run_inference(request)


__all__ = ["run_inference"]
