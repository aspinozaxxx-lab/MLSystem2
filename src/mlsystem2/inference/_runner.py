"""Заглушка инференса."""

from __future__ import annotations

from mlsystem2.models.api import load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest

from .contracts import InferenceRequest, InferenceResult


def run_inference(request: InferenceRequest) -> InferenceResult:
    load_checkpoint(
        LoadCheckpointRequest(
            checkpoint_uri=request.config.checkpoint_uri,
            model_spec=request.model_spec,
            map_location=request.config.device,
        )
    )
    raise NotImplementedError(
        "Реальный Python/PyTorch инференс по тайлам намеренно оставлен для шага миграции из старого проекта."
    )
