"""Публичный фасад моделей."""

from __future__ import annotations

from ._checkpoint import load_checkpoint as _load_checkpoint
from ._checkpoint import save_checkpoint as _save_checkpoint
from ._factory import create_model as _create_model
from ._factory import list_supported_models as _list_supported_models
from .contracts import (
    CheckpointArtifact,
    LoadCheckpointRequest,
    LoadedCheckpoint,
    ModelHandle,
    ModelSpec,
    SaveCheckpointRequest,
)


def list_supported_models() -> list[ModelSpec]:
    return _list_supported_models()


def create_model(spec: ModelSpec) -> ModelHandle:
    return _create_model(spec)


def load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint:
    return _load_checkpoint(request)


def save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact:
    return _save_checkpoint(request)


__all__ = ["list_supported_models", "create_model", "load_checkpoint", "save_checkpoint"]
