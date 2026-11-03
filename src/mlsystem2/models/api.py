"""Публичный фасад моделей."""

from __future__ import annotations

from ._checkpoint import load_checkpoint as _load_checkpoint
from ._checkpoint import save_checkpoint as _save_checkpoint
from ._factory import create_model as _create_model
from ._factory import list_supported_models as _list_supported_models
from ._initialization import compatible_training_pipelines as _compatible_training_pipelines
from ._initialization import initialize_model as _initialize_model
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


def compatible_training_pipelines(spec: ModelSpec) -> list[str]:
    return _compatible_training_pipelines(spec)


def initialize_model(source: ModelHandle, target_spec: ModelSpec, *, preserve_global_preprocessing: bool = True) -> ModelHandle:
    return _initialize_model(source, target_spec, preserve_global_preprocessing=preserve_global_preprocessing)


def load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint:
    return _load_checkpoint(request)


def save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact:
    return _save_checkpoint(request)


__all__ = ["list_supported_models", "create_model", "compatible_training_pipelines", "initialize_model", "load_checkpoint", "save_checkpoint"]
