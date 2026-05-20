"""Публичные контракты моделей."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelsError(RuntimeError):
    """Ошибка модели или чекпойнта."""


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_channels: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    pretrained: bool = False
    parameters: dict[str, object] = Field(default_factory=dict)


class ModelHandle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    spec: ModelSpec
    model: object


class LoadCheckpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_uri: str
    model_spec: ModelSpec | None = None
    map_location: str | None = None


class SaveCheckpointRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: ModelHandle
    checkpoint_uri: str
    metadata: dict[str, object] = Field(default_factory=dict)


class CheckpointArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    format: str
    metadata: dict[str, object] = Field(default_factory=dict)


class LoadedCheckpoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: ModelHandle
    artifact: CheckpointArtifact


__all__ = [
    "CheckpointArtifact",
    "LoadCheckpointRequest",
    "LoadedCheckpoint",
    "ModelHandle",
    "ModelSpec",
    "ModelsError",
    "SaveCheckpointRequest",
]
