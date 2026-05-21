"""Публичные контракты инференса."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlsystem2.models.contracts import ModelSpec


class InferenceError(RuntimeError):
    """Ошибка инференса."""


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_uri: str
    threshold: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(gt=0)
    device: str


class InferenceArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    kind: str
    metadata: dict[str, object] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: InferenceConfig
    images_dir: str
    output_uri: str
    model_spec: ModelSpec | None = None


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    artifacts: list[InferenceArtifact]
    report: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "InferenceArtifact",
    "InferenceConfig",
    "InferenceError",
    "InferenceRequest",
    "InferenceResult",
]
