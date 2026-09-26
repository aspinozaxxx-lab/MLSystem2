"""Публичные контракты инференса."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

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


class ObjectSeparationConfig(BaseModel):
    """Общие параметры выделения объектов из области и границ."""

    model_config = ConfigDict(extra="forbid")
    foreground_threshold: float = Field(default=0.5, gt=0, lt=1)
    boundary_threshold: float = Field(default=0.5, gt=0, lt=1)
    min_marker_pixels: int = Field(default=4, ge=1)


class ObjectSceneRequest(BaseModel):
    """Размер сцены и дисковый каталог промежуточных карт."""

    model_config = ConfigDict(extra="forbid")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tile_size: int = Field(gt=0)
    work_dir: str | None = None
    separation: ObjectSeparationConfig = Field(default_factory=ObjectSeparationConfig)


class ObjectWindowPrediction(BaseModel):
    """Два канала вероятностей и маска валидности одного полного окна."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    x: int
    y: int
    probabilities: Any
    valid_pixels: Any


class ObjectSceneResult(BaseModel):
    """Дисковые массивы действительны до закрытия накопителя."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    probabilities: Any
    valid_pixels: Any
    instances: Any
    object_count: int = Field(ge=0)


@runtime_checkable
class ObjectSceneAccumulator(Protocol):
    """Накопление окон без хранения целого снимка в оперативной памяти."""

    def add_window(self, window: ObjectWindowPrediction) -> None: ...
    def finish(self) -> ObjectSceneResult: ...
    def close(self) -> None: ...


__all__ = [
    "InferenceArtifact",
    "InferenceConfig",
    "InferenceError",
    "InferenceRequest",
    "InferenceResult",
    "ObjectSeparationConfig",
    "ObjectSceneRequest",
    "ObjectWindowPrediction",
    "ObjectSceneResult",
    "ObjectSceneAccumulator",
]
