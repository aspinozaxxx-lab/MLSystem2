"""Публичные контракты подготовки тайлов."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mlsystem2.dataset_preparing.contracts import PreparedDataset


class TilePreparationError(RuntimeError):
    """Ошибка подготовки тайлов."""


class TileBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    inputs: object
    targets: object | None = None
    scene_ids: list[str]
    metadata: dict[str, object] = Field(default_factory=dict)


class TilePreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_batches_prepared: int = Field(ge=0)
    val_batches_prepared: int = Field(ge=0)
    queue_capacity: int = Field(ge=0)
    worker_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


@runtime_checkable
class TileBatchSource(Protocol):
    def __iter__(self) -> Iterator[TileBatch]: ...

    def close(self) -> None: ...

    def profile_snapshot(self) -> TilePreparationReport: ...


class TileSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: PreparedDataset
    tile_size: int = Field(gt=0)
    stride: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    prefetch_workers: int = Field(gt=0)
    prefetch_batches: int = Field(gt=0)


class TileSourceBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train: TileBatchSource
    val: TileBatchSource
    report: TilePreparationReport


__all__ = [
    "TileBatch",
    "TileBatchSource",
    "TilePreparationError",
    "TilePreparationReport",
    "TileSourceBundle",
    "TileSourceRequest",
]
