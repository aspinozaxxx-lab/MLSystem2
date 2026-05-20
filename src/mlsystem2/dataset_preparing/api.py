"""Публичный фасад подготовки датасета."""

from __future__ import annotations

from ._prepare import prepare_dataset as _prepare_dataset
from .contracts import DatasetPreparationRequest, DatasetPreparationResult


def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    return _prepare_dataset(request)


__all__ = ["prepare_dataset"]
