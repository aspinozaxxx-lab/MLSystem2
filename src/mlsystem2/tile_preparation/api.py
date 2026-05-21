"""Публичный фасад подготовки тайлов."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._dataloader import create_tile_dataloader as _create_tile_dataloader
from .contracts import TileDataloaderRequest

if TYPE_CHECKING:
    import torch


def create_tile_dataloader(
    request: TileDataloaderRequest,
) -> "torch.utils.data.DataLoader":
    return _create_tile_dataloader(request)


__all__ = ["create_tile_dataloader"]
