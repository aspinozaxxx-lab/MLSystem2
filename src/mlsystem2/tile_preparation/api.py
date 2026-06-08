"""Публичный фасад подготовки тайлов."""

from __future__ import annotations

from ._dataloader import create_tile_dataloader as _create_tile_dataloader
from .contracts import TileDataloaderRequest


def create_tile_dataloader(
    request: TileDataloaderRequest,
) -> object:
    return _create_tile_dataloader(request)


__all__ = ["create_tile_dataloader"]
