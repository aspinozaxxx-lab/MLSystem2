"""Публичный фасад подготовки тайлов."""

from __future__ import annotations

from ._prefetch import build_tile_sources as _build_tile_sources
from .contracts import TileSourceBundle, TileSourceRequest


def build_tile_sources(request: TileSourceRequest) -> TileSourceBundle:
    return _build_tile_sources(request)


__all__ = ["build_tile_sources"]
