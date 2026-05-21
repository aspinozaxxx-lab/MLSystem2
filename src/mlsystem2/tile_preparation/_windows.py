"""Построение внутренних окон тайлов."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


def axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def build_tile_windows(width: int, height: int, tile_size: int, stride: int) -> list[TileWindow]:
    return [
        TileWindow(x=x, y=y, width=tile_size, height=tile_size)
        for y in axis_origins(height, tile_size, stride)
        for x in axis_origins(width, tile_size, stride)
    ]
