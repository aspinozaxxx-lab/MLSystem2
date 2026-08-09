"""Построение внутренних окон тайлов."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


def regular_axis_origins(start: int, length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    return list(range(start, start + length, stride))


def build_tile_windows(width: int, height: int, tile_size: int, stride: int) -> list[TileWindow]:
    return [
        TileWindow(x=x, y=y, width=tile_size, height=tile_size)
        for y in regular_axis_origins(0, height, stride)
        for x in regular_axis_origins(0, width, stride)
    ]
