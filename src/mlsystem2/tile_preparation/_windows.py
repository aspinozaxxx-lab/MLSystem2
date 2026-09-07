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


def build_tile_windows(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    context: int = 0,
    *,
    full_windows_only: bool = False,
) -> list[TileWindow]:
    if context < 0 or tile_size <= 2 * context:
        raise ValueError("tile_size должен быть больше удвоенного context")
    if full_windows_only:
        if context:
            raise ValueError("Полные окна next-gen2 не используют контекст")
        return [
            TileWindow(x=x, y=y, width=tile_size, height=tile_size)
            for y in range(0, height - tile_size + 1, stride)
            for x in range(0, width - tile_size + 1, stride)
        ]
    return [
        TileWindow(x=x - context, y=y - context, width=tile_size, height=tile_size)
        for y in regular_axis_origins(0, height, stride)
        for x in regular_axis_origins(0, width, stride)
    ]


def core_tile_window(window: TileWindow, context: int) -> TileWindow:
    if context < 0 or window.width <= 2 * context or window.height <= 2 * context:
        raise ValueError("Размер окна должен быть больше удвоенного context")
    return TileWindow(
        x=window.x + context,
        y=window.y + context,
        width=window.width - 2 * context,
        height=window.height - 2 * context,
    )
