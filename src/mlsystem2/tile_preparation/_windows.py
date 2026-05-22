"""Построение внутренних окон тайлов."""

from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET


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


def build_vrt_source_windows(
    vrt_xml: str,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
) -> list[TileWindow]:
    source_rects = _parse_source_rects(vrt_xml)
    if not source_rects:
        return build_tile_windows(width, height, tile_size, stride)

    windows: list[TileWindow] = []
    seen: set[tuple[int, int]] = set()
    for rect in source_rects:
        for y in _regular_rect_origins(rect.y, rect.height, stride):
            for x in _regular_rect_origins(rect.x, rect.width, stride):
                key = (x, y)
                if key in seen:
                    continue
                seen.add(key)
                windows.append(TileWindow(x=x, y=y, width=tile_size, height=tile_size))
    return windows


@dataclass(frozen=True, slots=True)
class _SourceRect:
    x: int
    y: int
    width: int
    height: int


def _parse_source_rects(vrt_xml: str) -> list[_SourceRect]:
    root = ET.fromstring(vrt_xml)
    first_band = root.find("VRTRasterBand")
    if first_band is None:
        return []

    rects: list[_SourceRect] = []
    for source in first_band:
        if source.tag not in {"SimpleSource", "ComplexSource"}:
            continue
        dst_rect = source.find("DstRect")
        if dst_rect is None:
            continue
        x = math.floor(float(dst_rect.attrib["xOff"]))
        y = math.floor(float(dst_rect.attrib["yOff"]))
        right = math.ceil(float(dst_rect.attrib["xOff"]) + float(dst_rect.attrib["xSize"]))
        bottom = math.ceil(float(dst_rect.attrib["yOff"]) + float(dst_rect.attrib["ySize"]))
        rects.append(_SourceRect(x=x, y=y, width=max(1, right - x), height=max(1, bottom - y)))
    return rects


def _regular_rect_origins(start: int, length: int, stride: int) -> list[int]:
    return [max(origin, 0) for origin in regular_axis_origins(start, length, stride)]
