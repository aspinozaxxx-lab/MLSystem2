"""Одинаковое определение компактности для Geoalert и совместимого инференса."""

import math


def is_compact_polygon(geometry, *, min_isoperimetric_quotient: float, max_bbox_ratio: float) -> bool:
    return (
        geometry.length > 0
        and 4.0 * math.pi * geometry.area / geometry.length ** 2 >= min_isoperimetric_quotient
        and _minimum_rectangle_ratio(geometry) < max_bbox_ratio
    )


def _minimum_rectangle_ratio(geometry) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    exterior = getattr(rectangle, "exterior", None)
    if exterior is None:
        return _bounds_ratio(geometry)
    coords = list(exterior.coords)
    if len(coords) < 4:
        return _bounds_ratio(geometry)
    lengths = [math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
               for i in range(min(4, len(coords) - 1))]
    positive = [value for value in lengths if value > 0]
    return max(positive) / min(positive) if positive else 0.0


def _bounds_ratio(geometry) -> float:
    min_x, min_y, max_x, max_y = geometry.bounds
    width, height = abs(max_x - min_x), abs(max_y - min_y)
    shortest = min(width, height)
    return max(width, height) / shortest if shortest > 0 else math.inf if max(width, height) > 0 else 0.0
