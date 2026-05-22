import shapely
import shapely.geometry as sg
from shapely.ops import unary_union
from typing import Union, List

_Polygon = Union[sg.Polygon, sg.MultiPolygon]

__all__ = ['intersection_over_union', 'complex_intersection_over_union']


def intersection_over_union(geometry1: _Polygon, geometry2: _Polygon, ignore_errors: bool = False) -> float:
    """Calculate IoU/Jaccard metric for two geometries"""
    try:
        intersection = geometry1.intersection(geometry2).area
        union = geometry1.area + geometry2.area - intersection
        iou = intersection / union if union != 0 else 0
    except shapely.errors.TopologicalError as e:
        if ignore_errors:
            iou = 0
        else:
            raise e
    return iou


def complex_intersection_over_union(
        geometries1: Union[_Polygon, List[_Polygon]],
        geometries2: Union[_Polygon, List[_Polygon]],
) -> float:
    """
    Calculate IoU over 2 sets of geometries
    Args:
        geometries1: list of geometry/shape
        geometries2: list of geometry/shape

    Returns:
        score: float number in range 0..1
    """
    if isinstance(geometries1, (list, tuple)):
        geometries1 = unary_union(geometries1)
    if isinstance(geometries2, (list, tuple)):
        geometries2 = unary_union(geometries2)
    return intersection_over_union(geometries1, geometries2)
