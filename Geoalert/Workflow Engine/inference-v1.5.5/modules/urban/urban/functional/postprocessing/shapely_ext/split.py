import shapely
import shapely.geometry as sg
from typing import Union, List, Tuple

_Point = Tuple[float, float]

__all__ = ['split_to_points', 'split_to_lines', 'split_rectangle_to_four_parts']


def split_to_points(geometry: Union[sg.LineString, sg.MultiLineString, sg.Polygon, sg.MultiPolygon]) -> Tuple[_Point]:
    """Split geometry to list of coordinate points, work with `Multi` geometries"""
    if isinstance(geometry, (sg.Polygon, sg.MultiPolygon)):
        return split_to_points(geometry.boundary)
    elif isinstance(geometry, sg.MultiLineString):
        coords = []
        for sub_geometry in geometry.geoms:
            coords.extend(split_to_points(sub_geometry))
        return tuple(coords)
    elif isinstance(geometry, sg.LineString):
        return tuple(geometry.coords)
    else:
        raise ValueError("Wrong geometry type {}".format(type(geometry)))


def split_to_lines(geometry: Union[sg.LineString, sg.MultiLineString, sg.Polygon, sg.MultiPolygon]) -> List[
    sg.LineString]:
    """Split geometry to separate lines (2-point lines), return list of LineStrings"""
    if isinstance(geometry, (sg.Polygon, sg.MultiPolygon)):
        return split_to_lines(geometry.boundary)
    if isinstance(geometry, sg.MultiLineString):
        lines = []
        for sub_geometry in geometry.geoms:
            lines.extend(split_to_lines(sub_geometry))
        return lines
    elif isinstance(geometry, sg.LineString):
        coords = list(geometry.coords)
        lines = [shapely.geometry.LineString([coords[i], coords[i + 1]])
                 for i in range(len(coords) - 1)]
        return lines
    else:
        raise ValueError("Wrong geometry type {}".format(type(geometry)))


def split_rectangle_to_four_parts(polygon: sg.Polygon) -> Tuple[sg.Polygon,sg.Polygon, sg.Polygon, sg.Polygon]:
    """Generate 4-equal polygons from given by dividing each edge by 2.
    Resulted tuple  is ordered (bottom->top, left->right) as follows:

        | 1 | 3 |
        | 0 | 2 |

    """
    x1, y1, x2, y2 = polygon.bounds
    xm = x1 + (x2 - x1) / 2
    ym = y1 + (y2 - y1) / 2
    p1 = sg.box(x1, y1, xm, ym)
    p2 = sg.box(x1, ym, xm, y2)
    p3 = sg.box(xm, ym, x2, y2)
    p4 = sg.box(xm, y1, x2, ym)
    return p1, p2, p3, p4
