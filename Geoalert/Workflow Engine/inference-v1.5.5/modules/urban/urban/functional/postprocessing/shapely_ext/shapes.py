import shapely.geometry as sg
from .split import split_to_points
from ._algorithms.enclosing_circle import make_circle
from typing import Union

__all__ = ["make_minimum_enclosing_circle", "make_l_shape"]


def make_minimum_enclosing_circle(geometry: Union[sg.Polygon, sg.MultiPolygon]):
    """Return minimum enclosing circle for given Polygon"""
    geometry_coords = split_to_points(geometry)
    x, y, radius = make_circle(geometry_coords)
    circle = sg.Point(x, y).buffer(radius)
    return circle


def make_l_shape(x1: float, x2: float, x3: float, y1: float, y2: float, y3: float) -> sg.Polygon:
    """Create L-shape geometry from given coordinates:

    Form:
                ____________
    (x1, y3)   |            |  (x3, y3)
               |            |
    (x1, y2)   |_____       |
                     |      |
           (x2, y2)  |      |
                     |      |
                     |______|
     y     (x2, y1)            (x3, y1)
    |
    |____ x

    """
    p1 = (x1, y2)
    p2 = (x1, y3)
    p3 = (x3, y3)
    p4 = (x3, y1)
    p5 = (x2, y1)
    p6 = (x2, y2)
    p7 = p1
    return sg.Polygon(shell=[p1, p2, p3, p4, p5, p6, p7])
