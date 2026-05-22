import shapely
from ._simplified_geometry import SimplifiedGeometry, PolygonType
from ..constants import Shape
from .. import shapely_ext as se
from loguru import logger

NEGATIVE_BUFFER = - 0.5


def simplify_geometry_with_circle(geometry: PolygonType, **kwargs) -> SimplifiedGeometry:
    if kwargs:
        logger.warning(f'{list(kwargs.keys())} are no loger supported in "simplify_geometry_with_circle"')
    # circle
    # circle = se.make_minimum_enclosing_circle(geometry).buffer(NEGATIVE_BUFFER)
    circle = shapely.minimum_bounding_circle(geometry)
    iou = se.intersection_over_union(circle, geometry, ignore_errors=True)

    # rectangle
    rectangle = geometry.minimum_rotated_rectangle.buffer(NEGATIVE_BUFFER)
    rect_iou = se.intersection_over_union(rectangle, geometry)

    # if rectangle is better not simplify with circle (return bad metrics)
    iou = iou if iou > rect_iou else 0.

    simple_geometry = SimplifiedGeometry(
        origin_geometry=geometry,
        simple_geometry=circle,
        simple_geometry_type=Shape.CIRCLE,
        iou=iou,
    )
    return simple_geometry
