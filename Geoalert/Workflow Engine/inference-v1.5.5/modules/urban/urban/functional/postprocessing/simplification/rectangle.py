from shapely.geometry import Polygon, MultiPolygon
from ._simplified_geometry import SimplifiedGeometry, PolygonType
from ..constants import Shape
from loguru import logger


def _make_rectangles(geometry: PolygonType,
                     negative_buffer: float = -0.5,
                     min_hole_area: float = 50) -> PolygonType:
    if negative_buffer > 0:
        raise ValueError(f'Negative buffer {negative_buffer} is not < 0')
    if isinstance(geometry, Polygon):
        shell = geometry.exterior.minimum_rotated_rectangle.buffer(negative_buffer).exterior.coords
        holes = [h.minimum_rotated_rectangle.buffer(negative_buffer).exterior.coords for h in geometry.interiors
                 if h.area > min_hole_area]
        return Polygon(shell, holes)
    elif isinstance(geometry, MultiPolygon):
        geoms = [_make_rectangles(polygon) for polygon in geometry.geoms if isinstance(polygon, Polygon)]
        return MultiPolygon(geoms) if len(geoms) > 1 else geoms[0]
    else:
        return geometry


def simplify_geometry_with_rectangle(
        geometry: PolygonType,
        rect_neg_buffer: float = -0.5,
        rect_min_hole_area: float = 50,
        **kwargs
) -> SimplifiedGeometry:
    if kwargs:
        logger.warning(f'{list(kwargs.keys())} are no loger supported in "simplify_geometry_with_rectangle"')
    return SimplifiedGeometry(
        origin_geometry=geometry,
        simple_geometry=_make_rectangles(geometry, negative_buffer=rect_neg_buffer, min_hole_area=rect_min_hole_area),
        simple_geometry_type=Shape.RECTANGLE,
    )
