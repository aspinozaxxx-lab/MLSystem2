"""Low-level utils to work with shapely geometry"""
import shapely
from loguru import logger
from functools import wraps
import math
import shapely.geometry as sg
from typing import List, Tuple


def apply_buffer_if_invalid(func):
    """Decorator to check output geometry is valid, if not apply .buffer(0)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if not res.is_valid:
            res = res.buffer(0)
        return res
    return wrapper


def return_empty_if_error(geometry_class):
    def decorator(func):
        """Decorator to catch TopologicalError and return empty Polygon instead"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(e)
                return geometry_class()
        return wrapper
    return decorator


def to_multipolygon(geometries: List) -> shapely.geometry.MultiPolygon:
    """Convert a list of Polygons and MultiPolygons to a single MultiPolygon"""
    polygons = []
    for g in geometries:
        if isinstance(g, shapely.geometry.Polygon):
            polygons.append(g)
        elif isinstance(g, shapely.geometry.MultiPolygon):
            for p in g.geoms:
                polygons.append(p)
    return shapely.geometry.MultiPolygon(polygons)


def calculate_iou(g1, g2):
    """Calculate Intersection over Union between two geoms, if failed return 0."""
    intersection_area = intersection(g1, g2).area
    union_area = union(g1, g2).area
    return intersection_area / union_area if union_area != 0. else 0.


@return_empty_if_error(shapely.geometry.MultiPolygon)
@apply_buffer_if_invalid
def intersection(g1, g2):
    return g1.intersection(g2)


@return_empty_if_error(shapely.geometry.MultiPolygon)
@apply_buffer_if_invalid
def union(g1, g2):
    return g1.union(g2)


# TODO: move this to utils/angleutils


_Point = Tuple[float, float]

__all__ = ['get_angle_for_line', 'get_angles_for_polygon']


def get_angle_for_line(line: sg.LineString) -> float:
    """Calculate line angle, line should contain only 2 points"""
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def get_angles_for_polygon(polygon: sg.Polygon, sort: bool = True) -> List[float]:
    """Return positive line angles (% 180) in degrees for all lines in geometry,
    could be sorted in descending order according to line length"""
    lines = split_to_lines(polygon)
    lines = sorted(lines, key=lambda x: x.length, reverse=True) if sort else lines
    angles = [get_angle_for_line(l) % 180 for l in lines]
    return angles


def angle_envelope(geometry):
    envelope = geometry.oriented_envelope.exterior
    vec1 = np.array(envelope.coords[0]) - np.array(envelope.coords[1])
    return np.rad2deg(np.arctan2(vec1[1], vec1[0]))


def angle_PCA(geometry, pre_simplification: float = 0.1, segmentize: float = 100, cond_thr: float = 4):
    geometry = geometry.simplify(pre_simplification)
    if segmentize:
        geometry = shapely.segmentize(geometry, segmentize)
    c = np.array(geometry.exterior.coords)
    vectors = c - np.roll(c, 1, axis=0)
    cov_matrix = np.cov(vectors, rowvar=False)
    if np.linalg.cond(cov_matrix) < cond_thr:
        return angle_envelope(geometry)
    eigval, eigvec = np.linalg.eig(cov_matrix)
    eigvec = (eigvec*eigval).T
    return np.rad2deg(np.arctan2(eigvec[0, 1], eigvec[0, 0]))