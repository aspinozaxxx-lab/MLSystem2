import functools
import scipy
import scipy.spatial
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
from typing import Tuple, Union, List
from collections.abc import Iterable
from .. import shapely_ext as se
from ..constants import Shape
from ._simplified_geometry import SimplifiedGeometry, PolygonType
from loguru import logger

SIMPLIFICATION_RATE = 1.


# ----------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------


def contour_to_polygon(contour: list) -> sg.Polygon:
    """Convert contour to Polygon"""
    return sg.Polygon(shell=contour).buffer(0)


def polygon_to_contour(polygon: sg.Polygon) -> list:
    """Convert Polygon !! EXTERIOR !! to contour"""
    return list(polygon.exterior.coords)


def polygon_to_contours(polygon: sg.Polygon) -> list:
    """Convert Polygon to contours. First one is shell, other holes."""
    contours = [list(interior.coords) for interior in polygon.interiors]
    contours.insert(0, list(polygon.exterior.coords))
    return contours


def contours_to_polygon(contours: list) -> sg.Polygon:
    """Convert contours to Polygon. First one is shell, other holes."""
    polygon = sg.Polygon(shell=contours[0], holes=contours[1:])
    polygon = polygon.buffer(0)
    return polygon


def roll_to_longest(line_string: list, k: int = 1) -> list:
    """ Roll line string to start with long line

    Args:
        line_string: line_string
        k: index of line in sorted lines list by length

    """
    contour = np.array(line_string)

    lengths = np.asarray(
        [np.linalg.norm(contour[i + 1] - contour[i]) for i in range(contour.shape[0] - 1)]
    )

    max_len_ind = np.argsort(lengths)[-k]

    new_contour = np.roll(contour[:-1], -max_len_ind, axis=0).tolist()
    new_contour.append(new_contour[0])

    return new_contour


def _make_polar_grid(
        radius_range: tuple,
        step: int,
        angles: tuple = (0, 90),
        min_values: Union[Tuple[float, float], float] = 0.1,
) -> np.array:
    """ Polar grid with specified angles and radius

    Args:
        radius_range: tuple of two int, e.g. (-200, 200)
        step: grid step
        angles: grid angles
        min_values: min value of radius on the grid for each angle (abs. value)

    Returns:
        grid: np.array of shape (N_POINTS, 2)

    """

    points = []

    if type(min_values) not in [list, tuple]:
        min_values = [min_values] * len(angles)

    for angle, min_value in zip(angles, min_values):
        cos = np.cos(np.radians(angle))
        sin = np.sin(np.radians(angle))
        for r in np.arange(radius_range[0], radius_range[1] + step, step):
            if np.abs(r) > min_value:
                x, y = r * cos, r * sin
                points.append((x, y))

    return np.array(points)


class GridSnapper:

    def __init__(self, grid: np.array):
        self.grid = grid
        self.index = scipy.spatial.cKDTree(self.grid)

    @staticmethod
    def _get_params(ls: list) -> tuple:
        assert len(ls) == 3
        x0, y0 = ls[0]
        x1, y1 = ls[1]
        theta = np.pi / 2 - np.arctan2((x1 - x0), (y1 - y0))
        return -theta, (-x1, -y1)

    @staticmethod
    def _to_zero(
            line_string: list,
            theta: float,
            xoff: float,
            yoff: float,
    ) -> list:
        ls = sg.LineString(line_string)
        ls = sa.translate(ls, xoff=xoff, yoff=yoff)
        ls = sa.rotate(ls, theta, origin=(0, 0), use_radians=True)
        return list(ls.coords)

    @staticmethod
    def _to_origin(
            line_string: list,
            theta: float,
            xoff: float,
            yoff: float,
    ) -> list:
        ls = sg.LineString(line_string)
        ls = sa.rotate(ls, -theta, origin=(0, 0), use_radians=True)
        ls = sa.translate(ls, xoff=-xoff, yoff=-yoff)
        return list(ls.coords)

    def snap_linestring(
            self,
            line_string: list,
            theta_diff: float = 0.,
    ) -> list:
        assert len(line_string) == 3

        # move to center and rotate
        theta, (xoff, yoff) = self._get_params(line_string)

        norm_ls = self._to_zero(line_string, theta + theta_diff, xoff, yoff)

        # find point
        target_point = norm_ls[2]
        distance, ind = self.index.query(target_point)
        snap_norm_ls = norm_ls[:2] + [tuple(self.grid[ind])]

        # return back
        ls = self._to_origin(snap_norm_ls, theta, xoff, yoff)

        return ls

    def snap_contour(self, contour):
        snapped_contour = contour[:2]

        for i in range(2, len(contour) - 1):
            ls = snapped_contour[-2:] + [contour[i]]
            snapped_point = self.snap_linestring(ls)[-1]
            snapped_contour.append(snapped_point)

        snapped_contour.append(snapped_contour[0])

        return snapped_contour


@functools.lru_cache(maxsize=None)
def get_grid_snapper(radius_range=(-400, 400), step=1, angles=(0, 90), min_values=(3., 3.)):
    if not isinstance(radius_range, Iterable):
        radius_range = (-radius_range, radius_range)
    if not isinstance(min_values, Iterable):
        min_values = (min_values, min_values)

    return GridSnapper(_make_polar_grid(radius_range=radius_range, step=step, angles=angles, min_values=min_values))


def _simplify_contour_with_grid_snap(grid_snapper: GridSnapper, contour: List):
    """Function to simplify given polygon contour with grid snapping"""
    geometry = contour_to_polygon(contour)
    best_geometry = geometry.envelope
    best_iou_score = se.intersection_over_union(geometry, best_geometry)

    contour = polygon_to_contour(geometry)
    for k in range(1, min(4, len(contour))):

        contour = roll_to_longest(contour, k)

        simple_contour = grid_snapper.snap_contour(contour)
        simple_geometry = contour_to_polygon(simple_contour)

        iou_score = se.intersection_over_union(
            geometry,
            simple_geometry,
        )

        if iou_score > best_iou_score:
            best_iou_score = iou_score
            best_geometry = simple_geometry

    return best_geometry


def _simplify_geometry_with_grid_snap(grid_snapper: GridSnapper, geometry: PolygonType):
    """Function to handle polygons anf multipolygons representation of geometry"""
    if isinstance(geometry, sg.Polygon):
        contours = polygon_to_contours(geometry)
        simple_geometry, *holes = [_simplify_contour_with_grid_snap(grid_snapper, c) for c in contours]
        for hole in holes:
            simple_geometry = simple_geometry.difference(hole)
    elif isinstance(geometry, sg.MultiPolygon):
        simple_polygons = []
        for p in geometry.geoms:
            geometry = _simplify_geometry_with_grid_snap(grid_snapper, p)
            if isinstance(geometry, sg.MultiPolygon):
                simple_polygons.extend([_ for _ in geometry.geoms])
            else:
                simple_polygons.append(geometry)
        simple_geometry = sg.MultiPolygon(simple_polygons)
    else:
        raise ValueError("Expected Polygon o MultiPolygon geometry, got '{}'.".format(type(geometry)))

    # correct geometry
    simple_geometry = simple_geometry.buffer(-0.1).buffer(0.1).simplify(0.5)
    return simple_geometry


def simplify_geometry_with_grid_snap(geometry: PolygonType,
                                     radius_range=(-400, 400),
                                     step=1,
                                     angles=(0, 90),
                                     min_values=(3.0, 3.0),
                                     **kwargs) -> SimplifiedGeometry:

    if kwargs:
        logger.warning(f'{list(kwargs.keys())} are no loger supported in "simplify_geometry_with_grid_snap"')

    grid_snapper = get_grid_snapper(radius_range, step, angles, min_values)
    try:
        simplified_geometry = _simplify_geometry_with_grid_snap(grid_snapper, geometry.simplify(SIMPLIFICATION_RATE))
        iou = se.intersection_over_union(simplified_geometry, geometry)
    except:
        simplified_geometry = geometry
        iou = 0.

    simplified_geometry = SimplifiedGeometry(
        origin_geometry=geometry,
        simple_geometry=simplified_geometry,
        simple_geometry_type=Shape.GRID_SNAP,
        iou=iou,
    )

    return simplified_geometry
