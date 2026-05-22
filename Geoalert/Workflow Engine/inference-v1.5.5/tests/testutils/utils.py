import shapely
from typing import Sequence, Final
import numpy as np
import shapely.affinity
from shapely import Polygon
from urban.functional.utils.angleutils import vector_from_height, invert_angle
from urban.functional.postprocessing.constants import Shape


def random_points(n: int, aoi: Polygon, n_grid: int = 5, margin: float = 0.1) -> np.ndarray:
    """Returns (N, 2) shaped array representing random points within aoi snapped to grid"""
    grid = np.stack(np.meshgrid(np.linspace(-0.5+margin, 0.5-margin, n_grid),
                                np.linspace(-0.5+margin, 0.5-margin, n_grid))).reshape(2, -1).T
    coords = grid[np.random.choice(len(grid), n, replace=False)]
    coords[:, 0] = coords[:, 0] * (aoi.bounds[2] - aoi.bounds[0])
    coords[:, 1] = coords[:, 1] * (aoi.bounds[3] - aoi.bounds[1])
    return coords + aoi.centroid.coords


def create_triangle_polygon(base_vertex: Sequence[float],
                            height: float = 1,
                            azimuth: float = 0,
                            elevation: float = 45,
                            delta_height: float = 0.9,
                            delta_angle: float = 5):
    azimuth = invert_angle(azimuth)
    vec_primary = vector_from_height(height=height, azimuth=azimuth, elevation=elevation)
    vec_secondary = vector_from_height(height=height*delta_height, azimuth=azimuth-delta_angle, elevation=elevation)
    return shapely.Polygon([[base_vertex[0], base_vertex[1]],
                            [base_vertex[0] + vec_primary[0], base_vertex[1] + vec_primary[1]],
                            [base_vertex[0] + vec_secondary[0], base_vertex[1] + vec_secondary[1]]])


def create_square_polygon(center: Sequence[float],
                          rotation: float = 0,
                          polygon_width: float = 10,
                          polygon_height: float = 10) -> Polygon:
    p = shapely.Polygon([[center[0] - polygon_width / 2, center[1] - polygon_height / 2],
                         [center[0] - polygon_width / 2, center[1] + polygon_height / 2],
                         [center[0] + polygon_width / 2, center[1] + polygon_height / 2],
                         [center[0] + polygon_width / 2, center[1] - polygon_height / 2]])
    if rotation:
        p = shapely.affinity.rotate(p, rotation)
    return p


def create_lshape_polygon(center: Sequence[float],
                          rotation: float = 0,
                          polygon_width: float = 10,
                          polygon_height: float = 10,
                          corner_x: float = 5,
                          corner_y: float = 5) -> Polygon:
    p = shapely.Polygon([[center[0] - polygon_width / 2, center[1] - polygon_height / 2],
                         [center[0] - polygon_width / 2, center[1] + polygon_height / 2],
                         [center[0] + polygon_width / 2, center[1] + polygon_height / 2],
                         [center[0] + polygon_width / 2, center[1] - polygon_height / 2 + corner_y],
                         [center[0] - polygon_width / 2 + corner_x, center[1] - polygon_height / 2 + corner_y],
                         [center[0] - polygon_width / 2 + corner_x, center[1] - polygon_height / 2],
                         [center[0] - polygon_width / 2, center[1] - polygon_height / 2]
                         ])
    if rotation:
        p = shapely.affinity.rotate(p, rotation)
    return p


def create_regular_polygon(center: Sequence[float],
                           rotation: float = 0,
                           radius: float = 10,
                           n_vertices: int = 8) -> Polygon:
    p = shapely.Point(center).buffer(radius, resolution=n_vertices)
    if rotation:
        p = shapely.affinity.rotate(p, rotation)
    return p


building_shapes_generators: Final[dict] = {
    Shape.RECTANGLE: create_square_polygon,
    Shape.CIRCLE: create_regular_polygon,
    Shape.LSHAPE: create_lshape_polygon
}
