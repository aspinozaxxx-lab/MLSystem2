import numpy as np
import shapely
import shapely.affinity as sa
import shapely.geometry as sg
import sklearn
from sklearn.cluster import KMeans
from typing import Union, List
from .. import shapely_ext as se
from ..constants import Shape
from ._simplified_geometry import SimplifiedGeometry, PolygonType
import warnings
from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sklearn.cluster._kmeans.ConvergenceWarning)


def _check_l_shape_coords(c1, c2, c3, min_width_abs=5., min_width_rel=0.25):
    """Check that L-shape figure have either 25% minimal width of L-part or 5 meters"""
    is_valid = False

    if (
            (
                    abs(c2 - c1) > min_width_abs and
                    abs(c3 - c2) > min_width_abs
            ) or
            min_width_rel < abs((c2 - c1) / abs(c3 - c1)) < (1 - min_width_rel)
    ):
        is_valid = True

    return is_valid


def _get_three_peaks(coordinates: List[float]) -> List[float]:
    data = np.array(coordinates).reshape(-1, 1)  # for kmeans
    clusters = KMeans(n_clusters=3, n_init=3, init='random', algorithm='lloyd').fit_predict(data)

    min_c = min(coordinates)
    max_c = max(coordinates)
    interval = (max_c - min_c) / 5

    # correct with interval
    min_c = np.median([c for c in coordinates if c - interval < min_c])
    max_c = np.median([c for c in coordinates if c + interval > max_c])

    mean_c = sorted([np.median(data[clusters == i]) for i in range(3)])[1]
    return [min_c, mean_c, max_c]


# def _get_three_peaks(coordinates: List[float]) -> List[float]:
#     # find min and max points
#     min_c = min(coordinates)
#     max_c = max(coordinates)
#     interval = (max_c - min_c) / 5
#
#     # correct with interval
#     min_c = np.median([c for c in coordinates if c - interval < min_c])
#     max_c = np.median([c for c in coordinates if c + interval > max_c])
#
#     # find center point
#     # we get all points between min and max coordinates
#     # and center point should be at least 5 meters far from each or them or
#     # lay in 0.1 - 0.9 relative length between points
#     width = min((max_c - min_c) / 10, 5.)
#     interval_min = min_c + width
#     interval_max = max_c - width
#
#     try:
#         center_coords = [c for c in coordinates if interval_min < c < interval_max]
#         step = min((interval_max - interval_min) / 10, 2.)
#         n_steps = int(round((interval_max - interval_min) / step))
#         hist, bin_edges = np.histogram(center_coords, bins=n_steps)
#         smooth_hist = [hist[i - 1] + hist[i] + hist[i + 1] for i in range(1, len(hist) - 2)]
#         peak = np.argmax(smooth_hist) + 1
#         edge_left = bin_edges[peak - 1]
#         edge_right = bin_edges[peak + 1]
#         mean_c = np.median([c for c in center_coords if edge_left < c < edge_right])
#     except Exception as e:
#         print(e)
#         mean_c = min_c + (max_c - min_c) / 2
#
#     return [min_c, mean_c, max_c]


def _maybe_intersection_area(g1: shapely.geometry.Polygon, g2: shapely.geometry.Polygon) -> float:
    # TODO: yet another IoU
    try:
        return g1.intersection(g2).area
    except shapely.errors.TopologicalError:
        return 0.


def _get_l_shape_coordinates(polygon: Union[sg.Polygon, sg.MultiPolygon]) -> List[float]:
    """Cluster points for each axis to 3 clusters, and return median point of each cluster as proposed center"""
    polygon = polygon.simplify(1.)
    points = set(se.split_to_points(polygon))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return list(_get_three_peaks(xs)) + list(_get_three_peaks(ys))


def _get_l_shape_position(polygon: Union[sg.Polygon, sg.MultiPolygon]) -> float:
    """Return the position of empty space in L-shape geometry"""
    rectangle = polygon.envelope
    parts = se.split_rectangle_to_four_parts(rectangle)
    areas = [_maybe_intersection_area(p, polygon) for p in parts]
    return np.argmin(areas)


def _get_l_shape(polygon: Union[sg.Polygon, sg.MultiPolygon], angle: float) -> Union[sg.Polygon, None]:
    # get rotation point
    centroid = polygon.centroid

    # rotate to position with edges orthogonal to axes
    polygon = sa.rotate(polygon, - angle, origin=centroid)

    # correct position, empty space of L-shape should be closer ot (0, 0) coordinates
    l_shape_pos = _get_l_shape_position(polygon)
    correction_angle = l_shape_pos * 90
    polygon = sa.rotate(polygon, correction_angle, origin=centroid)

    # generate L-shape geometry
    try:
        coordinates = _get_l_shape_coordinates(polygon)
    except Exception as e:
        logger.warning(f'Failed L-Shape simplification: {str(e)}')
        return None

    # check l-shape coordinates for validity
    is_valid_x = _check_l_shape_coords(*coordinates[:3])
    is_valid_y = _check_l_shape_coords(*coordinates[3:])

    if not is_valid_x or not is_valid_y:
        l_shape_geometry = None

    else:
        l_shape_geometry = se.make_l_shape(*coordinates)

        # rotate it back to start angle
        l_shape_geometry = shapely.affinity.rotate(
            l_shape_geometry, angle - correction_angle, origin=centroid
        )

    return l_shape_geometry


def simplify_geometry_with_l_shape(geometry: PolygonType, **kwargs) -> SimplifiedGeometry:
    if kwargs:
        logger.warning(f'{list(kwargs.keys())} are no loger supported in "simplify_geometry_with_l_shape"')


    if not geometry or geometry.is_empty:
        return SimplifiedGeometry(
            origin_geometry=geometry,
            simple_geometry=geometry,
            simple_geometry_type=Shape.UNKNOWN,
            iou=1,
        )

    angles = se.get_angles_for_polygon(geometry.simplify(0.5))

    best_shape = None
    best_angle = None
    best_iou = - 1

    # chose best angle from top 3 geometry angles
    for angle in angles[:3]:
        l_shape = _get_l_shape(geometry, angle)
        if l_shape is None:
            continue
        iou = se.intersection_over_union(geometry, l_shape, ignore_errors=True)

        if iou > best_iou:
            best_iou = iou
            best_shape = l_shape
            best_angle = angle

    if best_iou > 0.7:  # speed up constant
        # vary angle to find best shape
        best_correction = 0
        for angle_corr in range(-10, 11, 1):
            l_shape = _get_l_shape(geometry, best_angle + angle_corr)
            if l_shape is None:
                continue
            iou = se.intersection_over_union(geometry, l_shape)

            if iou > best_iou:
                best_iou = iou
                best_shape = l_shape
                best_correction = angle_corr

        best_angle += best_correction

    simplified_geometry = SimplifiedGeometry(
        origin_geometry=geometry,
        simple_geometry=best_shape if best_shape is not None else geometry,
        simple_geometry_type=Shape.LSHAPE,
        iou=best_iou,
    )

    return simplified_geometry
