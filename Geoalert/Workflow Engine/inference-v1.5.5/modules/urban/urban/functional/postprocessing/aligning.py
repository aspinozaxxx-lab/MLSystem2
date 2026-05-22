import math
import pandas as pd
import shapely
import shapely.geometry
import shapely.affinity
import numpy as np
from gpdadapter import FeatureCollection
from loguru import logger
from typing import Union, List, Tuple
from . import shapely_ext as se
from .constants import Tag, Shape


# -------------------------------------------------------------------------------
# Utility functions to work with geometries
# -------------------------------------------------------------------------------

def is_rectangular(feature: shapely.geometry.Polygon, shape_type: str = Shape.UNKNOWN):  # TODO: do we use it anywhere?
    """Check if polygon is rectangle (naive approach with number of vertices)"""
    n_vertices = len(se.split_to_points(feature))
    if shape_type in [Shape.LSHAPE, Shape.RECTANGLE] or n_vertices == 5:
        return True
    return False


def get_polygon_angle(polygon: shapely.geometry.Polygon):
    """Return angle of the longest line of Polygon"""
    try:
        angles = se.get_angles_for_polygon(polygon)
        angle = angles[0]  # longest line
    except Exception as e:
        print(type(polygon), polygon.wkt)
        logger.exception(e)
        raise e
    return angle


def get_rotation_angles_for_rectangles(angles: List[float], repeat: bool = True, filter_outliers: bool = True):
    """Return minimum rotation angles for rectangles to mean cluster angle"""
    rectangle_angles = [a % 90 for a in angles]
    if any([a > 70 for a in rectangle_angles]) and any([a < 20 for a in rectangle_angles]) and repeat:
        rotation_angles = get_rotation_angles_for_rectangles(
            [a + 20 for a in angles],
            repeat=False,
            filter_outliers=filter_outliers,
        )
        return rotation_angles
    mean_angle = np.mean(rectangle_angles)
    rotation_angles = [mean_angle - c_angle for c_angle in rectangle_angles]

    if filter_outliers:
        mask = get_not_outliers_mask(rotation_angles)
        mean_angle = np.mean(np.array(rectangle_angles)[mask])
        rotation_angles = [mean_angle - c_angle for c_angle in rectangle_angles]

    return rotation_angles


def get_min_angle_between_two_lines(angle_line_1: float, angle_line_2: float):  # TODO: same, as in angleutils
    """Return minimum angle between two lines, angles should be in range 0..90 degrees"""
    if not(0 <= angle_line_1 <= 90):
        logger.warning(f'angle_line_1 must be in [0, 90] got {angle_line_1}')
    if not(0 <= angle_line_2 <= 90):
        logger.warning(f'angle_line_1 must be in [0, 90] got {angle_line_2}')

    sign = -1 if angle_line_2 > angle_line_1 else 1
    abs_diff = abs(angle_line_1 - angle_line_2)
    if abs_diff >= 45:
        abs_diff = 90 - abs_diff
        sign *= -1
    return sign * abs_diff


def get_not_outliers_mask(x: List[float]):
    x = np.array(x)
    q1 = np.quantile(x, 0.1)
    q2 = np.quantile(x, 0.9)
    mask = (x > q1) * (x < q2)
    return mask


def filter_outliers(x: List[float], min_length: float = 4) -> List[float]:
    if len(x) >= min_length:
        x = np.array(x)
        mask = get_not_outliers_mask(x)
        x = list(x[mask])

    return x


# -------------------------------------------------------------------------------
# Main functions for aligning to roads
# -------------------------------------------------------------------------------

def assign_roads_to_buildings(
        buildings_fc: FeatureCollection,
        roads_fc: FeatureCollection,
        road_buffer_size: Union[int, float] = 20,
        cluster_id_tag: str = Tag.BLD_ROAD_ID,
        road_angle_tag: str = Tag.BLD_ROAD_ANGLE,
) -> FeatureCollection:

    buildings_fc[:, road_angle_tag] = pd.Series(dtype=float)
    buildings_fc[:, cluster_id_tag] = pd.Series(dtype=int)

    if buildings_fc.empty or roads_fc.empty:
        return buildings_fc
    if buildings_fc.crs != roads_fc.crs:
        logger.warning('CRS mismatch in assign_roads_to_buildings()')

    # prepare roads by disassembling them to separate lines
    road_lines = []
    for idx in range(len(roads_fc)):
        road_lines.extend(se.split.split_to_lines(roads_fc[idx, 'geometry']))
    road_lines = sorted(road_lines, key=lambda x: x.length, reverse=True)

    # iterate over each road line and group buildings around it in range of `road_buffer_size`
    for road_cluster_id, road_line in enumerate(road_lines):

        road_angle = se.get_angle_for_line(road_line)
        # road_buffered_area = strip_line(road_line, road_buffer_size).buffer(road_buffer_size)
        # TODO: check strip value
        road_buffered_area = se.strip_line(road_line, 0).buffer(road_buffer_size)
        nearest_buildings_indexes = buildings_fc.query(road_buffered_area)

        # iterate over each building and compute it properties
        for idx in nearest_buildings_indexes:
            # compute building angle (angle of the longest line in polygon)
            building_angle = get_polygon_angle(buildings_fc[idx, 'geometry'])

            # get aligning to road angle if it is already assigned
            previous_road_angle = buildings_fc[idx, road_angle_tag]
            previous_aligning_angle = get_min_angle_between_two_lines(
                previous_road_angle % 90,
                building_angle % 90,
            ) if not np.isnan(previous_road_angle) else math.inf

            # compute aligning angle to current road line
            new_aligning_angle = get_min_angle_between_two_lines(road_angle % 90, building_angle % 90)

            # reassign road if angle to new road is lower
            if abs(new_aligning_angle) < abs(previous_aligning_angle):
                buildings_fc[idx, cluster_id_tag] = road_cluster_id
                buildings_fc[idx, road_angle_tag] = round(road_angle, 2)
    return buildings_fc


def _create_cluster_with_roads(
        parent_building_idx,
        buildings_fc: FeatureCollection,
        cluster_distance: float,
        cluster_id_tag: str = Tag.BLD_ROAD_ID,
        road_angle_tag: str = Tag.BLD_ROAD_ANGLE,
):
    # recursively select all nearest features (graph depth-first search)
    nearest_buildings_idxs = buildings_fc.query(
        buildings_fc[parent_building_idx, 'geometry'].buffer(cluster_distance))

    for child_building_idx in nearest_buildings_idxs:
        # TODO: refactor/optimize
        # same feature
        if child_building_idx == parent_building_idx:
            continue

        # filter features with from same cluster
        if buildings_fc[parent_building_idx, cluster_id_tag] == buildings_fc[child_building_idx, cluster_id_tag] and \
           not np.isnan(buildings_fc[parent_building_idx, cluster_id_tag]):
            continue

        # algorithm for features from other clusters
        # compare rotation angle to current road and new one (from incoming parent feature)
        if not np.isnan(buildings_fc[child_building_idx, cluster_id_tag]):

            child_building_angle = get_polygon_angle(buildings_fc[child_building_idx, 'geometry'])

            child_road_angle = buildings_fc[child_building_idx, road_angle_tag]
            parent_road_angle = buildings_fc[parent_building_idx, road_angle_tag]

            current_aligning_angle = get_min_angle_between_two_lines(
                child_building_angle % 90,
                child_road_angle % 90,
            )
            new_aligning_angle = get_min_angle_between_two_lines(
                child_building_angle % 90,
                parent_road_angle % 90,
            )

            if abs(current_aligning_angle) > abs(new_aligning_angle):
                buildings_fc[child_building_idx, cluster_id_tag] = buildings_fc[parent_building_idx, cluster_id_tag]
                buildings_fc[child_building_idx, road_angle_tag] = buildings_fc[parent_building_idx, road_angle_tag]
            else:
                continue

        # assign cluster id for all non-clustered features
        else:
            buildings_fc[child_building_idx, cluster_id_tag] = buildings_fc[parent_building_idx, cluster_id_tag]
            buildings_fc[child_building_idx, road_angle_tag] = buildings_fc[parent_building_idx, road_angle_tag]

        _create_cluster_with_roads(
            parent_building_idx=child_building_idx,
            buildings_fc=buildings_fc,
            cluster_distance=cluster_distance,
            cluster_id_tag=cluster_id_tag,
            road_angle_tag=road_angle_tag,
        )


def expand_roads_clusters(
        buildings_fc: FeatureCollection,
        expand_distance: float = 5.,
        cluster_id_tag: str = Tag.BLD_ROAD_ID,
        road_angle_tag: str = Tag.BLD_ROAD_ANGLE,
) -> FeatureCollection:
    """Expand clusters"""
    for parent_building_idx in range(len(buildings_fc)):
        if not np.isnan(buildings_fc[parent_building_idx, cluster_id_tag]):
            _create_cluster_with_roads(
                parent_building_idx=parent_building_idx,
                buildings_fc=buildings_fc,
                cluster_distance=expand_distance,
                cluster_id_tag=cluster_id_tag,
                road_angle_tag=road_angle_tag,
            )
    return buildings_fc


def compute_rotation_to_road_angles(
        buildings_fc: FeatureCollection,
        cluster_id_tag: str = Tag.BLD_ROAD_ID,
        road_angle_tag: str = Tag.BLD_ROAD_ANGLE,
        rotation_angle_tag: str = Tag.BLD_ROTATION_ANGLE,
        max_mean_angle_diff: float = 10,
) -> FeatureCollection:
    # assemble clusters
    buildings_fc[:, rotation_angle_tag] = pd.Series(dtype=float)
    cluster_indexes_groups = buildings_fc.groupby_indexes(cluster_id_tag)

    # compute rotation angles in clusters
    for cluster_id, cluster_indexes in cluster_indexes_groups.items():
        # TODO: refactor/optimize
        building_angles = [get_polygon_angle(bld) for bld in buildings_fc[cluster_indexes].geometry]
        road_angles = buildings_fc[cluster_indexes, road_angle_tag]
        rotation_to_road_angles = [get_min_angle_between_two_lines(x1 % 90, x2 % 90)
                                   for x1, x2 in zip(building_angles, road_angles)]

        rotation_to_road_angles_filtered = filter_outliers(rotation_to_road_angles, min_length=4)
        if abs(np.mean(rotation_to_road_angles_filtered)) < max_mean_angle_diff:
            buildings_fc[cluster_indexes, rotation_angle_tag] = rotation_to_road_angles
    return buildings_fc

# -------------------------------------------------------------------------------
# Main functions for in cluster aligning
# -------------------------------------------------------------------------------

# TODO: Use conventional 2d clustering algorithm


def _create_distance_cluster(
        parent_building_idx,
        buildings_fc: FeatureCollection,
        cluster_distance: float,
        cluster_id_tag: str = Tag.BLD_CLUSTER_ID,
):
    # recursively select all nearest features (graph depth-first search)
    nearest_buildings_idxs = buildings_fc.query(
        buildings_fc[parent_building_idx, 'geometry'].buffer(cluster_distance))

    for child_building_idx in nearest_buildings_idxs:
        # TODO: refactor/optimize
        if child_building_idx == parent_building_idx:
            continue
        if buildings_fc[parent_building_idx, cluster_id_tag] == buildings_fc[child_building_idx, cluster_id_tag] and \
           not np.isnan(buildings_fc[parent_building_idx, cluster_id_tag]):
            continue

        buildings_fc[child_building_idx, cluster_id_tag] = buildings_fc[parent_building_idx, cluster_id_tag]
        _create_distance_cluster(
            parent_building_idx=child_building_idx,
            buildings_fc=buildings_fc,
            cluster_distance=cluster_distance,
            cluster_id_tag=cluster_id_tag,
        )


def create_distance_clusters(
        buildings_fc: FeatureCollection,
        cluster_distance: float,
        cluster_id_tag: str = Tag.BLD_CLUSTER_ID,
        ignore_tag: str = Tag.BLD_ROTATION_ANGLE,
) -> FeatureCollection:
    buildings_fc[:, cluster_id_tag] = pd.Series(dtype=int)
    if ignore_tag not in buildings_fc.columns:
        buildings_fc[:, ignore_tag] = pd.Series(dtype=float)
    for idx in range(len(buildings_fc)):
        if np.isnan(buildings_fc[idx, cluster_id_tag]) and np.isnan(buildings_fc[idx, ignore_tag]):
            buildings_fc[idx, cluster_id_tag] = idx
            _create_distance_cluster(
                parent_building_idx=idx,
                buildings_fc=buildings_fc,
                cluster_distance=cluster_distance,
                cluster_id_tag=cluster_id_tag,
            )
    return buildings_fc


def compute_rotation_in_cluster_angles(
        buildings_fc: FeatureCollection,
        cluster_id_tag: str = Tag.BLD_CLUSTER_ID,
        rotation_angle_tag: str = Tag.BLD_ROTATION_ANGLE,
        max_mean_angle_diff: float = 10,
) -> FeatureCollection:
    # assemble clusters
    clusters = buildings_fc.groupby_indexes(cluster_id_tag)

    # compute rotation angles in clusters
    for cluster_id, cluster_indexes in clusters.items():
        if len(cluster_indexes) > 1:
            building_angles = [get_polygon_angle(bld) for bld in buildings_fc[cluster_indexes].geometry]
            rotation_angles = get_rotation_angles_for_rectangles(building_angles, repeat=True, filter_outliers=True)

            if np.mean(np.abs(rotation_angles)) < max_mean_angle_diff:
                buildings_fc[cluster_indexes, rotation_angle_tag] = -np.array(rotation_angles)
    return buildings_fc


def rotate_buildings(
        buildings_fc: FeatureCollection,
        rotation_angle_tag: str = Tag.BLD_ROTATION_ANGLE,
        is_rotated_tag: str = Tag.BLD_IS_ROTATED,
) -> FeatureCollection:
    """Apply rotation to buildings if rotation angle is provided in tags"""
    buildings_fc[:, is_rotated_tag] = False
    for idx in range(len(buildings_fc)):
        if not np.isnan(buildings_fc[idx, rotation_angle_tag]):
            buildings_fc[idx, 'geometry'] = shapely.affinity.rotate(buildings_fc[idx, 'geometry'],
                                                                    - buildings_fc[idx, rotation_angle_tag])
            buildings_fc[idx, is_rotated_tag] = True
    return buildings_fc


def align_buildings(
        buildings_fc: FeatureCollection,
        roads_fc: FeatureCollection,
        max_area: float = 600.,
        cluster_distance: float = 6.,
        shape_types: Tuple[str] = (Shape.GRID_SNAP, Shape.LSHAPE, Shape.RECTANGLE),
):
    """Align building with nearby roads or to cluster mean angle

    Args:
        buildings_fc: FeatureCollection of buildings polygons
        roads_fc: FeatureCollection of roads LineStrings
        max_area: maximum area of feature to apply aligning
        cluster_distance: maximum distance between buildings to assign in one cluster
        shape_types: shape types for building alignment (other ignored)

    Returns:
        aligned_fc: ds.FeatureCollection
    """
    if buildings_fc.empty:
        return buildings_fc
    if roads_fc.empty:
        logger.warning('No roads to align buildings with')
        return buildings_fc

    if Tag.BLD_SHAPE_TYPE not in buildings_fc.columns:
        logger.warning('Can not to align buildings before simplification')
        return buildings_fc

    for_alignment_idxs = buildings_fc[:, Tag.BLD_SHAPE_TYPE].map(lambda x: x in shape_types) &\
                         buildings_fc.geometry.area < max_area
    features_for_alignment = buildings_fc[for_alignment_idxs]
    other_features = buildings_fc[~for_alignment_idxs]
    # road aligning
    features_for_alignment = assign_roads_to_buildings(features_for_alignment, roads_fc)
    features_for_alignment = expand_roads_clusters(features_for_alignment, cluster_distance)
    features_for_alignment = compute_rotation_to_road_angles(features_for_alignment, max_mean_angle_diff=10.)

    # distance aligning
    features_for_alignment = create_distance_clusters(features_for_alignment, cluster_distance=cluster_distance)
    features_for_alignment = compute_rotation_in_cluster_angles(features_for_alignment, max_mean_angle_diff=10)

    # rotate building with computed angles
    features_for_alignment = rotate_buildings(features_for_alignment)
    other_features.append(features_for_alignment)

    return other_features
