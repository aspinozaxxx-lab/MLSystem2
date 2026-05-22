import numpy as np
from typing import Sequence
from shapely.geometry.base import BaseGeometry
from gpdadapter import FeatureCollection


def union(features: Sequence[BaseGeometry]):
    """
    Union list of features to a single geometry
    Args:
        features: list of `Feature`/shape

    Returns:
        shape: shapely.geometry
    """
    if len(features) == 1:
        return features[0]

    union_feature = features[0]
    for f in features[1:]:
        union_feature = union_feature.union(f)
    return union_feature


def iou(first: BaseGeometry, second: BaseGeometry):  # TODO: it is defined in multiple places
    """
    Calculate  intersection over union between two features/geometries/shapes
    Args:
        first: geometry/shape
        second: geometry/shape

    Returns:
        score: float number in range 0..1
    """
    intersection = first.intersection(second)
    union = first.union(second)
    return intersection.area / (union.area + 10e-18)


def greater(a, thresh):  # TODO: we never use it
    """
    Return True if any of array numbers is greater than given thresh
    Args:
        a: list of numbers
        thresh: number

    Returns:
        result: bool
    """
    return (np.array(a) > thresh).any()


def complex_iou(feature, other_features):
    """
    Calculate IoU over one feature and the whole list of other features
    Args:
        feature: geometry/shape
        other_features: list of geometry/shape

    Returns:
        score: float number in range 0 .. 1
    """
    union_feature = union(other_features)
    return iou(feature, union_feature)


def max_intersection(feature: BaseGeometry, fc: FeatureCollection) -> float:
    """
    Find maximum intersection along all fc and return area of intersection.
    Args:
        feature: Feature of shapely geometry object
        fc: list of features/FeatureCollection

    Returns:
        float max area
    """
    max_area = 0
    feature_area = feature.area
    if feature_area == 0:
        return 0
    intersection_indexes = fc.query(feature)

    for idx in intersection_indexes:
        a = fc[idx, 'geometry'].area
        if a >= max_area:
            max_area = a
            if max_area / feature_area > 0.5:
                break
    return max_area
