from gpdadapter import FeatureCollection
from shapely import Polygon
import numpy as np
EPS = 10e-14


def distance(f1: Polygon, f2: Polygon):
    """
    Distance between two features centroids
    Args:
        f1: shapely.Polygon
        f2: shapely.Polygon

    Returns:
        distance, float
    """
    return f1.centroid.distance(f2.centroid)


def iou(f1: Polygon, f2: Polygon):
    """
    Intersection over union between two features
    Args:
        f1: shapely.Polygon
        f2: shapely.Polygon

    Returns:
        IoU, float (0..1)
    """
    return f1.intersection(f2).area / (f1.union(f2).area + EPS)


def coverage(f1: Polygon, f2: Polygon):
    """
    Rate of coverage feature "1" by feature "2"
    Args:
        f1: shapely.Polygon
        f2: shapely.Polygon

    Returns:
        rate of coverage, float (0..1)
    """
    return f1.intersection(f2).area / (f1.area + EPS)


def detect_changes(pre_fc: FeatureCollection,
                   post_fc: FeatureCollection,
                   metric: str = 'cover',
                   threshold: float = 0.4) -> FeatureCollection:
    """
    Compare two feature collections and detect disappeared objects.
    Returns features of pre_fc that have changed in post_fc
    Args:
        pre_fc: first FeatureCollection
        post_fc: second FeatureCollection
        metric: one of
            'cover' - compare features by coverage rate. If features from second
                collection cover feature of first collection in total more than
                threshold feature is considered as NOT disappeared.
            'iou' - compare features with intersection over union.
                If any of features from second collection has IoU > threshold
                feature is considered as NOT disappeared.
            'distance' - compare features with distance between centroids.
                If any feature from second collection has distance less than
                threshold to feature from first collection feature is considered
                as NOT disappeared.
        threshold: value to compare metric with

    Returns:
        changes: FeatureCollection
    """
    if pre_fc.empty and post_fc.empty:
        return pre_fc
    if pre_fc.empty:
        return pre_fc  # TODO: or return post_fc here?
    if post_fc.empty:
        return pre_fc
    assert pre_fc.crs == post_fc.crs

    # indexes shape is (2, n)
    # The first subarray contains input geometry integer indices.
    # The second subarray contains tree geometry integer indices.
    indexes = post_fc.query(pre_fc.geometry.buffer(4))
    if metric == 'cover':
        metrics = np.array([coverage(pre_fc[i, 'geometry'], post_fc[j, 'geometry']) for i, j in indexes])
        changed_indexes = metrics[0, np.where(metrics < threshold)]  # this is indexes in pre_fc

    elif metric == 'iou':
        metrics = np.array([iou(pre_fc[i, 'geometry'], post_fc[j, 'geometry']) for i, j in indexes])
        changed_indexes = metrics[0, np.where(metrics < threshold)]  # this is indexes in pre_fc

    elif metric == 'distance':
        metrics = np.array([distance(pre_fc[i, 'geometry'], post_fc[j, 'geometry']) for i, j in indexes])
        changed_indexes = metrics[0, np.where(metrics > threshold)]  # this is indexes in pre_fc
    else:
        raise ValueError(f'metric must be one of "cover", "iou", "distance", got {metric}')

    return pre_fc[changed_indexes]
