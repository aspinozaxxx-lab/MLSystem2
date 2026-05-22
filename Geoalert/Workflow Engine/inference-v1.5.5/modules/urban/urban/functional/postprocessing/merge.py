from .shapely_ext import intersection_over_union
from gpdadapter import FeatureCollection
import numpy as np
from . import shapely_ext as se
from shapely import Polygon, MultiPolygon
from typing import Optional, Tuple, List, Sequence, Union


def find_best_features(feature: Union[Polygon, MultiPolygon],
                       other_features: Sequence[Union[Polygon, MultiPolygon]]) -> Tuple[float,
                                                                                        List[Union[Polygon,
                                                                                                   MultiPolygon]]]:
    """
    Look for best combination of `other features` to have max IoU with given `feature`
    Args:
        feature: FeatureCollection with a single row
        other_features: FeatureCollection

    Returns:
        score: float, between 0 .. 1, max IoU
        best_features: list of features that give best IoU with given `feature`
    """
    max_iou = 0
    best_features = list()

    # add features one by one if IoU increasing
    for f in other_features:

        proposed_features = best_features + [f]
        proposed_iou = se.complex_intersection_over_union(geometries1=feature, geometries2=proposed_features)
        if proposed_iou > max_iou:
            best_features = best_features + [f]
            max_iou = proposed_iou

    # do the same but throw out first feature
    if len(best_features) > 1:
        proposed_features = best_features[1:]
        proposed_iou = se.complex_intersection_over_union(geometries1=feature, geometries2=proposed_features)
        if proposed_iou > max_iou:
            best_features = proposed_features
            max_iou = proposed_iou

    return max_iou, best_features


def replace_with_best_suitable(fc1: FeatureCollection,
                               fc2: FeatureCollection,
                               min_iou: float = 0.7,
                               tag: Optional[str] = None,
                               value_if_input=None,
                               value_if_prior=None,
                               attributes: Optional[Sequence[str]]=None) -> FeatureCollection:
    """
    Replace features from fc_1 by features from fc_2 if they match (IoU > replace_threshold)
    Args:
        fc1: predicted features
        fc2: prior features
        min_iou: float number, features from fc_1 that have IoU with features from fc_2
            above this value will be replaced
        tag: add bool `tag` to properties (replaced/not replaced)
        value_if_input: value on `tag` attribute if the feature is from 'input' fc
        value_if_prior: value on `tag` attribute if the feature is from 'prior' fc
        attributes: list of attributes to use from fc_2 features, Optional

    Returns:
        fc: merged feature collection
    """
    if fc1.empty:
        return fc2
    if fc2.empty:
        return fc1
    # validate input
    assert fc1.crs == fc2.crs

    if tag is not None and tag not in fc1.columns:
        fc1[:, tag] = value_if_input  # create column 'tag' with default values

    fc2.explode(inplace=True)  # There are multipolygons in OSM
    # first subarray contains input geometry indices, second subarray contains tree geometry indices
    intersections = fc1.query(fc2)
    for idx1 in range(len(fc1)):
        # all the geometries in fc_2 intersecting with this one
        candidates = intersections[0, intersections[1] == idx1]
        if len(candidates) == 0:
            continue
        IoUs = [intersection_over_union(fc1[idx1, 'geometry'], fc2[idx2, 'geometry'])
                for idx2 in candidates]
        if max(IoUs) < min_iou:
            continue
        best_candidate = candidates[np.argmax(IoUs)]
        fc1[idx1, 'geometry'] = fc2[best_candidate, 'geometry']
        if attributes:
            for attr in attributes:
                fc1[idx1, attr] = fc2[best_candidate, attr]
        if tag is not None:
            fc1[idx1, tag] = value_if_prior

    return fc1
