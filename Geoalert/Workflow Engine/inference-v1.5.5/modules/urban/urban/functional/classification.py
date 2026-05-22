import numpy as np
from .compare import max_intersection
from gpdadapter import FeatureCollection
from typing import Dict
from ..base.defaults import BUILDING_CLASS_TAG


def classify(fc: FeatureCollection, classes_fcs: Dict[str, FeatureCollection],
             tag=BUILDING_CLASS_TAG) -> FeatureCollection:
    """
    Classify features according to classes feature collections as follows:
        for each feature find the best candidate among each collection and assign class
        of candidate with the best intersection
    Args:
        fc: FeatureCollection to classify
        classes_fcs: dict of classes GeoDataFrames, {'cls1': df1, 'cls2: df2', ...}
        tag: tag for class_id

    Returns:
        fc: FeatureCollection with tagged features (e.g. feature.properties['type_id'] -> 'cls1')
    """
    classes = list(classes_fcs.keys())
    # classify all features
    for feature_idx in range(len(fc)):
        # get score for each class according to intersection with features form each class collection
        intersections = [max_intersection(fc[feature_idx, 'geometry'], classes_fcs[cls]) for cls in classes]
        # choose class
        cls = classes[np.argmax(intersections)]
        # assign class to feature in-place
        fc[feature_idx, tag] = cls
    return fc
