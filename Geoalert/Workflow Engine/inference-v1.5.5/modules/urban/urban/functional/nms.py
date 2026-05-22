from gpdadapter import FeatureCollection
import numpy as np
from loguru import logger
from .utils.geomutils import intersection, union


def nms(fc: FeatureCollection, confidence_tag: str = 'confidence',
        predicate: str = 'intersects', buffer: float = 0,
        rel_area_thr: float = 0, iou_thr: float = 0) -> FeatureCollection:
    """Non-maximum suppression, deletes features according to the predicate and confidence property
    Args:
        fc: input FeatureCollection (will be modified inplace)
        confidence_tag: confidence property name
        predicate: spatial predicate, one of “contains”, “contains_properly”, “covered_by”, “covers”, “crosses”,
                    “intersects”, “overlaps”, “touches”, “within”, “dwithin”
        buffer: buffer to apply before calculating spatial query, in UTM
        rel_area_thr: float - threshold for each feature relative intersection area
                        (intersection_area / feature_area)
        iou_thr: float - threshold for a pair of features to consider deleting one of them
    Returns:
        filtered FeatureCollection"""
    if len(fc) < 2:
        return fc

    if confidence_tag is not None and confidence_tag not in fc.columns:
        logger.warning(f'{confidence_tag} not in fc columns, skipping NMS')
        return fc

    pairs = fc.query(fc.geometry.buffer(buffer), predicate=predicate).T  # shape (n, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # exclude intersections of every geometry with self
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)  # exclude permutations (each intersection was listed twice)
    to_drop = set()
    for idx, intersect_idx in pairs:
        intersection_area= intersection(fc[idx, 'geometry'], fc[intersect_idx, 'geometry']).area
        union_area = union(fc[idx, 'geometry'], fc[intersect_idx, 'geometry']).area
        iou = intersection_area / union_area if union_area > 0 else 0.

        if iou > iou_thr:  # consider deleting only if we satisfy IoU condition
            if confidence_tag is None:
                try:  # compare areas only for Polygons,
                    if fc[idx, 'geometry'].area < fc[intersect_idx, 'geometry'].area:
                        to_drop.add(idx)
                    else:
                        to_drop.add(intersect_idx)
                except AttributeError:  # non-Polygons
                    pass
            else:
                if fc[idx, confidence_tag] < fc[intersect_idx, confidence_tag]:
                    feature_area = fc[idx, 'geometry'].area
                    rel_area = intersection_area / feature_area if feature_area > 0 else 0
                    if rel_area > rel_area_thr:  # check relative intersection area
                        to_drop.add(idx)

                elif fc[idx, confidence_tag] > fc[intersect_idx, confidence_tag]:
                    feature_area = fc[intersect_idx, 'geometry'].area
                    rel_area = intersection_area / feature_area if feature_area > 0 else 0
                    if rel_area > rel_area_thr:  # check relative intersection area
                        to_drop.add(intersect_idx)

                else:  # equal confidence
                    try:  # compare areas only for Polygons,
                        if fc[idx, 'geometry'].area < fc[intersect_idx, 'geometry'].area:
                            to_drop.add(idx)
                        else:
                            to_drop.add(intersect_idx)
                    except AttributeError:  # non-Polygons
                        pass
    return fc.drop(list(to_drop))
