from gpdadapter import FeatureCollection
from urban.functional.utils.geomutils import calculate_iou
from typing import Sequence, Tuple
from loguru import logger


def validate_collection(pred: FeatureCollection,
                        gt: FeatureCollection,
                        iou_thr: float = 0.9,
                        categorical_properties: Sequence[str] = (),
                        numerical_properties: Sequence[Tuple[str, float]] = (),
                        verbose: bool = True) -> bool:
    """Checks that each feature in gt has one and only one corresponding feature in pred with the same geometry
    (by iou threshold) e.g. collections are practically the same
    Args:
        pred: FeatureCollection to validate
        gt: ground truth FeatureCollection
        iou_thr: threshold for IoU to consider features from pred and gt the same
        categorical_properties: sequence of feature properties to check for exact match
        numerical_properties: sequence of feature properties to check for being equal within tolerance as pairs
                              (property_name, tolerance)
        verbose: verbose
    """
    if gt.empty:
        if pred.empty:
            logger.info('Both collections are empty!')
            return True
        else:
            logger.info(f'gt collection is empty while pred has {len(pred)} features!')
            return False
    if not pred.crs == gt.crs:
        pred.to_crs(gt.crs, inplace=True)

    checked_gt = set()
    checked_pred = set()
    properties_ok = True

    # The first subarray contains input geometry integer indices.
    # The second subarray contains tree geometry integer indices.
    for pred_idx, gt_idx in gt.query(pred).T:
        # geometry
        if (calculate_iou(pred[pred_idx, 'geometry'], gt[gt_idx, 'geometry']) > iou_thr and
                pred_idx not in checked_pred and
                gt_idx not in checked_gt):
            checked_gt.add(gt_idx)
            checked_pred.add(pred_idx)

            # categorical_properties
            for property_name in categorical_properties:
                if pred[pred_idx, property_name] != gt[gt_idx, property_name]:
                    logger.info(f'Found non-matching categorical property: pred[{pred_idx},{property_name}] = '
                                f'{pred[pred_idx, property_name]} != gt[{gt_idx}, {property_name}] ='
                                f' {gt[gt_idx, property_name]}')
                    properties_ok = False

            # numerical_properties
            for property_name, tolerance in numerical_properties:
                if abs(pred[pred_idx, property_name] - gt[gt_idx, property_name]) > tolerance:
                    logger.info(f'Found non-equal numerical property: pred[{pred_idx},{property_name}] = '
                                f'{pred[pred_idx, property_name]} != gt[{gt_idx}, {property_name}] = '
                                f'{gt[gt_idx, property_name]}')
                    properties_ok = False

    if not (len(checked_gt) == len(checked_pred) == len(gt) == len(pred)):
        logger.info(f'geometry validation failed: {len(gt)} features in gt, {len(pred)} features in pred,'
                    f' {len(checked_pred)} intersecting features')
    return (len(checked_gt) == len(checked_pred) == len(gt)) and properties_ok

