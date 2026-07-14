"""Расчёт объектовой F1 по маскам одного тайла."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .contracts import MetricsError, ObjectF1Request, ObjectF1Result


def compute_object_f1(request: ObjectF1Request) -> ObjectF1Result:
    true_instances = np.asarray(request.y_true_instances, dtype=np.int64)
    predicted_mask = np.asarray(request.y_pred_mask, dtype=bool)
    if true_instances.shape != predicted_mask.shape or true_instances.ndim != 2:
        raise MetricsError(
            "y_true_instances и y_pred_mask должны быть двумерными массивами одинаковой формы"
        )
    predicted_instances, predicted_count = ndimage.label(
        predicted_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    true_ids = np.unique(true_instances)
    true_ids = true_ids[true_ids > 0]
    if true_ids.size == 0 or predicted_count == 0:
        return _result(
            true_positive=0,
            false_positive=int(predicted_count),
            false_negative=int(true_ids.size),
        )

    predicted_ids = np.arange(1, predicted_count + 1, dtype=np.int64)
    adjacency = np.zeros((true_ids.size, predicted_ids.size), dtype=np.uint8)
    predicted_areas = np.bincount(predicted_instances.ravel(), minlength=predicted_count + 1)
    for true_position, true_id in enumerate(true_ids):
        true_mask = true_instances == true_id
        true_area = int(np.count_nonzero(true_mask))
        overlaps = np.bincount(
            predicted_instances[true_mask],
            minlength=predicted_count + 1,
        )
        for predicted_id in np.flatnonzero(overlaps[1:]) + 1:
            intersection = int(overlaps[predicted_id])
            union = true_area + int(predicted_areas[predicted_id]) - intersection
            iou = intersection / union if union else 0.0
            if iou >= request.iou_threshold:
                adjacency[true_position, predicted_id - 1] = 1

    matching = maximum_bipartite_matching(csr_matrix(adjacency), perm_type="column")
    true_positive = int(np.count_nonzero(matching >= 0))
    return _result(
        true_positive=true_positive,
        false_positive=int(predicted_count) - true_positive,
        false_negative=int(true_ids.size) - true_positive,
    )


def _result(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> ObjectF1Result:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    denominator = precision + recall
    return ObjectF1Result(
        precision=precision,
        recall=recall,
        f1=2.0 * precision * recall / denominator if denominator else 0.0,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
