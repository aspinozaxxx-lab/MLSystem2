"""Расчёт объектовой F1 по маскам одного тайла."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .contracts import MetricsError, ObjectF1Request, ObjectF1Result


def compute_object_f1(request: ObjectF1Request) -> ObjectF1Result:
    true_instances = np.asarray(request.y_true_instances, dtype=np.int64)
    if request.y_pred_instances is not None:
        predicted_instances = np.asarray(request.y_pred_instances, dtype=np.int64)
        if np.any(predicted_instances < 0):
            raise MetricsError("y_pred_instances не может содержать отрицательные идентификаторы")
    else:
        predicted_mask = np.asarray(request.y_pred_mask, dtype=bool)
        predicted_instances, _ = ndimage.label(
            predicted_mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
    if true_instances.shape != predicted_instances.shape or true_instances.ndim != 2:
        raise MetricsError(
            "y_true_instances и предсказание должны быть двумерными массивами одинаковой формы"
        )
    predicted_ids, predicted_area_counts = np.unique(
        predicted_instances,
        return_counts=True,
    )
    positive_ids = predicted_ids > 0
    predicted_ids = predicted_ids[positive_ids]
    predicted_area_counts = predicted_area_counts[positive_ids]
    predicted_count = int(predicted_ids.size)
    true_ids = np.unique(true_instances)
    true_ids = true_ids[true_ids > 0]
    if true_ids.size == 0 or predicted_count == 0:
        return _result(
            true_positive=0,
            false_positive=int(predicted_count),
            false_negative=int(true_ids.size),
        )

    adjacency = np.zeros((true_ids.size, predicted_ids.size), dtype=np.uint8)
    predicted_areas = dict(
        zip(
            (int(predicted_id) for predicted_id in predicted_ids),
            (int(area) for area in predicted_area_counts),
            strict=True,
        )
    )
    for true_position, true_id in enumerate(true_ids):
        true_mask = true_instances == true_id
        true_area = int(np.count_nonzero(true_mask))
        overlap_ids, overlap_counts = np.unique(
            predicted_instances[true_mask],
            return_counts=True,
        )
        for predicted_id, raw_intersection in zip(overlap_ids, overlap_counts, strict=True):
            if predicted_id <= 0:
                continue
            intersection = int(raw_intersection)
            union = true_area + predicted_areas[int(predicted_id)] - intersection
            iou = intersection / union if union else 0.0
            if iou >= request.iou_threshold:
                predicted_position = int(np.searchsorted(predicted_ids, predicted_id))
                adjacency[true_position, predicted_position] = 1

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
