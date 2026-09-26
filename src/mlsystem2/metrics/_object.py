"""Расчёт объектовой F1 по маскам одного тайла."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .contracts import MetricsError, ObjectF1Request, ObjectF1Result


def compute_object_f1(request: ObjectF1Request) -> ObjectF1Result:
    true_instances = np.asarray(request.y_true_instances)
    if request.y_pred_instances is not None:
        predicted_instances = np.asarray(request.y_pred_instances)
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
    # Таблица площадей и пересечений строится за один проход полосами.
    # Стоимость больше не пропорциональна числу объектов × площади снимка.
    from collections import Counter

    true_areas, predicted_areas, intersections = Counter(), Counter(), Counter()
    for y in range(0, true_instances.shape[0], 256):
        truth = true_instances[y:y+256]
        predicted = predicted_instances[y:y+256]
        for values, destination in ((truth, true_areas), (predicted, predicted_areas)):
            ids, counts = np.unique(values, return_counts=True)
            destination.update({int(key): int(count) for key, count in zip(ids, counts, strict=True) if key > 0})
        common = (truth > 0) & (predicted > 0)
        if np.any(common):
            pairs, counts = np.unique(np.column_stack((truth[common], predicted[common])), axis=0, return_counts=True)
            intersections.update({(int(pair[0]), int(pair[1])): int(count) for pair, count in zip(pairs, counts, strict=True)})
    true_ids = sorted(true_areas)
    predicted_ids = sorted(predicted_areas)
    predicted_count = len(predicted_ids)
    if not true_ids or not predicted_ids:
        return _result(true_positive=0, false_positive=predicted_count, false_negative=len(true_ids))
    true_positions = {key: i for i, key in enumerate(true_ids)}
    pred_positions = {key: i for i, key in enumerate(predicted_ids)}
    rows, columns = [], []
    for (true_id, predicted_id), intersection in intersections.items():
        union = true_areas[true_id] + predicted_areas[predicted_id] - intersection
        if intersection / union >= request.iou_threshold:
            rows.append(true_positions[true_id])
            columns.append(pred_positions[predicted_id])
    adjacency = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)), shape=(len(true_ids), len(predicted_ids)))

    matching = maximum_bipartite_matching(adjacency, perm_type="column")
    true_positive = int(np.count_nonzero(matching >= 0))
    return _result(
        true_positive=true_positive,
        false_positive=int(predicted_count) - true_positive,
        false_negative=len(true_ids) - true_positive,
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
