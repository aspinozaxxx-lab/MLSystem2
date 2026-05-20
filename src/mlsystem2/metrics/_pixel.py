"""Реализация пиксельных метрик."""

from __future__ import annotations

import numpy as np

from .contracts import MetricsError, PixelF1Request, PixelF1Result


def compute_pixel_f1(request: PixelF1Request) -> PixelF1Result:
    y_true = np.asarray(request.y_true) > 0
    y_pred = np.asarray(request.y_pred) >= request.threshold
    if y_true.shape != y_pred.shape:
        raise MetricsError("y_true и y_pred должны иметь одинаковую форму")

    true_positive = int(np.logical_and(y_true, y_pred).sum())
    false_positive = int(np.logical_and(~y_true, y_pred).sum())
    false_negative = int(np.logical_and(y_true, ~y_pred).sum())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return PixelF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
