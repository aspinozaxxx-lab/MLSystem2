from __future__ import annotations

import pytest

from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import MetricsError, ObjectF1Request


def test_object_f1_matches_instances_one_to_one() -> None:
    result = compute_object_f1(
        ObjectF1Request(
            y_true_instances=[[1, 1, 1, 2, 2, 2]],
            y_pred_mask=[[1, 1, 1, 1, 1, 1]],
        )
    )

    assert result.true_positive == 1
    assert result.false_positive == 0
    assert result.false_negative == 1
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.f1 == pytest.approx(2.0 / 3.0)


def test_object_f1_rejects_mismatched_masks() -> None:
    with pytest.raises(MetricsError, match="одинаковой формы"):
        compute_object_f1(
            ObjectF1Request(
                y_true_instances=[[1, 0]],
                y_pred_mask=[[1], [0]],
            )
        )


def test_object_f1_preserves_explicit_adjacent_instances() -> None:
    explicit = compute_object_f1(
        ObjectF1Request(
            y_true_instances=[[1, 1, 2, 2]],
            y_pred_instances=[[10, 10, 20, 20]],
        )
    )
    binary = compute_object_f1(
        ObjectF1Request(
            y_true_instances=[[1, 1, 2, 2]],
            y_pred_mask=[[1, 1, 1, 1]],
        )
    )

    assert explicit.true_positive == 2
    assert explicit.f1 == 1.0
    assert explicit.matched_pairs == [(1, 10), (2, 20)]
    assert binary.true_positive == 1
    assert len(binary.matched_pairs) == binary.true_positive


def test_object_f1_pairs_use_original_sparse_ids_and_maximum_matching() -> None:
    result = compute_object_f1(ObjectF1Request(
        y_true_instances=[[4, 4, 0, 95, 95, 0]],
        y_pred_instances=[[0, 80, 0, 777, 777, 1000]],
    ))
    assert result.matched_pairs == [(4, 80), (95, 777)]
    assert result.true_positive == 2
    assert result.false_positive == 1


def test_object_f1_requires_exactly_one_prediction_representation() -> None:
    with pytest.raises(ValueError, match="ровно одно"):
        ObjectF1Request(y_true_instances=[[1]])
    with pytest.raises(ValueError, match="ровно одно"):
        ObjectF1Request(
            y_true_instances=[[1]],
            y_pred_mask=[[1]],
            y_pred_instances=[[1]],
        )
