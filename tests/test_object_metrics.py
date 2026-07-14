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
