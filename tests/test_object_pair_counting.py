"""Точная совместимость ускоренного подсчёта пересечений с прежней сортировкой."""

from unittest.mock import patch

import numpy as np
import pytest

from mlsystem2.metrics import _object
from mlsystem2.metrics.contracts import ObjectF1Request
from mlsystem2.train import _object_v2


def _reference_pairs(truth, predicted):
    return np.unique(np.column_stack((truth, predicted)), axis=0, return_counts=True)


@pytest.mark.parametrize("module", [_object, _object_v2])
@pytest.mark.parametrize("dtype,ids", [
    (np.int32, [1, 17, 9000]),
    (np.uint32, [1, 2147483648, 4294967295]),
    (np.int64, [1, 2147483648, 4294967295]),
    (np.uint64, [1, 4294967296, 2**63 + 7]),
    (np.int64, [1, 4294967296, 2**62 + 7]),
    (np.float64, [0.5, 1.5, 9000.5]),
    (np.bool_, [False, True, True]),
])
def test_pair_counts_and_lexicographic_order_are_identical(module, dtype, ids):
    rng = np.random.default_rng(128)
    labels = np.asarray(ids, dtype=dtype)
    # Срезы намеренно неконтинуальны: помощник не должен менять входные данные.
    truth = labels[rng.integers(0, len(labels), 2000)][::2]
    predicted = labels[rng.integers(0, len(labels), 2000)][::2]
    original = truth.copy(), predicted.copy()
    expected = _reference_pairs(truth, predicted)
    actual = module._unique_instance_pairs(truth, predicted)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(truth, original[0])
    np.testing.assert_array_equal(predicted, original[1])


@pytest.mark.parametrize("module", [_object, _object_v2])
def test_empty_pairs_are_unchanged(module):
    values = np.array([], dtype=np.int32)
    expected = _reference_pairs(values, values)
    actual = module._unique_instance_pairs(values, values)
    np.testing.assert_array_equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])


def _scene(dtype, ids):
    rng = np.random.default_rng(99)
    height, width = 513, 129  # Три полосы, включая неполную последнюю.
    y, x = np.indices((height, width))
    truth = ((y // 32 * 5 + x // 32) % 4 + 1).astype(np.int64)
    truth[(x % 32 == 0) | (y % 32 == 0)] = 0
    predicted = np.roll(truth, 5, axis=1)
    predicted[(y < 64) & (predicted == 2)] = 1  # Слияние.
    predicted[(y > 400) & (x % 2 == 0) & (predicted == 3)] = 4  # Дробление.
    predicted[rng.random(truth.shape) < 0.04] = 0
    labels = np.asarray([0, *ids], dtype=dtype)
    truth, predicted = labels[truth], labels[predicted]
    valid = rng.random(truth.shape) > 0.12
    probabilities = rng.random((2, height, width), dtype=np.float32)
    boundary = (x % 32 < 4) | (y % 32 < 4)
    boundary_valid = valid & (rng.random(truth.shape) > 0.1)
    return truth, predicted, probabilities, valid, boundary, boundary_valid


@pytest.mark.parametrize("dtype,ids", [
    (np.int32, [1, 14, 765, 10003]),
    (np.uint32, [1, 14, 2147483648, 4294967295]),
    (np.int64, [1, 14, 2147483648, 4294967295]),
    (np.uint64, [1, 14, 4294967296, 2**63 + 7]),
    (np.float64, [1, 14, 765, 10003]),
])
@pytest.mark.parametrize("threshold", [0.35, 0.5, 0.65])
def test_full_metrics_including_matching_and_diagnostics_are_exact(dtype, ids, threshold):
    arrays = _scene(dtype, ids)
    request = ObjectF1Request(y_true_instances=arrays[0], y_pred_instances=arrays[1])
    actual_metric = _object.compute_object_f1(request).model_dump()
    actual_scene = _object_v2._scene_metrics(*arrays, threshold)
    with patch.object(_object, "_unique_instance_pairs", _reference_pairs), \
            patch.object(_object_v2, "_unique_instance_pairs", _reference_pairs):
        expected_metric = _object.compute_object_f1(request).model_dump()
        expected_scene = _object_v2._scene_metrics(*arrays, threshold)
    assert actual_metric == expected_metric
    assert actual_scene == expected_scene


@pytest.mark.parametrize("iou", [0.2, 0.5, 1.0])
def test_ambiguous_matching_and_binary_legacy_prediction_remain_exact(iou):
    truth = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    predicted = np.array([[5, 5, 5, 5], [8, 8, 8, 8]], dtype=np.int32)
    for prediction in ({"y_pred_instances": predicted}, {"y_pred_mask": predicted > 0}):
        request = ObjectF1Request(y_true_instances=truth, iou_threshold=iou, **prediction)
        actual = _object.compute_object_f1(request).model_dump()
        with patch.object(_object, "_unique_instance_pairs", _reference_pairs):
            expected = _object.compute_object_f1(request).model_dump()
        assert actual == expected


@pytest.mark.parametrize("empty", ["truth", "prediction", "both", "valid"])
def test_empty_or_invalid_scene_remains_exact(empty):
    arrays = list(_scene(np.int32, [1, 14, 765, 10003]))
    if empty in ("truth", "both"):
        arrays[0][:] = 0
    if empty in ("prediction", "both"):
        arrays[1][:] = 0
    if empty == "valid":
        arrays[3][:] = False
    actual = _object_v2._scene_metrics(*arrays, 0.5)
    with patch.object(_object, "_unique_instance_pairs", _reference_pairs), \
            patch.object(_object_v2, "_unique_instance_pairs", _reference_pairs):
        expected = _object_v2._scene_metrics(*arrays, 0.5)
    assert actual == expected
