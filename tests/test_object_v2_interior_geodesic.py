"""Сердцевины участков и связное распределение контуров в object f1 v2."""

import numpy as np
import pytest
from scipy import ndimage

from mlsystem2.inference.api import create_object_scene, separate_objects
from mlsystem2.inference.contracts import (
    ObjectSceneRequest, ObjectSeparationConfig, ObjectWindowPrediction,
)


def _config(**updates):
    return ObjectSeparationConfig(profile="object_f1_v2", **updates)


def _assert_connected(labels):
    for instance_id in np.unique(labels[labels > 0]):
        assert ndimage.label(labels == instance_id)[1] == 1


def _same_partition(left, right):
    """Сравнить принадлежность пикселей без зависимости от нумерации ID."""
    np.testing.assert_array_equal(left > 0, right > 0)
    matched = set()
    for instance_id in np.unique(left[left > 0]):
        values = np.unique(right[left == instance_id])
        assert len(values) == 1 and values[0] > 0
        assert values[0] not in matched
        matched.add(values[0])
        np.testing.assert_array_equal(left == instance_id, right == values[0])


def _parcels_with_false_ring(count):
    height, width = 64, 24 * count + 16
    foreground = np.zeros((height, width), np.float32)
    foreground[4:-4, 4:-4] = .95
    boundary = foreground * (0.9 / .95)
    for index in range(count):
        left = 12 + 24 * index
        boundary[20:44, left:left + 16] = .05
    # Замкнутая полоса имеет foreground >= .5 и boundary < любого порога,
    # но внутренность .35 проигрывает фону .40: это не допустимое семя.
    ring = np.zeros_like(foreground, bool)
    ring[8:56, 8:-8] = True
    ring[12:52, 12:-12] = False
    foreground[ring] = .60
    boundary[ring] = .25
    return foreground, boundary, np.ones_like(ring), ring


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("threshold", [.35, .5, .65])
def test_background_dominated_ring_is_distributed_to_adjacent_interiors(count, threshold):
    foreground, boundary, valid, ring = _parcels_with_false_ring(count)
    config = _config(boundary_threshold=threshold)
    labels = separate_objects(foreground, boundary, valid, config)
    assert labels.max() == count
    np.testing.assert_array_equal(labels > 0, foreground >= .5)
    interior = foreground - boundary
    support = (interior > boundary) & (interior > 1 - foreground)
    for instance_id in range(1, count + 1):
        assert np.any((labels == instance_id) & support)
    assert len(np.unique(labels[ring])) == count
    _assert_connected(labels)
    np.testing.assert_array_equal(labels, separate_objects(foreground, boundary, valid, config))


@pytest.mark.parametrize("shape", ["ring", "strip"])
@pytest.mark.parametrize("foreground_value,boundary_value", [(.95, .9), (.6, .25), (.75, .375)])
def test_contour_without_dominant_interior_cannot_create_an_object(shape, foreground_value, boundary_value):
    mask = np.zeros((64, 64), bool)
    if shape == "ring":
        mask[8:56, 8:56] = True
        mask[12:52, 12:52] = False
    else:
        mask[30:34, 4:60] = True
    foreground = mask.astype(np.float32) * foreground_value
    boundary = mask.astype(np.float32) * boundary_value
    valid = np.ones_like(mask)
    assert not separate_objects(foreground, boundary, valid, _config()).any()
    # Исторический профиль по-прежнему допускает резервный центр на контуре.
    legacy = separate_objects(foreground, boundary, valid, ObjectSeparationConfig())
    np.testing.assert_array_equal(legacy > 0, mask)


def test_geodesic_assignment_preserves_holes_nodata_and_connected_ids():
    foreground, boundary, valid, _ = _parcels_with_false_ring(3)
    foreground[28:34, 34:39] = 0
    boundary[28:34, 34:39] = 0
    valid[31:35, 60:66] = False
    valid[:6] = False
    labels = separate_objects(foreground, boundary, valid, _config())
    assert labels.max() == 3
    np.testing.assert_array_equal(labels > 0, (foreground >= .5) & valid)
    assert not labels[~valid].any()
    _assert_connected(labels)


@pytest.mark.parametrize("axis", [0, 1])
def test_geodesic_partition_is_reflection_equivariant_without_distance_ties(axis):
    # Две одинаковые сердцевины и чётный зазор: нет пикселей, равноудалённых
    # от них. Произвольный порядок ID не участвует в сравнении разделения.
    foreground = np.zeros((48, 80), np.float32)
    foreground[4:44, 4:76] = .95
    boundary = np.where(foreground > .5, .9, 0).astype(np.float32)
    boundary[12:36, 12:28] = .05
    boundary[12:36, 52:68] = .05
    valid = np.ones_like(foreground, bool)
    expected = separate_objects(foreground, boundary, valid, _config())
    mirrored = separate_objects(np.flip(foreground, axis), np.flip(boundary, axis),
                                np.flip(valid, axis), _config())
    _same_partition(expected, np.flip(mirrored, axis))
    _assert_connected(expected)


@pytest.mark.parametrize("use_peak", [False, True])
def test_fallback_marker_uses_real_interior_not_larger_boundary_maximum(monkeypatch, use_peak):
    import skimage.morphology

    foreground = np.zeros((48, 64), np.float32)
    foreground[4:44, 4:60] = .95
    boundary = np.where(foreground > .5, .9, 0).astype(np.float32)
    boundary[8, 9] = .05
    support = np.zeros_like(foreground, bool)
    support[8, 9] = True
    calls = []
    peak_masks = []
    maxima = []
    original = ndimage.distance_transform_edt
    original_label = ndimage.label
    original_argmax = np.argmax

    def capture_distance(mask):
        result = original(mask)
        calls.append(result.copy())
        return result

    def peaks(distance, prominence):
        result = np.zeros_like(distance, bool)
        # Самый глубокий центр полной области лежит в границе. Он не должен
        # становиться резервным маркером даже при большом расстоянии до края.
        result[20, 28] = True
        if use_peak:
            result[4, 5] = True
        return result

    def label(mask):
        peak_masks.append(mask.copy())
        return original_label(mask)

    def argmax(values):
        maxima.append(values.copy())
        return original_argmax(values)

    monkeypatch.setattr(ndimage, "distance_transform_edt", capture_distance)
    monkeypatch.setattr(ndimage, "label", label)
    monkeypatch.setattr(np, "argmax", argmax)
    monkeypatch.setattr(skimage.morphology, "h_maxima", peaks)
    labels = separate_objects(foreground, boundary, np.ones_like(support), _config())
    assert labels.max() == 1
    assert len(calls) == 1
    assert calls[0].max() > calls[0][5, 6]
    if use_peak:
        np.testing.assert_array_equal(peak_masks[-1], support[4:44, 4:60])
        assert not maxima
    else:
        assert not peak_masks[-1].any()
        assert len(maxima) == 1
        np.testing.assert_array_equal(maxima[0] >= 0, support[4:44, 4:60])
    np.testing.assert_array_equal(labels > 0, foreground >= .5)
    assert labels[support].item() == 1


def test_tiny_geodesic_fragment_is_redistributed_without_a_hole():
    foreground = np.full((48, 48), .95, np.float32)
    boundary = np.full_like(foreground, .05)
    boundary[19:29, 19:29] = .9
    boundary[23:25, 23:25] = .05
    valid = np.ones_like(foreground, bool)
    unfiltered = separate_objects(foreground, boundary, valid,
                                  _config(min_marker_pixels=1, min_object_pixels=1, boundary_closing_radius=0))
    counts = np.bincount(unfiltered.ravel())[1:]
    assert len(counts) == 2 and counts.min() < 64
    filtered = separate_objects(foreground, boundary, valid,
                                _config(min_marker_pixels=1, min_object_pixels=64, boundary_closing_radius=0))
    assert filtered.max() == 1 and np.all(filtered == 1)


def test_disconnected_supportless_component_does_not_steal_id():
    foreground = np.zeros((40, 80), np.float32)
    foreground[4:36, 4:36] = .95
    foreground[4:36, 44:76] = .95
    boundary = np.where(foreground > .5, .9, 0).astype(np.float32)
    boundary[12:28, 52:68] = .05
    labels = separate_objects(foreground, boundary, np.ones_like(foreground, bool), _config())
    assert labels.max() == 1
    assert not labels[:, :40].any()
    assert np.all(labels[4:36, 44:76] == 1)
    _assert_connected(labels)


def test_accumulator_and_resegment_use_same_geodesic_partition(tmp_path):
    foreground, boundary, valid, _ = _parcels_with_false_ring(2)
    size = foreground.shape[0]
    assert foreground.shape == (size, size)
    accumulator = create_object_scene(ObjectSceneRequest(width=size, height=size, tile_size=size,
        work_dir=str(tmp_path), separation=_config()))
    try:
        accumulator.add_window(ObjectWindowPrediction(x=0, y=0,
            probabilities=np.stack((foreground, boundary)), valid_pixels=valid))
        first = accumulator.finish()
        expected = separate_objects(first.probabilities[0], first.probabilities[1], valid, _config())
        np.testing.assert_array_equal(first.instances, expected)
        assert first.object_count == 2
        for threshold in (.35, .65, .5):
            config = _config(boundary_threshold=threshold)
            result = accumulator.resegment(config)
            expected = separate_objects(result.probabilities[0], result.probabilities[1], valid, config)
            np.testing.assert_array_equal(result.instances, expected)
            assert result.object_count == 2
    finally:
        accumulator.close()
