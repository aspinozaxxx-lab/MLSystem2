"""Погрешность float32 не ломает сборку сцен и не скрывает некорректный прогноз."""

import numpy as np
import pytest

from mlsystem2.inference.api import create_object_scene
from mlsystem2.inference.contracts import (
    InferenceError, ObjectSceneRequest, ObjectSeparationConfig, ObjectWindowPrediction,
)


_ROUNDING_LOGITS = [-36.5786247253418, 4.432613372802734, -3.373986005783081]


def _scene(tmp_path, profile="object_f1_v2", size=32):
    return create_object_scene(ObjectSceneRequest(width=size, height=size, tile_size=32,
        work_dir=str(tmp_path), separation=ObjectSeparationConfig(profile=profile)))


def _window(probabilities, valid=None, x=0, y=0):
    if valid is None:
        valid = np.ones((32, 32), bool)
    return ObjectWindowPrediction(x=x, y=y, probabilities=probabilities, valid_pixels=valid)


def test_real_finite_cpu_softmax_roundoff_passes_full_scene_validation(tmp_path):
    torch = pytest.importorskip("torch")
    # Реальный воспроизводитель: p1=0.9995931983, p2=0.0004068742, сумма=1+eps32.
    logits = torch.tensor(_ROUNDING_LOGITS, dtype=torch.float32).reshape(1, 3, 1, 1)
    logits = logits.expand(1, 3, 32, 32).contiguous()
    semantic = logits.softmax(1)
    probabilities = torch.cat((semantic[:, 1:].sum(1, keepdim=True), semantic[:, 2:3]), 1)[0].numpy()
    assert torch.isfinite(logits).all()
    assert np.isfinite(probabilities).all()
    assert 1 < probabilities[0].max() < 1 + 1e-6
    before = probabilities.copy()
    probabilities.setflags(write=False)
    valid = np.ones((32, 32), bool)
    valid[:3] = False
    scene = _scene(tmp_path)
    try:
        scene.add_window(_window(probabilities, valid))
        scene.add_window(_window(probabilities, valid))
        result = scene.finish()
        assert np.isfinite(result.probabilities).all()
        assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))
        np.testing.assert_array_equal(result.probabilities[0][valid], 1)
        np.testing.assert_array_equal(result.probabilities[:, ~valid], 0)
        np.testing.assert_array_equal(result.instances > 0, valid)
        np.testing.assert_array_equal(probabilities, before)
        assert result.object_count == 1
    finally:
        scene.close()
    assert not list(tmp_path.iterdir())


def test_training_validation_with_real_softmax_roundoff_finishes_metrics():
    torch = pytest.importorskip("torch")
    from mlsystem2.train._object_f1 import validate_objects
    from mlsystem2.train.contracts import TrainConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fixed_logits", torch.tensor(_ROUNDING_LOGITS, dtype=torch.float32).reshape(1, 3, 1, 1))

        def forward(self, images):
            return self.fixed_logits.expand(images.shape[0], 3, 32, 32).contiguous()

    model = Model()
    images = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    raw_logits = model(images)
    assert torch.isfinite(raw_logits).all()
    assert raw_logits.softmax(1)[:, 1:].sum(1).max() > 1
    valid = torch.ones((2, 32, 32), dtype=torch.bool)
    valid[:, :3] = False
    masks = valid[:, None].float()
    # Два перекрывающихся окна образуют одну полную сцену; nodata не входит в F1.
    meta = {
        "object_instances": valid.int(), "valid_pixels": valid,
        "boundary_target": torch.zeros((2, 32, 32), dtype=torch.float32),
        "boundary_valid": valid.clone(), "scene_ids": ["округление", "округление"],
        "scene_shapes": [{"width": 48, "height": 32}] * 2,
        "windows": [{"x": 0, "y": 0}, {"x": 16, "y": 0}], "overlap_pixels": 0,
    }
    config = TrainConfig(pipeline_variant="object_f1_v2", task="binary", quality_metric="objects",
        class_weights=[1, 1, 3], epochs=1, batch_size=2, device="cpu", learning_rate=1e-4,
        weight_decay=.01, loss="cross_entropy_tversky", early_stopping_patience=1,
        object_separation=ObjectSeparationConfig(profile="object_f1_v2"))
    result = validate_objects(torch, model, [(images, masks, meta)], "cpu", config, 1, None)
    assert not model.training
    assert all(np.isfinite(result[key]) for key in ("loss", "region_loss", "boundary_loss"))
    assert result["quality_f1"] == result["best_threshold_pixel_f1"] == 1
    assert result["object_separation"]["boundary_threshold"] == .5
    assert len(result["object_threshold_sweep"]) == 3
    assert all(row["object_f1"] == 1 for row in result["object_threshold_sweep"])
    assert len(result["per_scene_metrics"]) == 1
    scene = result["per_scene_metrics"][0]
    assert scene["true_objects"] == scene["predicted_objects"] == scene["tp"] == 1
    assert scene["fp"] == scene["fn"] == 0
    assert scene["pixel_counts"] == {"tp": 29 * 48, "fp": 0, "fn": 0}


@pytest.mark.parametrize("value, expected", [(-5e-7, 0), (1 + 5e-7, 1)])
@pytest.mark.parametrize("channel", [0, 1])
def test_v2_clips_small_roundoff_without_changing_caller_array(tmp_path, value, expected, channel):
    probabilities = np.full((2, 32, 32), .25, np.float32)
    probabilities[channel, 7, 9] = value
    before = probabilities.copy()
    probabilities.setflags(write=False)
    scene = _scene(tmp_path)
    try:
        scene.add_window(_window(probabilities))
        result = scene.finish()
        assert result.probabilities[channel, 7, 9] == expected
        np.testing.assert_array_equal(probabilities, before)
    finally:
        scene.close()


@pytest.mark.parametrize("value", [
    -np.finfo(np.float32).eps,
    np.nextafter(np.float32(1), np.float32(np.inf)),
])
def test_legacy_keeps_strict_range_without_roundoff_tolerance(tmp_path, value):
    scene = _scene(tmp_path, "object_f1")
    probabilities = np.full((2, 32, 32), .25, np.float32)
    probabilities[0, 4, 5] = value
    try:
        with pytest.raises(InferenceError):
            scene.add_window(_window(probabilities))
        assert not scene.probabilities.any()
        assert not scene.weights.any()
        assert not scene.valid.any()
    finally:
        scene.close()


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
@pytest.mark.parametrize("value", [
    np.nan, np.inf, -np.inf, -.001, 1.001,
    np.nextafter(np.float32(-1e-6), np.float32(-np.inf)),
    np.nextafter(np.float32(1 + 1e-6), np.float32(np.inf)),
])
@pytest.mark.parametrize("invalid_pixel", [False, True])
def test_bad_probability_is_rejected_before_any_accumulation(tmp_path, profile, value, invalid_pixel):
    scene = _scene(tmp_path, profile)
    good = np.full((2, 32, 32), .25, np.float32)
    probabilities = good.copy()
    probabilities[1, 4, 5] = value
    before = probabilities.copy()
    valid = np.ones((32, 32), bool)
    valid[4, 5] = not invalid_pixel
    try:
        scene.add_window(_window(good))
        maps = [array.copy() for array in (scene.probabilities, scene.weights, scene.valid)]
        with pytest.raises(InferenceError) as captured:
            scene.add_window(_window(probabilities, valid))
        if profile == "object_f1_v2" and np.isfinite(value):
            assert "минимум=" in str(captured.value) and "максимум=" in str(captured.value)
        for actual, previous in zip((scene.probabilities, scene.weights, scene.valid), maps):
            np.testing.assert_array_equal(actual, previous)
        np.testing.assert_array_equal(probabilities, before)
        # Ошибка отдельного окна не портит ранее добавленные данные.
        result = scene.finish()
        np.testing.assert_array_equal(result.probabilities, good)
    finally:
        scene.close()


def test_valid_probabilities_keep_legacy_gaussian_values_bit_for_bit(tmp_path):
    rng = np.random.default_rng(7361)
    legacy, current = _scene(tmp_path, "object_f1", 48), _scene(tmp_path, size=48)
    try:
        for x, y in ((0, 0), (16, 0), (0, 16), (16, 16)):
            probabilities = rng.uniform(.1, .9, (2, 32, 32)).astype(np.float32)
            before = probabilities.copy()
            probabilities.setflags(write=False)
            valid = rng.random((32, 32)) > .1
            for scene in (legacy, current):
                scene.add_window(_window(probabilities, valid, x, y))
            np.testing.assert_array_equal(probabilities, before)
            np.testing.assert_array_equal(current.probabilities, legacy.probabilities)
            np.testing.assert_array_equal(current.weights, legacy.weights)
            np.testing.assert_array_equal(current.valid, legacy.valid)
        previous, result = legacy.finish(), current.finish()
        np.testing.assert_array_equal(result.probabilities, previous.probabilities)
        np.testing.assert_array_equal(result.valid_pixels, previous.valid_pixels)
    finally:
        legacy.close()
        current.close()


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
def test_only_v2_clips_roundoff_after_gaussian_normalization(tmp_path, profile):
    scene = _scene(tmp_path, profile)
    probabilities = np.zeros((2, 32, 32), np.float32)
    probabilities[0] = 1
    try:
        scene.add_window(_window(probabilities))
        # Моделируем один ULP ошибки накопленной Gaussian-суммы перед нормализацией.
        scene.probabilities[0, 7, 9] = np.nextafter(scene.weights[7, 9], np.float32(np.inf))
        scene.probabilities[1, 5, 11] = -5e-7 * scene.weights[5, 11]
        result = scene.finish()
        if profile == "object_f1_v2":
            assert result.probabilities[0, 7, 9] == 1
            assert result.probabilities[1, 5, 11] == 0
        else:
            assert result.probabilities[0, 7, 9] > 1
            assert result.probabilities[1, 5, 11] < 0
    finally:
        scene.close()
