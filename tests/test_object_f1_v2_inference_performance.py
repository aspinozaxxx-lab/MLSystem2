"""Ускорение object f1 v2 сохраняет точные ID и не повторяет лишнюю работу."""

from hashlib import sha256

import numpy as np
import pytest
from scipy import ndimage
from skimage import segmentation

from mlsystem2.inference.api import create_object_scene, separate_objects
from mlsystem2.inference.contracts import (
    ObjectSceneRequest, ObjectSeparationConfig, ObjectWindowPrediction,
)


def _config(**values):
    return ObjectSeparationConfig(profile="object_f1_v2", **values)


def _complex_case(name):
    area = np.zeros((64, 64), np.float32)
    boundary = np.zeros_like(area)
    valid = np.ones_like(area, bool)
    options = {}
    if name == "empty":
        pass
    elif name == "single_holes":
        area[3:61, 2:62] = .95
        area[18:25, 12:22] = 0
        valid[40:46, 40:47] = False
    elif name == "marker_gaps":
        area[:] = .95
        boundary[:] = .9
        boundary[0, 0] = .1
        boundary[2:60, 3:23] = .1
        boundary[2:60, 40:61] = .1
        options["boundary_closing_radius"] = 0
    elif name == "thin_fallback":
        area[25, 5:59] = .95
        boundary[:] = .9
    elif name == "lobes_fallback":
        y, x = np.mgrid[:64, :64]
        mask = ((x - 17) ** 2 + (y - 32) ** 2 <= 12 ** 2)
        mask |= (x - 47) ** 2 + (y - 32) ** 2 <= 12 ** 2
        mask[31:34, 17:48] = True
        area[mask] = .95
        boundary[mask] = .9
    elif name == "all_small_regions":
        area[4:60, 4:60] = .95
        boundary[::4] = .9
        boundary[:, ::4] = .9
        options.update(min_marker_pixels=1, min_object_pixels=128, boundary_closing_radius=0)
    elif name == "closing_noise":
        area[4:59, 4:59] = .95
        area[1, 1::3] = .95
        boundary[4:59, 31] = .9
        boundary[29, 31] = 0
        valid[40:44, 44:48] = False
    elif name == "partial_thresholds":
        area[3:61, 3:61] = .55
        area[8:56, 8:56] = .8
        area[12:52, 12:52] = .95
        boundary[3:61, 24] = .4
        boundary[3:61, 42] = .7
        valid[:, 35] = False
        options.update(foreground_threshold=.7, boundary_threshold=.35)
    else:
        raise AssertionError(name)
    return area, boundary, valid, _config(**options)


def _random_case(index):
    rng = np.random.default_rng(87012 + index)
    height, width = 37 + index % 7, 49 + index % 11
    area = np.zeros((height, width), np.float32)
    for _ in range(8):
        y, x = rng.integers((0, 0), (height - 3, width - 3))
        h, w = rng.integers(3, 24, size=2)
        area[y:y+h, x:x+w] = rng.choice([.3, .45, .55, .7, .95])
    boundary = np.zeros_like(area)
    if index % 3 == 0:
        boundary[:] = rng.random(area.shape, dtype=np.float32)
    elif index % 3 == 1:
        boundary[::4] = .8
        boundary[:, ::6] = .6
        boundary[rng.random(area.shape) < .025] = .1
    else:
        boundary[:] = .9
        for _ in range(8):
            y, x = rng.integers((0, 0), (height - 3, width - 3))
            boundary[y:y+5, x:x+7] = .1
    valid = rng.random(area.shape) >= .04
    config = _config(
        foreground_threshold=[.35, .5, .65][index % 3],
        boundary_threshold=[.35, .5, .65][(index // 3) % 3],
        min_marker_pixels=[1, 4, 16, 50][index % 4],
        min_object_pixels=[1, 16, 64, 128][(index // 4) % 4],
        boundary_closing_radius=(index // 16) % 3,
        fallback_peak_prominence=[1., 2., 3.][(index // 5) % 3],
    )
    return area, boundary, valid, config


def _digest(instances):
    return sha256(np.asarray(instances, dtype="<i4").tobytes()).hexdigest()


# Эталоны получены реализацией 396abfb до ускорения, включая точный порядок ID.
_COMPLEX_GOLDEN = {
    "empty": "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe",
    "single_holes": "a44719d8ccd7c3ffda935768a78befaef358399aa351d6c6821eafb70f37c7b5",
    "marker_gaps": "4931831204f80a3200e3f386e9a0eb510c11bc959fd953be0219e6db8537144a",
    "thin_fallback": "c619638daf019890ce6c82b86eb2cd3a93e2b7df49d5e8e377fad9bd6448b0a9",
    "lobes_fallback": "82cab53f0b3e849a6a4b97ce547c15a60f1963867400df59f576252604a7e101",
    "all_small_regions": "bf6a222922d1bf8c5409b8510434510d86d356e97b8aaeaf5d7e1e3445d94818",
    "closing_noise": "58a784cd2fec23ba0ec964e6b95c0897ea11f5f60ab8bde854da7d0e2a2ccd31",
    "partial_thresholds": "3a5df8871babad0bed1d9179f493f1b4392155c480e402be8ac81dba30d1b20d",
}
_RANDOM_GOLDEN = [
    "7fefb9bf1269080a88cb96b4c715831ba5c1623c46145eb244f6bf442ee87bf4",
    "e2a228b03fd2bf4e552ba39c2c1d2dfdfc4e6a48d678122567cb801d3b3b2dfb",
    "dc7fdf466f95cd83233cbce4ac3e0df332d72b8d6796074a55cd31d1dda7eafe",
    "3feed7e47ff2b87251059abe7b9b87ddbffa342048f154c00ae3f3d3849b13a7",
    "8d771914bcf8603bf3479da84a4e2424c283d762ff75c1b886e30f82628517fa",
    "d48eb0eb61276de601a82ec827222c5c5df2f49922831f372be04b11f49a3548",
    "11ed46e951449f60cac339fd3d4533e3d2a33757bde03c1c7419541602b9b877",
    "340d8107922ea656d20afaafe4f0998436b3b0754e8bf8a86cef1943152c1539",
]
_SCENE_GOLDEN = [
    "14b372f4c6235719a86596448ed14be9a052f2f216bf6beb343628c839f6eb6a",
    "b3518b36e511c6df913581794d26a7f9030bb1de1c2daf5d885b48dab94b0def",
    "14b372f4c6235719a86596448ed14be9a052f2f216bf6beb343628c839f6eb6a",
    "14b372f4c6235719a86596448ed14be9a052f2f216bf6beb343628c839f6eb6a",
    "69f7bd860100362e1a24ee9678f4460aefec5624aee65fcd389d962f03204360",
    "3a5df8871babad0bed1d9179f493f1b4392155c480e402be8ac81dba30d1b20d",
    "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe",
    "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe",
    "14b372f4c6235719a86596448ed14be9a052f2f216bf6beb343628c839f6eb6a",
]


@pytest.mark.parametrize("name", _COMPLEX_GOLDEN)
def test_complex_masks_keep_previous_exact_ids(name):
    assert _digest(separate_objects(*_complex_case(name))) == _COMPLEX_GOLDEN[name]


@pytest.mark.parametrize("group", range(8))
def test_random_masks_keep_previous_exact_ids(group):
    digest = sha256()
    for index in range(group * 16, (group + 1) * 16):
        digest.update(np.asarray(separate_objects(*_random_case(index)), dtype="<i4").tobytes())
    assert digest.hexdigest() == _RANDOM_GOLDEN[group]


@pytest.mark.parametrize("name, expected_calls", [
    ("empty", 0), ("single_holes", 0), ("thin_fallback", 0),
    ("marker_gaps", 1), ("lobes_fallback", 1), ("all_small_regions", 2),
])
def test_watershed_runs_only_for_necessary_markers(monkeypatch, name, expected_calls):
    original = segmentation.watershed
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(segmentation, "watershed", counted)
    result = separate_objects(*_complex_case(name))
    assert _digest(result) == _COMPLEX_GOLDEN[name]
    assert len(calls) == expected_calls


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
def test_scene_component_cache_tracks_foreground_threshold_only(tmp_path, monkeypatch, profile):
    area, boundary, valid, _ = _complex_case("partial_thresholds")
    config = ObjectSeparationConfig(profile=profile, boundary_threshold=.5)
    scene = create_object_scene(ObjectSceneRequest(width=64, height=64, tile_size=64,
        work_dir=str(tmp_path), separation=config))
    original = ndimage.label
    calls = []

    def counted(array, *args, **kwargs):
        if kwargs.get("output") is getattr(scene, "_components", None):
            calls.append(1)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(ndimage, "label", counted)
    try:
        scene.add_window(ObjectWindowPrediction(x=0, y=0,
            probabilities=np.stack((area, boundary)), valid_pixels=valid))
        result = scene.finish()
        probabilities = result.probabilities.copy()
        assert len(calls) == 1
        assert _digest(result.instances) == _SCENE_GOLDEN[0]
        cases = [
            ({"boundary_threshold": .35}, 1),
            ({"boundary_threshold": .65}, 1),
            ({"min_marker_pixels": 32, "min_object_pixels": 24}, 1),
            ({"foreground_threshold": .7}, 2),
            ({"foreground_threshold": .7, "boundary_threshold": .35}, 2),
            ({"foreground_threshold": .99}, 3),
            ({"foreground_threshold": .99, "boundary_threshold": .65}, 3),
            ({"foreground_threshold": .5}, 4),
        ]
        for index, (updates, expected_v2_calls) in enumerate(cases, 2):
            current = config.model_copy(update=updates)
            result = scene.resegment(current)
            expected_calls = expected_v2_calls if profile == "object_f1_v2" else index
            assert len(calls) == expected_calls
            np.testing.assert_array_equal(result.probabilities, probabilities)
            assert _digest(result.instances) == _SCENE_GOLDEN[index - 1]
            assert result.object_count == result.instances.max()
            if profile == "object_f1_v2":
                # Для v1 сохраняем старую обработку bbox, включая разрешение равенств.
                expected = separate_objects(probabilities[0], probabilities[1], valid, current)
                np.testing.assert_array_equal(result.instances, expected)
    finally:
        scene.close()
    assert not list(tmp_path.iterdir())
