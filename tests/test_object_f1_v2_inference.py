"""Разделение соседних участков и единый контракт вероятностей object f1 v2."""

from types import SimpleNamespace

import numpy as np
import pytest

from mlsystem2.inference.api import (
    create_object_scene, object_separation_config, separate_objects,
)
from mlsystem2.inference.contracts import (
    InferenceError, ObjectSceneRequest, ObjectSeparationConfig, ObjectWindowPrediction,
)
from mlsystem2.models.contracts import ModelSpec


def _config(**values):
    return ObjectSeparationConfig(profile="object_f1_v2", **values)


def _touching(gap=False):
    foreground = np.zeros((64, 64), np.float32)
    foreground[8:56, 8:56] = .95
    boundary = np.zeros_like(foreground)
    boundary[8:56, 32] = .9
    if gap:
        boundary[31, 32] = 0
    return foreground, boundary, np.ones_like(foreground, bool)


@pytest.mark.parametrize("gap", [False, True])
def test_touching_parcels_keep_two_ids_and_full_area(gap):
    area, boundary, valid = _touching(gap)
    result = separate_objects(area, boundary, valid, _config())
    assert result.max() == 2
    np.testing.assert_array_equal(result > 0, area > .5)
    assert result[24, 20] != result[24, 44]
    if gap:
        # Исторический профиль намеренно сохраняет прежнее поведение.
        assert separate_objects(area, boundary, valid).max() == 1


def test_small_noise_is_removed_before_fallback():
    area, boundary, valid = _touching()
    area[2, 2::4] = .99
    result = separate_objects(area, boundary, valid, _config())
    assert result.max() == 2
    assert not result[2].any()
    assert separate_objects(area, boundary, valid).max() > 2


def test_thin_parcel_uses_small_real_interior_and_background_stays_empty():
    area = np.zeros((32, 32), np.float32)
    area[15, 4:28] = .95
    boundary = np.where(area > .5, .9, 0).astype(np.float32)
    # Одного пикселя мало для обычного маркера, но он подтверждает внутренность.
    boundary[15, 16] = .05
    result = separate_objects(area, boundary, np.ones_like(area, bool), _config())
    assert result.max() == 1
    np.testing.assert_array_equal(result > 0, area > .5)
    assert not separate_objects(np.zeros_like(area), boundary, np.ones_like(area, bool), _config()).any()


def test_holes_and_nodata_are_not_filled():
    area = np.full((64, 64), .95, np.float32)
    area[24:32, 24:32] = 0
    valid = np.ones_like(area, bool)
    valid[:8] = False
    valid[40:44, 40:44] = False
    result = separate_objects(area, np.zeros_like(area), valid, _config())
    assert result.max() == 1
    np.testing.assert_array_equal(result > 0, (area > .5) & valid)


def test_distance_fallback_uses_significant_lobes_not_every_local_pixel():
    y, x = np.mgrid[:80, :80]
    mask = ((x - 24) ** 2 + (y - 40) ** 2 <= 14 ** 2) | ((x - 56) ** 2 + (y - 40) ** 2 <= 14 ** 2)
    mask[39:42, 24:57] = True
    area = mask.astype(np.float32) * .95
    boundary = mask.astype(np.float32) * .9
    # Обе доли имеют настоящую внутренность; чистый контур объектом не считается.
    boundary[40, 24] = boundary[40, 56] = .05
    result = separate_objects(area, boundary, np.ones_like(mask), _config())
    assert result.max() == 2
    np.testing.assert_array_equal(result > 0, mask)
    np.testing.assert_array_equal(result, separate_objects(area, boundary, np.ones_like(mask), _config()))


def test_accumulator_and_direct_separation_are_identical(tmp_path):
    area, boundary, valid = _touching(True)
    valid[2] = False
    scene = create_object_scene(ObjectSceneRequest(width=64, height=64, tile_size=64,
        work_dir=str(tmp_path), separation=_config()))
    try:
        scene.add_window(ObjectWindowPrediction(x=0, y=0, probabilities=np.stack((area, boundary)), valid_pixels=valid))
        result = scene.finish()
        assert result.object_count == 2
        np.testing.assert_array_equal(result.instances, separate_objects(area, boundary, valid, _config()))
    finally:
        scene.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("second_boundary,high_threshold_count", [(.45, 1), (.6, 2)])
def test_resegment_reuses_disk_maps_without_repeated_normalization(
    tmp_path, monkeypatch, second_boundary, high_threshold_count,
):
    area = np.zeros((64, 64), np.float32)
    area[8:56, 8:56] = .95
    boundary = np.zeros_like(area)
    boundary[8:56, 24] = .4
    boundary[8:56, 40] = second_boundary
    valid = np.ones_like(area, bool)
    accumulator = create_object_scene(ObjectSceneRequest(width=64, height=64, tile_size=64,
        work_dir=str(tmp_path), separation=_config()))
    with pytest.raises(InferenceError, match="после finish"):
        accumulator.resegment(_config())
    try:
        window = ObjectWindowPrediction(x=0, y=0, probabilities=np.stack((area, boundary)), valid_pixels=valid)
        accumulator.add_window(window)
        accumulator.add_window(window)
        first = accumulator.finish()
        probabilities = first.probabilities.copy()
        map_identity = first.instances
        files = sorted(str(path) for path in tmp_path.rglob("*"))

        def unexpected_allocation(*args, **kwargs):
            raise AssertionError("Повторное разделение не должно создавать дисковые карты")

        monkeypatch.setattr(accumulator, "_array", unexpected_allocation)
        # При .45 внутренность .50 преобладает и высокий порог объединяет
        # сердцевины; при .60 внутренность .35 остаётся слабее границы.
        for threshold, count in ((.35, 3), (.425, 2), (.65, high_threshold_count)):
            config = _config(boundary_threshold=threshold)
            result = accumulator.resegment(config)
            assert result.instances is map_identity
            assert result.object_count == count
            np.testing.assert_array_equal(result.probabilities, probabilities)
            np.testing.assert_array_equal(result.instances,
                separate_objects(probabilities[0], probabilities[1], valid, config))
        empty = accumulator.resegment(_config(foreground_threshold=.99))
        assert empty.object_count == 0 and not empty.instances.any()
        assert accumulator.resegment(_config(boundary_threshold=.35)).object_count == 3
        assert sorted(str(path) for path in tmp_path.rglob("*")) == files
    finally:
        accumulator.close()
    with pytest.raises(InferenceError, match="до close"):
        accumulator.resegment(_config())
    assert not list(tmp_path.iterdir())


def test_metadata_contract_rejects_absent_or_wrong_profile():
    values = dict(name="smp_segformer_b0", input_channels=3, output_channels=1)
    expected = _config(boundary_threshold=.35)
    spec = ModelSpec(**values, parameters={"pipeline_variant": "object_f1_v2", "object_separation": expected.model_dump()})
    assert object_separation_config(spec) == expected
    for raw in [None, {}, {"profile": "object_f1"}]:
        with pytest.raises(InferenceError):
            object_separation_config(ModelSpec(**values, parameters={"pipeline_variant": "object_f1_v2", "object_separation": raw}))
    legacy = object_separation_config(ModelSpec(**values, parameters={"pipeline_variant": "object_f1"}))
    assert legacy.min_marker_pixels == 4 and legacy.boundary_closing_radius == 0


def test_export_and_native_predictor_share_three_class_softmax(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    import yaml
    from mlsystem2.training_ui_api._model_export import _export_segmentation_mask_onnx, _pipeline_yaml
    from mlsystem2.training_ui_api._object_inference import torch_object_predictor
    from mlsystem2.training_ui_api._pseudo_runner import _native_inference_config

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x[:, :1], -x[:, :1], x[:, 1:2]), 1)

    path = tmp_path / "object_v2.onnx"
    _export_segmentation_mask_onnx(model=Model(), input_channels=3, output_channels=1, sample_size=32,
        threshold=.5, onnx_path=path, probability_output=True, object_output=True, object_profile="object_f1_v2")
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    images = np.random.default_rng(42).normal(size=(1, 3, 32, 32)).astype(np.float32)
    actual = reference.ReferenceEvaluator(graph).run(["probabilities"], {"input": images})[0]
    predictor = torch_object_predictor(torch, Model(), "cpu", 3, profile="object_f1_v2")
    expected = predictor(images)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.all(expected[:, 1] <= expected[:, 0])
    np.testing.assert_allclose(expected[:, 0], 1 - Model()(torch.from_numpy(images)).softmax(1)[:, 0].numpy(), atol=1e-6)
    separation = _config(boundary_threshold=.35, min_marker_pixels=24).model_dump()
    pipeline = yaml.safe_load(_pipeline_yaml("objects", 512, 3, probability_output=True,
        object_output=True, object_separation=separation, threshold=.9))
    bricks = pipeline["config"]["bricks"]
    assert bricks[0]["separation"] == separation
    assert bricks[0]["threshold"] == .5
    assert bricks[1]["value_property_name"] == "instance_id"
    loaded = SimpleNamespace(artifact=SimpleNamespace(metadata={"sample_size": 512}),
        model=SimpleNamespace(spec=ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=1,
            parameters={"pipeline_variant": "object_f1_v2", "object_separation": separation})))
    native = _native_inference_config(loaded, {"threshold": .9})
    assert native["object_separation"] == separation and native["threshold"] == .5


def test_export_cache_keeps_calibrated_separation_on_pipeline_rebuild(tmp_path, monkeypatch):
    import json
    import yaml
    from mlsystem2.training_ui_api import _geoalert_runner, _model_export

    separation = _config(boundary_threshold=.35, min_marker_pixels=24).model_dump()
    spec = ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=1,
        parameters={"pipeline_variant": "object_f1_v2", "object_separation": separation})
    loaded = SimpleNamespace(artifact=SimpleNamespace(metadata={"pipeline_variant": "object_f1_v2",
        "sample_size": 512, "inference_context": 0, "confidence_threshold": .5}),
        model=SimpleNamespace(model=object(), spec=spec))
    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", lambda _: loaded)
    monkeypatch.setattr(_model_export, "_checkpoint_task_schema", lambda _: ("binary", []))
    calls = []

    def export(**kwargs):
        calls.append(kwargs)
        kwargs["onnx_path"].write_bytes(b"onnx")

    monkeypatch.setattr(_model_export, "_export_segmentation_mask_onnx", export)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {"geoalert_model_repository": str(tmp_path / "models"),
        "geoalert_pipeline_root": str(tmp_path / "pipelines"), "pipeline_variant": "object_f1_v2",
        "tile_size": 512, "threshold": .9}
    first = _geoalert_runner._ensure_runtime_export(config, checkpoint, external_manifest=None, postprocess_config={})
    second = _geoalert_runner._ensure_runtime_export(config, checkpoint, external_manifest=None,
        postprocess_config={"postprocess.min_area_m2": 10})
    assert len(calls) == 1
    assert calls[0]["object_profile"] == "object_f1_v2"
    assert first.model_name == second.model_name and first.pipeline_path != second.pipeline_path
    for result in (first, second):
        bricks = yaml.safe_load(result.pipeline_path.read_text(encoding="utf-8"))["config"]["bricks"]
        assert bricks[0]["separation"] == separation and bricks[0]["threshold"] == .5
        assert bricks[1]["value_property_name"] == "instance_id"
    marker = json.loads((tmp_path / "pipelines" / f"{first.model_name}.model.json").read_text(encoding="utf-8"))
    assert marker["object_separation"] == separation


def test_small_rgba_raster_keeps_borders_alpha_and_separate_ids(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from mlsystem2.training_ui_api._object_inference import predict_instances

    path = tmp_path / "small.tif"
    with rasterio.open(path, "w", driver="GTiff", width=19, height=17, count=4,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 17, 1, 1)) as target:
        image = np.full((4, 17, 19), 255, np.uint8)
        image[3, 0] = 0
        target.write(image)
        target.colorinterp = tuple(rasterio.enums.ColorInterp[v] for v in ("red", "green", "blue", "alpha"))
    prediction = np.zeros((2, 32, 32), np.float32)
    prediction[0] = .95
    prediction[1, :, 9] = .9
    prediction[1, 8, 9] = 0
    with rasterio.open(path) as source:
        accumulator, result = predict_instances(source,
            lambda images: np.repeat(prediction[None], len(images), 0),
            input_indexes=(1, 2, 3), tile_size=32, separation=_config())
        try:
            assert result.object_count == 2
            np.testing.assert_array_equal(result.instances > 0, source.dataset_mask() > 0)
            assert result.instances[-1, -1] > 0
        finally:
            accumulator.close()
