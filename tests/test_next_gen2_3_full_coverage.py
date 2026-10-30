"""Полное покрытие новых инференсов next-gen2/3 без изменения весов и порогов."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

from mlsystem2.training_ui_api import _geoalert_notebook, _geoalert_runner, _pseudo_runner
from mlsystem2.training_ui_api._raster_valid_data import valid_data_mask
from test_training_ui_pseudo_runner import _FakeModel, _FakeTorch


def _raster(path, height, width):
    image = np.full((4, height, width), 80, np.uint8)
    image[:, :3, :4] = 0
    mask = np.full((height, width), 255, np.uint8)
    mask[10:13, 10:14] = 0
    with rasterio.open(path, "w", driver="GTiff", count=4, dtype="uint8",
                       width=width, height=height, crs="EPSG:3857", nodata=0,
                       transform=from_origin(0, height, 1, 1)) as output:
        output.write(image)
        output.write_mask(mask)


@pytest.mark.parametrize("variant", ["next_gen2", "next_gen3"])
@pytest.mark.parametrize("height,width,size", [(27, 31, 32), (64, 64, 32), (65, 77, 32), (2048, 2048, 768)])
def test_native_scene_covers_every_valid_pixel_including_remainders(tmp_path, monkeypatch, variant, height, width, size):
    image = tmp_path / "input.tif"
    _raster(image, height, width)
    before = image.read_bytes()

    def predict(torch, model, images, **kwargs):
        return None, np.full((len(images), size, size), .99, np.float32)

    monkeypatch.setattr(_pseudo_runner, "_predict_tiles", predict)
    monkeypatch.setattr(_pseudo_runner, "_features_from_mask", lambda mask, *args, **kwargs: mask)
    actual = _pseudo_runner._infer_scene(
        torch=None, model=None, input_channels=4, image_path=image, scene="снимок",
        config={"pipeline_variant": variant}, tile_size=size, stride=size // 2,
        batch_size=2, threshold=.9, device="cpu",
        postprocess_profile=_pseudo_runner._POSTPROCESS_NONE,
    )
    with rasterio.open(image) as source:
        valid = valid_data_mask(source)
    np.testing.assert_array_equal(actual > 0, valid)
    assert actual[-1, -1] == 1 and actual.shape == (height, width)
    assert image.read_bytes() == before


@pytest.mark.parametrize("channels", [1, 3])
def test_native_and_portable_gaussian_cover_same_pixels_and_exclude_nodata(tmp_path, monkeypatch, channels):
    image = tmp_path / "input.tif"
    _raster(image, 65, 77)
    size = 32

    def probabilities(batch):
        values = np.full((batch, channels, size, size), .005, np.float32)
        values[:, 0 if channels == 1 else 1] = .99
        return values[:, 0] if channels == 1 else values

    monkeypatch.setattr(_pseudo_runner, "_predict_tiles", lambda torch, model, images, **kwargs:
                        (None, probabilities(len(images))))
    with rasterio.open(image) as source:
        native, _ = _pseudo_runner._infer_notebook_scene_mask(
            dataset=source, input_indexes=(1, 2, 3, 4), input_channels=4,
            torch=None, model=None, tile_size=size, stride=16, batch_size=2,
            threshold=.9, device="cpu", pad_edges=True,
        )
        valid = valid_data_mask(source)
    _geoalert_notebook._split_unmasked(image, tmp_path, ["r", "g", "b", "n"])
    names = [f"p{index}" for index in range(channels)]
    _geoalert_notebook._sliding_window_probabilities(
        tmp_path, ["r", "g", "b", "n"], names,
        lambda image: probabilities(1).reshape(channels, size, size),
        size, 16, .25, pad_edges=True,
    )
    with rasterio.open(tmp_path / f"p{0 if channels == 1 else 1}.tif") as source:
        portable = source.read(1) > .9
    np.testing.assert_array_equal(native > 0, valid)
    np.testing.assert_array_equal(portable, valid)


@pytest.mark.parametrize("variant", ["next_gen2", "next_gen3"])
def test_test_f1_runner_uses_full_coverage_for_smaller_than_window_image(tmp_path, monkeypatch, variant):
    image, truth, checkpoint = tmp_path / "image.tif", tmp_path / "truth.tif", tmp_path / "best.pt"
    _raster(image, 27, 31)
    with rasterio.open(image) as source:
        valid, profile = valid_data_mask(source), source.profile
    profile.update(count=1)
    with rasterio.open(truth, "w", **profile) as output:
        output.write(valid.astype(np.uint8), 1)
    checkpoint.write_bytes("неизменяемые проверочные веса".encode("utf-8"))
    model, torch = _FakeModel(), _FakeTorch()
    loaded = SimpleNamespace(artifact=SimpleNamespace(metadata={"pipeline_variant": variant, "sample_size": 512}),
        model=SimpleNamespace(model=model, spec=SimpleNamespace(parameters={}, input_channels=4, output_channels=1)))
    monkeypatch.setattr(_pseudo_runner, "_torch", lambda: torch)
    monkeypatch.setattr(_pseudo_runner, "load_checkpoint", lambda request: loaded)
    monkeypatch.setattr(_pseudo_runner, "_predict_tiles", lambda torch, model, images, **kwargs:
                        (None, np.full((len(images), 512, 512), .99, np.float32)))
    report = _pseudo_runner.run_test_sample_f1({
        "run_root": str(tmp_path / "run"), "local_checkpoint_path": str(checkpoint),
        "device": "cpu", "tile_size": 512, "stride": 256,
        "tiles": [{"index": 1, "image_path": str(image), "mask_path": str(truth)}],
    })
    assert report["status"] == "ok", report
    assert report["processed"] == 1 and report["false_negative"] == report["false_positive"] == 0
    assert report["true_positive"] == int(valid.sum()) and report["threshold"] == .9
    assert checkpoint.read_bytes() == "неизменяемые проверочные веса".encode("utf-8")


@pytest.mark.parametrize("variant", ["next_gen2", "next_gen3"])
def test_geoalert_rebuilds_old_pipeline_but_keeps_identical_onnx(tmp_path, monkeypatch, variant):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         default=str).encode("utf-8")).hexdigest()

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {"pipeline_variant": variant, "tile_size": 768, "threshold": .9,
              "geoalert_model_repository": str(tmp_path / "models"),
              "geoalert_pipeline_root": str(tmp_path / "pipelines")}
    model_digest = digest({"runtime_contract": 4, "inference_profile": "notebook_valid_data_v2",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "tile_size": 768, "threshold": .9, "external": None, "python_site_packages": None})
    name = f"mls_ortho_{model_digest[:24]}"
    model_dir, pipeline_root = tmp_path / "models" / name, tmp_path / "pipelines"
    (model_dir / "1").mkdir(parents=True)
    pipeline_root.mkdir()
    (model_dir / "config.pbtxt").write_text(f'name: "{name}"', encoding="utf-8")
    model_path = model_dir / "1" / "model.onnx"
    model_path.write_bytes("неизменённый ONNX".encode("utf-8"))
    model_mtime = model_path.stat().st_mtime_ns
    marker = pipeline_root / f"{name}.model.json"
    marker.write_text(json.dumps({"identity_sha256": model_digest, "model_name": name,
        "input_channels": 4, "sample_size": 768, "task": "binary", "class_schema": [],
        "output_kind": "probabilities", "threshold": .9, "pipeline_variant": variant}), encoding="utf-8")
    marker_bytes = marker.read_bytes()
    old_digest = digest({"pipeline_contract": 4, "model_sha256": model_digest, "context": 0,
                         "resolution_m": None, "postprocess": {}, "class_schema": []})
    old_pipeline = pipeline_root / f"{name}_{old_digest[:12]}.yaml"
    old_pipeline.write_text("version: 0.1.4\n", encoding="utf-8")
    old_pipeline.with_suffix(".json").write_text(json.dumps({"identity_sha256": old_digest}), encoding="utf-8")

    def unexpected_export(**kwargs):
        pytest.fail("Исправление покрытия не должно повторно экспортировать ONNX")

    monkeypatch.setattr(_geoalert_runner, "build_triton_model_export_zip", unexpected_export)
    actual = _geoalert_runner._ensure_runtime_export(config, checkpoint, external_manifest=None, postprocess_config={})
    assert actual.model_name == name and actual.model_dir == model_dir
    assert actual.pipeline_path != old_pipeline
    bricks = yaml.safe_load(actual.pipeline_path.read_text(encoding="utf-8"))["config"]["bricks"]
    assert next(brick for brick in bricks if brick["_class"] == "SlidingWindowSegmentation")["pad_edges"] is True
    assert model_path.read_bytes() == "неизменённый ONNX".encode("utf-8") and model_path.stat().st_mtime_ns == model_mtime
    assert marker.read_bytes() == marker_bytes and old_pipeline.read_text(encoding="utf-8") == "version: 0.1.4\n"
    assert _geoalert_runner._ensure_runtime_export(config, checkpoint, external_manifest=None, postprocess_config={}) == actual
