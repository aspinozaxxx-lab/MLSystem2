"""Реальный NCHW batch v2 сохраняет окна, ID, nodata и контракт старых профилей."""

import json
import hashlib
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

from mlsystem2.training_ui_api import (
    _geoalert_compose_runner, _geoalert_notebook, _geoalert_runner, _model_export,
)
from mlsystem2.training_ui_api._object_inference import predict_instances


def _image(directory, height, width, channels=3):
    directory.mkdir(exist_ok=True)
    image = np.random.default_rng(42).integers(10, 256, (channels, height, width), dtype=np.uint8)
    path = directory / "input.tif"
    with rasterio.open(path, "w", driver="GTiff", count=channels, height=height, width=width,
                       crs="EPSG:3857", transform=from_origin(100, 500, .5, .5), dtype="uint8") as target:
        target.write(image)
        valid = np.full((height, width), 255, dtype=np.uint8)
        valid[3:12, 3:12] = 0
        target.write_mask(valid)
    return path, image, valid > 0


def _predict(images):
    # Выход зависит от каждого изображения отдельно, а не от позиции в batch.
    foreground = .8 + images[:, :1] / 2550
    boundary = images[:, 1:2] / 2550
    return np.concatenate((foreground, boundary), axis=1).astype(np.float32)


@pytest.mark.parametrize("height,width,batch_size", [(64, 64, 2), (64, 64, 4), (20, 39, 2), (20, 20, 4)])
def test_true_adapter_batch_and_tail_keep_exact_instances_and_nodata(tmp_path, height, width, batch_size):
    path, _, valid = _image(tmp_path, height, width)
    calls = []

    def adapter(images):
        assert images.ndim == 4 and images.dtype == np.float32
        assert images.shape[1:] == (3, 32, 32)
        calls.append(images.shape[0])
        return _predict(images)

    separation = {"profile": "object_f1_v2"}
    _geoalert_notebook._object_instance_segmentation(
        tmp_path, "input", "instances", adapter, input_channels=3, window_size=32,
        threshold=.5, separation=separation, batch_size=batch_size,
    )
    metrics = json.loads((tmp_path / "object_inference_performance.json").read_text(encoding="utf-8"))
    expected_windows = 9 if (height, width) == (64, 64) else 2 if width == 39 else 1
    assert sum(calls) == expected_windows
    assert calls == [batch_size] * (expected_windows // batch_size) + (
        [expected_windows % batch_size] if expected_windows % batch_size else []
    )
    assert metrics["prediction_batch_count"] == len(calls)
    assert metrics["tile_count"] == expected_windows
    assert metrics["configured_batch_size"] == metrics["effective_batch_size"] == batch_size
    for key in ["reading_sec", "prediction_sec", "gaussian_accumulation_sec", "object_separation_sec",
                "instance_write_sec", "cleanup_sec", "elapsed_sec"]:
        assert np.isfinite(metrics[key]) and metrics[key] >= 0
    with rasterio.open(path) as source:
        accumulator, result = predict_instances(source, _predict, input_indexes=(1, 2, 3),
                                               tile_size=32, batch_size=1, separation=separation)
        try:
            with rasterio.open(tmp_path / "instances.tif") as actual:
                instances = actual.read(1)
                np.testing.assert_array_equal(instances, result.instances)
                assert actual.width == width and actual.height == height
            assert not instances[~valid].any()
            assert instances[valid].min() > 0
            assert result.object_count == metrics["object_count"]
        finally:
            accumulator.close()


def test_old_object_profile_keeps_single_chw_requests_even_with_batch_config(tmp_path):
    _image(tmp_path, 64, 64)
    calls = []

    def adapter(image):
        assert image.shape == (3, 32, 32)
        calls.append(1)
        return _predict(image[None])[0]

    _geoalert_notebook._object_instance_segmentation(
        tmp_path, "input", "instances", adapter, input_channels=3, window_size=32,
        threshold=.5, separation=None, batch_size=4,
    )
    assert len(calls) == 9
    performance = json.loads((tmp_path / "object_inference_performance.json").read_text())
    assert performance["configured_batch_size"] == 4
    assert performance["effective_batch_size"] == 1


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_native_batch_fails_before_allocating_scene(tmp_path, batch_size):
    path, _, _ = _image(tmp_path, 20, 20)
    with rasterio.open(path) as source, pytest.raises(ValueError, match="положительным целым"):
        predict_instances(source, _predict, input_indexes=(1, 2, 3), tile_size=32, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_v2_pipeline_preserves_batch_dimension_and_old_profile_keeps_chw(batch_size):
    pipeline = yaml.safe_load(_model_export._pipeline_yaml(
        "objects", 1024, 3, probability_output=True, object_output=True,
        object_separation={"profile": "object_f1_v2"}, batch_size=batch_size,
    ))
    brick = pipeline["config"]["bricks"][0]
    assert brick["batch_size"] == batch_size
    assert brick["adapter"]["input_ndim"] == brick["adapter"]["output_ndim"] == 4
    old = yaml.safe_load(_model_export._pipeline_yaml(
        "objects", 1024, 3, probability_output=True, object_output=True, batch_size=batch_size,
    ))["config"]["bricks"][0]
    assert "batch_size" not in old
    assert old["adapter"]["output_ndim"] == 3


@pytest.mark.parametrize("batch_size", [0, -1, 33, True, 1.5])
def test_v2_pipeline_rejects_unbounded_batch(batch_size):
    with pytest.raises(_model_export.TrainingUIAPIError, match="от 1 до 32"):
        _model_export._pipeline_yaml("objects", 1024, 3, object_output=True,
                                    object_separation={"profile": "object_f1_v2"}, batch_size=batch_size)


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
def test_only_v2_uses_matched_edges_and_coverage_simplification(profile):
    pipeline = yaml.safe_load(_model_export._pipeline_yaml(
        "objects", 1024, 3, probability_output=True, object_output=True,
        object_separation={"profile": profile},
        postprocess_config={"postprocess.min_area_m2": 20., "postprocess.simplify_m": .5},
    ))
    bricks = pipeline["config"]["bricks"]
    operations = [operation for brick in bricks for operation in brick.get("bricks", [brick])]
    vectorize = next(brick for brick in bricks if brick["_class"] == "VectorizeMasks")
    if profile == "object_f1_v2":
        assert vectorize["match_instance_edges"] is True
        assert not any(brick["_class"] == "Simplify" for brick in operations)
        assert next(brick for brick in operations if brick["_class"] == "CoverageSimplify")["rate"] == .5
    else:
        assert "match_instance_edges" not in vectorize
        assert not any(brick["_class"] == "CoverageSimplify" for brick in operations)
        assert next(brick for brick in operations if brick["_class"] == "Simplify")["rate"] == .5


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
def test_onnx_dynamic_batch_exists_only_for_v2_and_matches_torch(tmp_path, profile):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    torch.set_num_threads(2)

    class Model(torch.nn.Module):
        def forward(self, inputs):
            return torch.cat((inputs[:, :1], -inputs[:, :1], inputs[:, 1:2]), 1)

    path = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(
        model=Model(), input_channels=3, output_channels=1, sample_size=32,
        threshold=.5, onnx_path=path, probability_output=True, object_output=True, object_profile=profile,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    dimension = graph.graph.input[0].type.tensor_type.shape.dim[0]
    assert dimension.dim_param == "batch" if profile == "object_f1_v2" else dimension.dim_value == 1
    output_dimension = graph.graph.output[0].type.tensor_type.shape.dim[0]
    if profile == "object_f1_v2":
        assert output_dimension.dim_param == "batch"
    evaluator = reference.ReferenceEvaluator(graph)
    for batch_size in ([1, 2, 4] if profile == "object_f1_v2" else [1]):
        inputs = np.random.default_rng(42).normal(size=(batch_size, 3, 32, 32)).astype(np.float32)
        logits = Model()(torch.from_numpy(inputs))
        if profile == "object_f1_v2":
            probabilities = logits.softmax(1)
            expected = torch.cat((probabilities[:, 1:].sum(1, keepdim=True), probabilities[:, 2:3]), 1).numpy()
        else:
            expected = torch.cat((logits[:, :2].softmax(1)[:, 1:2], logits[:, 2:3].sigmoid()), 1).numpy()
        actual = evaluator.run(["probabilities"], {"input": inputs})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    config = _model_export._triton_config("objects", 3, foreground_channels=2,
                                        probability_output=True, dynamic_batch=profile == "object_f1_v2")
    assert f"dims: [ {'-1' if profile == 'object_f1_v2' else '1'}, 3, -1, -1 ]" in config
    assert "max_batch_size: 0" in config
    assert ('key: "use_tf32" value: "0"' in config) == (profile == "object_f1_v2")


@pytest.mark.parametrize("architecture", ["smp_segformer_b0", "smp_segformer_b2"])
def test_real_v2_segformer_exports_dynamic_n_without_batch_leakage(tmp_path, architecture):
    from mlsystem2.models.api import create_model
    from mlsystem2.models.contracts import ModelSpec

    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    torch.set_num_threads(2)
    handle = create_model(ModelSpec(name=architecture, input_channels=3, output_channels=1, parameters={
        "pipeline_variant": "object_f1_v2", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6},
        "output_layout": "background_interior_boundary",
        "object_separation": {"profile": "object_f1_v2"},
    }))
    path = tmp_path / "real-v2.onnx"
    _model_export._export_segmentation_mask_onnx(
        model=handle.model, input_channels=3, output_channels=1, sample_size=64,
        threshold=.5, onnx_path=path, probability_output=True, object_output=True,
        object_profile="object_f1_v2",
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    evaluator = reference.ReferenceEvaluator(graph)
    images = np.random.default_rng(42).uniform(0, 255, (4, 3, 64, 64)).astype(np.float32)
    for count in [1, 2, 4]:
        actual = evaluator.run(["probabilities"], {"input": images[:count]})[0]
        with torch.no_grad():
            probabilities = handle.model(torch.from_numpy(images[:count])).softmax(1)
            expected = torch.cat((probabilities[:, 1:].sum(1, keepdim=True), probabilities[:, 2:3]), 1).numpy()
        assert actual.shape == (count, 2, 64, 64)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_compose_report_contains_saved_object_stage_timings(tmp_path, monkeypatch):
    performance = {"prediction_sec": .7, "gaussian_accumulation_sec": .1, "prediction_batch_count": 2}

    def run(directory):
        path = Path(directory)
        (path / "output.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        (path / "object_inference_performance.json").write_text(json.dumps(performance), encoding="utf-8")

    urban = ModuleType("urban")
    urban.Compose = SimpleNamespace(load=lambda *_args, **_kwargs: run)
    monkeypatch.setitem(sys.modules, "urban", urban)
    monkeypatch.setattr(_geoalert_notebook, "register_notebook_bricks", lambda: None)
    monkeypatch.setattr(Path, "symlink_to", lambda path, target: path.touch())
    result_path = tmp_path / "result.json"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"pipeline_path": "pipeline.yaml", "compose_root": str(tmp_path / "compose"),
        "progress_path": str(tmp_path / "progress.json"), "result_path": str(result_path),
        "scenes": [{"scene_id": "снимок", "image_path": str(tmp_path / "input.tif")}],
        "outputs": [{"key": "output", "filename": "output.geojson"}]}), encoding="utf-8")
    assert _geoalert_compose_runner.main(["--spec", str(spec)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["reports"][0]["performance"] == performance


@pytest.mark.parametrize("previous_contract", [6, 7])
def test_v2_reexports_old_model_and_batch_change_only_rebuilds_pipeline(tmp_path, monkeypatch, previous_contract):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    exports = []
    repository, pipeline_root = tmp_path / "models", tmp_path / "pipelines"
    pipeline_root.mkdir()
    previous_identity = {"runtime_contract": previous_contract, "checkpoint_sha256": checkpoint_sha, "tile_size": 1024,
                         "threshold": .5, "external": None, "python_site_packages": None}
    previous_digest = hashlib.sha256(json.dumps(previous_identity, sort_keys=True, ensure_ascii=False,
                                                default=str).encode("utf-8")).hexdigest()
    old_name = f"mls_ortho_{previous_digest[:24]}"
    old_dir = repository / old_name
    (old_dir / "1").mkdir(parents=True)
    (old_dir / "config.pbtxt").write_text('dims: [ 1, 3, -1, -1 ]', encoding="utf-8")
    (old_dir / "1" / "model.onnx").write_bytes(b"static-batch-one")
    (pipeline_root / f"{old_name}.model.json").write_text(json.dumps({
        "identity_sha256": previous_digest, "input_channels": 3, "sample_size": 1024,
        "output_kind": "object_probabilities", "object_separation": {"profile": "object_f1_v2"},
        "threshold": .5,
    }), encoding="utf-8")

    def export(**kwargs):
        name = kwargs["model_name"]
        exports.append(name)
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, "w") as archive:
            archive.writestr(f"{name}/config.pbtxt", 'dims: [ -1, 3, -1, -1 ]')
            archive.writestr(f"{name}/1/model.onnx", b"dynamic-batch")
        archive_path = tmp_path / "export.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("export_metadata.json", json.dumps({
                "model_archive": "model.zip", "pipeline": "pipeline.yaml", "input_channels": 3,
                "sample_size": 1024, "task": "binary", "output_kind": "object_probabilities",
                "threshold": .5, "pipeline_variant": "object_f1_v2",
                "object_separation": {"profile": "object_f1_v2"},
            }))
            archive.writestr("model.zip", nested.getvalue())
            archive.writestr("pipeline.yaml", _model_export.build_geoalert_pipeline_yaml(
                model_name=name, sample_size=1024, input_channels=3, object_output=True,
                object_separation={"profile": "object_f1_v2"}, batch_size=kwargs["inference_batch_size"],
            ))
        return SimpleNamespace(zip_path=archive_path, cleanup=lambda: None)

    monkeypatch.setattr(_geoalert_runner, "build_triton_model_export_zip", export)
    config = {"geoalert_model_repository": str(repository), "geoalert_pipeline_root": str(pipeline_root),
              "tile_size": 1024, "threshold": .5, "pipeline_variant": "object_f1_v2", "batch_size": 2}
    first = _geoalert_runner._ensure_runtime_export(config, checkpoint, external_manifest=None, postprocess_config={})
    assert first.model_name != old_name
    assert (old_dir / "1" / "model.onnx").read_bytes() == b"static-batch-one"
    model_path = first.model_dir / "1" / "model.onnx"
    mtime = model_path.stat().st_mtime_ns
    assert _geoalert_runner._ensure_runtime_export(
        config, checkpoint, external_manifest=None, postprocess_config={}) == first
    second = _geoalert_runner._ensure_runtime_export(
        {**config, "batch_size": 4}, checkpoint, external_manifest=None, postprocess_config={})
    assert second.model_name == first.model_name
    assert second.pipeline_path != first.pipeline_path
    assert exports == [first.model_name]
    assert model_path.read_bytes() == b"dynamic-batch" and model_path.stat().st_mtime_ns == mtime
    assert yaml.safe_load(first.pipeline_path.read_text(encoding="utf-8"))["config"]["bricks"][0]["batch_size"] == 2
    assert yaml.safe_load(second.pipeline_path.read_text(encoding="utf-8"))["config"]["bricks"][0]["batch_size"] == 4


@pytest.mark.parametrize("failed_prediction", [False, True])
def test_performance_write_failure_does_not_change_inference_outcome(tmp_path, monkeypatch, failed_prediction):
    _image(tmp_path, 20, 20)
    original_write = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "object_inference_performance.json":
            raise OSError("Замер нельзя записать")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)

    def adapter(images):
        if failed_prediction:
            raise RuntimeError("Исходная ошибка инференса")
        return _predict(images)

    if failed_prediction:
        with pytest.raises(RuntimeError, match="Исходная ошибка инференса"):
            _geoalert_notebook._object_instance_segmentation(
                tmp_path, "input", "instances", adapter, input_channels=3, window_size=32,
                threshold=.5, separation={"profile": "object_f1_v2"}, batch_size=1,
            )
    else:
        _geoalert_notebook._object_instance_segmentation(
            tmp_path, "input", "instances", adapter, input_channels=3, window_size=32,
            threshold=.5, separation={"profile": "object_f1_v2"}, batch_size=1,
        )
        assert (tmp_path / "instances.tif").is_file()
