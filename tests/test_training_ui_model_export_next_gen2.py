"""Контракт экспорта next_gen2: вероятности и порог после Gaussian-объединения."""
import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mlsystem2.training_ui_api import _model_export


def test_onnx_preserves_probabilities_instead_of_thresholding(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")

    class Model(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    path = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(
        model=Model(), input_channels=4, output_channels=1, sample_size=8,
        threshold=0.9, onnx_path=path, probability_output=True)
    image = np.linspace(-2, 3, 4 * 8 * 8, dtype="float32").reshape(1, 4, 8, 8)
    model = onnx.load(path)
    actual = reference.ReferenceEvaluator(model).run(["probabilities"], {"input": image})[0]
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, 1 / (1 + np.exp(-image[:, :1])), rtol=1e-6)
    assert model.ir_version == _model_export.ONNX_IR_VERSION


@pytest.mark.parametrize("override", [None, 0.8])
def test_archive_uses_eval_profile_and_consistent_float_output(monkeypatch, override):
    metadata = {"pipeline_variant": "next_gen2", "sample_size": 512,
                "inference_context": 0, "confidence_threshold": 0.5}
    loaded = SimpleNamespace(
        artifact=SimpleNamespace(metadata=metadata),
        model=SimpleNamespace(model=object(), spec=SimpleNamespace(
            input_channels=4, output_channels=1, parameters={"task": "binary"})))
    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", lambda _: loaded)
    monkeypatch.setattr(_model_export, "_checkpoint_task_schema", lambda _: ("binary", []))
    calls = []
    def export(**kwargs):
        calls.append(kwargs)
        kwargs["onnx_path"].write_bytes(b"test-onnx")
    monkeypatch.setattr(_model_export, "_export_segmentation_mask_onnx", export)
    archive = _model_export.build_triton_model_export_zip(
        model_name="floodings", checkpoint_filename="best.pt", checkpoint_bytes=b"checkpoint",
        sample_size=None, threshold=override)
    try:
        with zipfile.ZipFile(archive.zip_path) as outer:
            info = json.loads(outer.read("export_metadata.json"))
            pipeline = yaml.safe_load(outer.read("pipelines/floodings_triton.yaml"))["config"]
            with zipfile.ZipFile(io.BytesIO(outer.read("models-serving-service/floodings.zip"))) as inner:
                config = inner.read("floodings/config.pbtxt").decode()
        expected_threshold = 0.9 if override is None else override
        assert info["threshold"] == expected_threshold
        assert info["threshold_source"] == ("next_gen2_eval_notebook" if override is None else "request")
        assert info["inference_stride"] == 256
        assert info["output_kind"] == "probabilities"
        assert calls[0]["probability_output"] is True
        assert 'name: "probabilities"' in config and "TYPE_UINT8" not in config
        split, segmentation, threshold, vectorize = pipeline["bricks"]
        assert split["apply_mask"] is False
        assert segmentation["_class"] == "SlidingWindowSegmentation"
        assert segmentation["window_size"] == 512 and segmentation["stride"] == 256
        assert segmentation["adapter"]["output_dtype"] == "float32"
        assert threshold["_class"] == "MultiThresholding"
        assert threshold["thresholds"] == [expected_threshold] and threshold["strict_more"]
        assert vectorize["value_property_name"] == "class_id"
    finally:
        archive.cleanup()
