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
@pytest.mark.parametrize("tile_size,channels", [(512, 4), (768, 3), (1024, 4), (1536, 3)])
@pytest.mark.parametrize("variant", ["next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"])
def test_archive_uses_eval_profile_and_consistent_float_output(monkeypatch, override, tile_size, channels, variant):
    metadata = {"pipeline_variant": variant, "sample_size": tile_size,
                "inference_context": 0, "confidence_threshold": 0.5}
    loaded = SimpleNamespace(
        artifact=SimpleNamespace(metadata=metadata),
        model=SimpleNamespace(model=object(), spec=SimpleNamespace(
            input_channels=channels, output_channels=1, parameters={"task": "binary"})))
    if variant in {"next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
        loaded.model.spec.parameters["preprocessing"] = {"mode": "global_minmax", "low": [10] * channels,
            "high": [200] * channels, "source": "retained_train_windows"}
    if variant == "next_gen4_augmentation_v3":
        loaded.model.spec.parameters["augmentation_policy"] = "validity_guard_v3"
    if variant in {"next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
        loaded.model.spec.parameters.update(augmentation_policy="validity_guard_v4_context_crop_v1", context_crop={
            "area_scale": [0.25, 1.0], "ratio": [1.0, 1.0], "probability": 0.5,
            "output_size": tile_size, "source": "within_retained_train_window"})
    if variant in {"next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4"}:
        loaded.model.spec.parameters["context_consistency"] = {
            "version": ("bidirectional_confident_teacher_kl_with_gt_fallback_v1" if variant == "next_gen4_consistency_v4"
                        else "bidirectional_confident_teacher_kl_v1" if variant == "next_gen4_consistency_v3" else "confident_teacher_kl_v1"), "weight": 0.25, "probability": 0.5,
            "crop_side_fractions": [0.5, 0.75], "teacher_confidence": 0.9,
            "source": "within_retained_train_window", "evaluation_mode": True, "class_balance": True,
            **({"direction_reduction": "mean_of_active_directions"} if variant in {"next_gen4_consistency_v3", "next_gen4_consistency_v4"} else {}),
            **({"uncovered_supervision": "ground_truth_cross_entropy_v1"} if variant == "next_gen4_consistency_v4" else {})}
    if variant in {"next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
        loaded.model.spec.parameters["context_consistency"] = {
            "version": "full_query_detached_crop_kv_with_reference_gt_v1" if variant == "next_gen4_consistency_v6" else "full_query_detached_crop_kv_gt_v1", "weight": .25, "probability": .5,
            "crop_side_fractions": [.5, .75], "source": "within_retained_train_window",
            "evaluation_mode": True, "class_balance": True, "reference": "detached_same_model_crop_kv_v1",
            "supervision": "query_and_reference_ground_truth_cross_entropy_v1" if variant == "next_gen4_consistency_v6" else "full_query_ground_truth_cross_entropy_v1",
            **({"context_reduction": "mean_of_query_and_reference"} if variant == "next_gen4_consistency_v6" else {})}
    if variant in {"next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
        loaded.model.spec.parameters["decoder_normalization"] = {
            "version": "per_pixel_channel_layer_norm_v1", "epsilon": 1e-5, "affine": True}
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
        assert info["pipeline_variant"] == variant
        if variant in {"next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
            assert info["context_consistency"] == loaded.model.spec.parameters["context_consistency"]
        if variant in {"next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
            assert info["decoder_normalization"] == loaded.model.spec.parameters["decoder_normalization"]
        if variant in {"next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
            assert info["preprocessing"] == loaded.model.spec.parameters["preprocessing"]
        if variant == "next_gen4_augmentation_v3":
            assert info["augmentation_policy"] == "validity_guard_v3"
        if variant in {"next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
            assert info["augmentation_policy"] == "validity_guard_v4_context_crop_v1"
            assert info["context_crop"] == loaded.model.spec.parameters["context_crop"]
        assert info["threshold_source"] == ("next_gen2_eval_notebook" if override is None else "request")
        assert info["inference_stride"] == tile_size // 2
        assert calls[0]["input_channels"] == channels
        assert calls[0]["sample_size"] == tile_size
        assert info["output_kind"] == "probabilities"
        assert calls[0]["probability_output"] is True
        assert 'name: "probabilities"' in config and "TYPE_UINT8" not in config
        split, segmentation, threshold, vectorize, clip = pipeline["bricks"]
        assert split["apply_mask"] is False
        assert segmentation["_class"] == "SlidingWindowSegmentation"
        assert segmentation["window_size"] == tile_size and segmentation["stride"] == tile_size // 2
        assert segmentation["adapter"]["output_dtype"] == "float32"
        assert segmentation["pad_edges"] is (variant in {"next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"})
        assert threshold["_class"] == "MultiThresholding"
        assert threshold["thresholds"] == [expected_threshold] and threshold["strict_more"]
        assert vectorize["value_property_name"] == "class_id"
        assert clip == {"_class": "ClipToRasterValidData", "input_raster": "input", "vectors": vectorize["output_fcs"]}
        assert info["inference_profile"] == "notebook_valid_data_v2"
    finally:
        archive.cleanup()


def test_multiclass_archive_keeps_background_and_named_class_outputs(monkeypatch):
    schema = [{"id": i, "slug": f"type_{i}", "name": f"Тип {i}", "color": "#00ff66", "priority": 0} for i in range(1, 4)]
    loaded = SimpleNamespace(artifact=SimpleNamespace(metadata={"pipeline_variant": "next_gen2", "sample_size": 512,
        "inference_context": 0, "task": "multiclass", "class_schema": schema, "confidence_threshold": 0.9}),
        model=SimpleNamespace(model=object(), spec=SimpleNamespace(input_channels=3, output_channels=4, parameters={})))
    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", lambda _: loaded)
    calls = []
    def export(**kwargs):
        calls.append(kwargs)
        kwargs["onnx_path"].write_bytes(b"test-onnx")
    monkeypatch.setattr(_model_export, "_export_segmentation_mask_onnx", export)
    archive = _model_export.build_triton_model_export_zip(model_name="multitype", checkpoint_filename="best.pt", checkpoint_bytes=b"weights", sample_size=None)
    try:
        with zipfile.ZipFile(archive.zip_path) as outer:
            info = json.loads(outer.read("export_metadata.json"))
            pipeline = yaml.safe_load(outer.read("pipelines/multitype_triton.yaml"))["config"]
            assert "runtime/mlsystem2/training_ui_api/_geoalert_notebook.py" in outer.namelist()
            with zipfile.ZipFile(io.BytesIO(outer.read("models-serving-service/multitype.zip"))) as inner:
                config = inner.read("multitype/config.pbtxt").decode()
        assert calls[0]["output_channels"] == 4 and calls[0]["probability_output"]
        assert info["task"] == "multiclass" and info["output_channels"] == ["background", "type_1", "type_2", "type_3"]
        assert "dims: [ -1, 4, -1, -1 ]" in config
        _, windows, choose, vectorize, clip = pipeline["bricks"]
        assert len(windows["output_labels"]) == 4
        assert choose["_class"] == "MulticlassProbabilityMasks" and choose["threshold"] == 0.9
        assert choose["output_masks"] == ["type_1", "type_2", "type_3"]
        assert vectorize["output_fcs"] == ["type_1", "type_2", "type_3"]
        assert clip["vectors"] == ["type_1", "type_2", "type_3"]
    finally:
        archive.cleanup()
