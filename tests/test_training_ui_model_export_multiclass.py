from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from mlsystem2.training_ui_api import _model_export


def test_multiclass_onnx_matches_softmax_argmax_threshold(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("onnx.reference")

    class DeterministicModel(torch.nn.Module):
        def forward(self, images):
            background = torch.zeros_like(images[:, :1])
            return torch.cat((background, images[:, :1], images[:, 1:2]), dim=1)

    threshold = 0.7
    model_path = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(
        model=DeterministicModel(),
        input_channels=4,
        output_channels=3,
        sample_size=8,
        threshold=threshold,
        onnx_path=model_path,
    )
    images = np.zeros((1, 4, 8, 8), dtype=np.float32)
    images[:, 0, :4, :] = 4.0
    images[:, 1, 4:, :] = 4.0
    images[:, 0, 0, 0] = 0.2
    images[:, 1, 0, 0] = 0.1

    logits = np.concatenate(
        (np.zeros_like(images[:, :1]), images[:, :1], images[:, 1:2]),
        axis=1,
    )
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    labels[(labels > 0) & (confidence < threshold)] = 0
    expected = np.stack((labels == 1, labels == 2), axis=1).astype(np.uint8)

    evaluator = reference.ReferenceEvaluator(onnx.load(model_path))
    actual = evaluator.run(["mask"], {"input": images})[0]

    assert actual.dtype == np.uint8
    assert actual.shape == (1, 2, 8, 8)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual.sum(axis=1) <= 1)


def test_multiclass_pipeline_has_two_semantic_outputs() -> None:
    schema = [
        {"id": 1, "slug": "flooding", "name": "Переувлажнения"},
        {"id": 2, "slug": "waterlogging", "name": "Заболачивание"},
    ]

    pipeline = yaml.safe_load(
        _model_export._pipeline_yaml(
            "wetlands-b2",
            512,
            4,
            class_schema=schema,
        )
    )["config"]

    assert pipeline["outputs"] == ["flooding.geojson", "waterlogging.geojson"]
    segmentation = pipeline["bricks"][1]
    assert segmentation["output_labels"] == ["flooding", "waterlogging"]
    assert segmentation["adapter"]["output_dtype"] == "uint8"
