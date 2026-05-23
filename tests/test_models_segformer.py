from __future__ import annotations

import pytest

from mlsystem2.models.api import create_model, list_supported_models
from mlsystem2.models.contracts import ModelSpec, ModelsError


def test_list_supported_models_returns_only_segformer_b2() -> None:
    supported = list_supported_models()

    assert [item.name for item in supported] == ["segformer_b2"]


def test_create_model_rejects_other_architectures() -> None:
    with pytest.raises(ModelsError):
        create_model(ModelSpec(name="unet", input_channels=4, output_channels=1))


def test_create_segformer_b2_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    handle = create_model(
        ModelSpec(
            name="segformer_b2",
            input_channels=4,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 4, 128, 128), dtype=torch.float32))
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1
