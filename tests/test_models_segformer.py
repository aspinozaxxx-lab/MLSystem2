from __future__ import annotations

import pytest

from mlsystem2.models.api import create_model, list_supported_models
from mlsystem2.models.contracts import ModelSpec, ModelsError


def test_list_supported_models_returns_supported_architectures() -> None:
    supported = list_supported_models()

    assert [item.name for item in supported] == [
        "segformer_b0",
        "segformer_b2",
        "smp_segformer_b0",
        "smp_segformer_b2",
        "smp_deeplabv3plus_resnet50",
    ]


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


def test_create_segformer_b0_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    handle = create_model(
        ModelSpec(
            name="segformer_b0",
            input_channels=4,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 4, 128, 128), dtype=torch.float32))
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1


def test_create_smp_segformer_b0_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b0",
            input_channels=4,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 4, 128, 128), dtype=torch.float32))
    assert outputs.shape == (1, 1, 128, 128)


def test_create_smp_deeplabv3plus_resnet50_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_deeplabv3plus_resnet50",
            input_channels=4,
            output_channels=1,
            pretrained=False,
        )
    )

    handle.model.eval()
    with torch.no_grad():
        outputs = handle.model(torch.zeros((1, 4, 256, 256), dtype=torch.float32))

    assert outputs.shape == (1, 1, 256, 256)


def test_create_smp_segformer_b2_multiclass_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b2",
            input_channels=4,
            output_channels=14,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 4, 128, 128), dtype=torch.float32))
    assert outputs.shape == (1, 14, 128, 128)


def test_raw_input_wrapper_scales_uint8_range_to_unit_range() -> None:
    torch = pytest.importorskip("torch")
    from mlsystem2.models._factory import _SegFormerRawInputWrapper

    class Recorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return x

    recorder = Recorder()
    wrapper = _SegFormerRawInputWrapper(recorder)

    output = wrapper(torch.full((1, 4, 2, 2), 255.0))

    assert torch.allclose(recorder.seen, torch.ones((1, 4, 2, 2)))
    assert torch.allclose(output, torch.ones((1, 4, 2, 2)))
