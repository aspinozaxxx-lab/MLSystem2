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
        "smp_segformer_b1",
        "smp_segformer_b2",
        "smp_segformer_b3",
        "smp_deeplabv3plus_resnet50",
        "smp_unet_resnet34",
        "smp_unet_resnet50",
        "smp_unet_resnet101",
        "smp_unet_resnet152",
    ]


def test_create_model_rejects_other_architectures() -> None:
    with pytest.raises(ModelsError):
        create_model(ModelSpec(name="unet", input_channels=4, output_channels=1))


def test_create_segformer_b2_three_channel_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    handle = create_model(
        ModelSpec(
            name="segformer_b2",
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1


def test_create_segformer_b0_three_channel_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    handle = create_model(
        ModelSpec(
            name="segformer_b0",
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1


def test_create_smp_segformer_b0_three_channel_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b0",
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
    assert outputs.shape == (1, 1, 128, 128)


def test_create_smp_segformer_b1_three_channel_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b1",
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
    assert outputs.shape == (1, 1, 128, 128)


def test_create_smp_segformer_b1_uses_mit_b1_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlsystem2.models import _factory

    seen: dict[str, object] = {}

    class FakeSMP:
        @staticmethod
        def Segformer(**kwargs):
            seen.update(kwargs)
            return object()

    monkeypatch.setattr(_factory, "_import_smp", lambda: FakeSMP())
    monkeypatch.setattr(_factory, "_ensure_torch_for_smp", lambda: None)

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b1",
            input_channels=4,
            output_channels=3,
            pretrained=False,
        )
    )

    assert handle.model is not None
    assert seen == {
        "encoder_name": "mit_b1",
        "encoder_weights": None,
        "in_channels": 4,
        "classes": 3,
        "activation": None,
    }


def test_create_smp_deeplabv3plus_resnet50_three_channel_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_deeplabv3plus_resnet50",
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    handle.model.eval()
    with torch.no_grad():
        outputs = handle.model(torch.zeros((1, 3, 256, 256), dtype=torch.float32))

    assert outputs.shape == (1, 1, 256, 256)


def test_create_smp_segformer_b2_three_channel_multiclass_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b2",
            input_channels=3,
            output_channels=14,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
    assert outputs.shape == (1, 14, 128, 128)


def test_create_smp_segformer_b3_three_channel_forward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name="smp_segformer_b3",
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    outputs = handle.model(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
    assert outputs.shape == (1, 1, 128, 128)


@pytest.mark.parametrize(
    ("model_name", "spatial_size"),
    [
        ("smp_unet_resnet34", 64),
        ("smp_unet_resnet50", 64),
        ("smp_unet_resnet101", 64),
        ("smp_unet_resnet152", 64),
    ],
)
def test_create_smp_unet_three_channel_forward(model_name: str, spatial_size: int) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    handle = create_model(
        ModelSpec(
            name=model_name,
            input_channels=3,
            output_channels=1,
            pretrained=False,
        )
    )

    handle.model.eval()
    with torch.no_grad():
        outputs = handle.model(torch.zeros((1, 3, spatial_size, spatial_size), dtype=torch.float32))

    assert outputs.shape == (1, 1, spatial_size, spatial_size)


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
