"""Проверки SegFormer с весами и без них в обоих конвейерах."""

from types import SimpleNamespace

import pytest

from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec, SaveCheckpointRequest
from mlsystem2.training_ui_api import _model_export, _pseudo_runner
from mlsystem2.training_ui_api._templates import next_gen2_train_batch_size


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("variant", ["legacy", "next_gen2"])
@pytest.mark.parametrize("pretrained", [True, False])
def test_smp_trains_and_restores_without_pretrained_download(
    tmp_path, monkeypatch, architecture, channels, variant, pretrained,
):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(2)
    constructor = smp.Segformer
    initialization = []
    rgb_weights = []

    def build(**kwargs):
        initialization.append(kwargs["encoder_weights"])
        # Проверяем адаптацию настоящей сети без сетевых загрузок в CI.
        model = constructor(**{**kwargs, "encoder_weights": None})
        if kwargs["encoder_weights"] == "imagenet":
            rgb_weights.append(model.encoder.patch_embed1.proj.weight.detach().clone())
        return model

    monkeypatch.setattr(smp, "Segformer", build)
    handle = create_model(ModelSpec(
        name=architecture, input_channels=channels, output_channels=1, pretrained=pretrained,
        parameters=({"pipeline_variant": "next_gen2",
                    "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}}
                    if variant == "next_gen2" else {}),
    ))
    core = handle.model.model if pretrained or variant == "next_gen2" else handle.model
    input_weights = core.encoder.patch_embed1.proj.weight
    if pretrained:
        torch.testing.assert_close(input_weights[:, :3], rgb_weights[0], rtol=0, atol=0)
        if channels == 4:
            torch.testing.assert_close(input_weights[:, 3], rgb_weights[0][:, 0], rtol=0, atol=0)
    images = torch.rand(1, channels, 64, 64) * 255
    images[:, :, 0, 0] = 0
    masks = torch.randint(0, 2, (1, 64, 64))
    optimizer = torch.optim.AdamW(handle.model.parameters(), lr=1e-4)
    inputs = []
    hook = core.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    logits = handle.model(images, return_two_class_logits=True) if variant == "next_gen2" else handle.model(images)
    hook.remove()
    if variant == "next_gen2":
        assert logits.shape == (1, 2, 64, 64)
        loss = 0.25 * torch.nn.functional.cross_entropy(logits, masks) + 0.75 * smp.losses.TverskyLoss(
            mode="multiclass", alpha=0.75, beta=0.25,
        )(logits, masks)
        expected_input = images / images.amax(dim=(-2, -1), keepdim=True)
    else:
        assert logits.shape == (1, 1, 64, 64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks[:, None].float())
        expected_input = images
        if pretrained:
            mean = torch.tensor([0.485, 0.456, 0.406, 0.485][:channels]).view(1, -1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225, 0.229][:channels]).view(1, -1, 1, 1)
            expected_input = (images / 255 - mean) / std
        assert not (logits[:, :, 0, 0] == -1000).any()
    torch.testing.assert_close(inputs[0], expected_input)
    loss.backward()
    gradient = core.segmentation_head[0].weight.grad
    assert torch.isfinite(loss) and all(gradient[index].abs().sum() > 0 for index in range(logits.shape[1]))
    optimizer.step()
    handle.model.eval()
    with torch.no_grad():
        expected = handle.model(images)
        if variant == "next_gen2":
            logits = handle.model(images, return_two_class_logits=True)
            torch.testing.assert_close(torch.sigmoid(expected), logits.softmax(dim=1)[:, 1:2])
    path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(SaveCheckpointRequest(model=handle, checkpoint_uri=path, metadata={}))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=path, map_location="cpu", model_spec=handle.spec))
    restored.model.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.model.model(images), expected, rtol=0, atol=0)
    assert initialization == ["imagenet" if pretrained else None, None]


def test_pretrained_legacy_keeps_multiclass_head_and_raw_checkpoint_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(2)
    constructor = smp.Segformer
    monkeypatch.setattr(smp, "Segformer", lambda **kwargs: constructor(**{**kwargs, "encoder_weights": None}))
    spec = ModelSpec(name="smp_segformer_b2", input_channels=3, output_channels=3, pretrained=True)
    handle = create_model(spec)
    images = torch.rand(1, 3, 64, 64) * 255
    logits = handle.model(images)
    assert logits.shape == (1, 3, 64, 64)
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 3, (1, 64, 64)))
    loss.backward()
    assert torch.isfinite(loss)
    path = str(tmp_path / "multiclass.pt")
    save_checkpoint(SaveCheckpointRequest(model=handle, checkpoint_uri=path))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=path, model_spec=spec))
    handle.model.eval()
    restored.model.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.model.model(images), handle.model(images), rtol=0, atol=0)


@pytest.mark.parametrize("architecture,channels", [("smp_segformer_b0", 3), ("smp_segformer_b3", 4)])
@pytest.mark.parametrize("variant", ["legacy", "next_gen2"])
def test_smp_exports_probabilities_for_geoalert(tmp_path, monkeypatch, architecture, channels, variant):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    constructor = smp.Segformer
    monkeypatch.setattr(smp, "Segformer", lambda **kwargs: constructor(**{**kwargs, "encoder_weights": None}))
    onnx = pytest.importorskip("onnx")
    torch.set_num_threads(2)
    handle = create_model(ModelSpec(
        name=architecture, input_channels=channels, output_channels=1, pretrained=variant == "legacy",
        parameters=({"pipeline_variant": "next_gen2",
                    "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}}
                    if variant == "next_gen2" else {}),
    ))
    path = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(
        model=handle.model, input_channels=channels, output_channels=1, sample_size=64,
        threshold=0.9, onnx_path=path, probability_output=True,
    )
    exported = onnx.load(str(path))
    onnx.checker.check_model(exported)
    operations = {node.op_type for node in exported.graph.node}
    assert "Sigmoid" in operations
    if variant == "next_gen2":
        assert {"ReduceMin", "ReduceMax"} <= operations
    else:
        assert {"Div", "Sub"} <= operations
    output = exported.graph.output[0]
    assert output.name == "probabilities"
    assert output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert output.type.tensor_type.shape.dim[1].dim_value == 1


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("tile_size", [512, 768, 1024, 1536])
def test_inference_batch_follows_checkpoint_architecture(architecture, tile_size):
    loaded = SimpleNamespace(
        model=SimpleNamespace(spec=ModelSpec(
            name=architecture, input_channels=3, output_channels=1,
            parameters={"pipeline_variant": "next_gen2"},
        )), artifact=SimpleNamespace(metadata={"sample_size": tile_size}),
    )
    resolved = _pseudo_runner._native_inference_config(loaded, {"batch_size": 32})
    assert resolved["batch_size"] == 2 * next_gen2_train_batch_size(tile_size, architecture)
    assert resolved["threshold"] == 0.9
