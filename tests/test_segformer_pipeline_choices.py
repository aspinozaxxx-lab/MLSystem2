"""Проверки новых архитектур next-gen2 и совместимости сохранённых моделей."""

from types import SimpleNamespace

import pytest

from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec, SaveCheckpointRequest
from mlsystem2.training_ui_api import _model_export, _pseudo_runner
from mlsystem2.training_ui_api._templates import next_gen2_train_batch_size


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("channels", [3, 4])
def test_smp_next_gen2_trains_both_heads_and_restores_without_pretrained_download(
    tmp_path, monkeypatch, architecture, channels,
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
        name=architecture, input_channels=channels, output_channels=1, pretrained=True,
        parameters={"pipeline_variant": "next_gen2",
                    "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}},
    ))
    input_weights = handle.model.model.encoder.patch_embed1.proj.weight
    torch.testing.assert_close(input_weights[:, :3], rgb_weights[0], rtol=0, atol=0)
    if channels == 4:
        torch.testing.assert_close(input_weights[:, 3], rgb_weights[0][:, 0], rtol=0, atol=0)
    images = torch.rand(1, channels, 64, 64) * 255
    masks = torch.randint(0, 2, (1, 64, 64))
    optimizer = torch.optim.AdamW(handle.model.parameters(), lr=1e-4)
    logits = handle.model(images, return_two_class_logits=True)
    assert logits.shape == (1, 2, 64, 64)
    loss = 0.25 * torch.nn.functional.cross_entropy(logits, masks) + 0.75 * smp.losses.TverskyLoss(
        mode="multiclass", alpha=0.75, beta=0.25,
    )(logits, masks)
    loss.backward()
    gradient = handle.model.model.segmentation_head[0].weight.grad
    assert torch.isfinite(loss) and all(gradient[index].abs().sum() > 0 for index in (0, 1))
    optimizer.step()
    handle.model.eval()
    with torch.no_grad():
        logits = handle.model(images, return_two_class_logits=True)
        expected = handle.model(images)
        torch.testing.assert_close(torch.sigmoid(expected), logits.softmax(dim=1)[:, 1:2])
    path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(SaveCheckpointRequest(model=handle, checkpoint_uri=path, metadata={}))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=path, map_location="cpu"))
    restored.model.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.model.model(images), expected, rtol=0, atol=0)
    assert initialization == ["imagenet", None]


@pytest.mark.parametrize("architecture,channels", [("smp_segformer_b0", 3), ("smp_segformer_b3", 4)])
def test_smp_next_gen2_exports_probabilities_for_geoalert(tmp_path, architecture, channels):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    onnx = pytest.importorskip("onnx")
    torch.set_num_threads(2)
    handle = create_model(ModelSpec(
        name=architecture, input_channels=channels, output_channels=1,
        parameters={"pipeline_variant": "next_gen2",
                    "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}},
    ))
    path = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(
        model=handle.model, input_channels=channels, output_channels=1, sample_size=64,
        threshold=0.9, onnx_path=path, probability_output=True,
    )
    exported = onnx.load(str(path))
    onnx.checker.check_model(exported)
    assert {"ReduceMin", "ReduceMax", "Sigmoid"} <= {node.op_type for node in exported.graph.node}
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
