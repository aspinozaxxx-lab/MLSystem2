"""Отдельный нормализатор декодера, строгие веса и прежний конвейер подготовки."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlsystem2.models import _factory
from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelsError, SaveCheckpointRequest
from mlsystem2.train_pipeline import _runner
from mlsystem2.train_pipeline.contracts import TrainPipelineError
from mlsystem2.training_ui_api import _model_export, _service
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError
from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config

VARIANT = "next_gen4_consistency_v2"
NORMALIZATION = {"version": "per_pixel_channel_layer_norm_v1", "epsilon": 1e-5, "affine": True}


def spec(variant=VARIANT, architecture="smp_segformer_b0"):
    settings = SimpleNamespace(tile_preparation=SimpleNamespace(tile_size=512), train=SimpleNamespace(
        pipeline_variant=variant, input_channels=4, output_channels=1, pretrained=False, model_name=architecture))
    dataset = SimpleNamespace(band_histogram={"source": "retained_train_windows",
        "counts": [[1] + [0] * 254 + [1] for _ in range(4)], "valid_pixel_count": 2, "window_count": 1})
    return _runner._model_spec(settings, SimpleNamespace(dataset=dataset))


def test_spec_changes_only_variant_and_decoder_normalization():
    before, after = spec("next_gen4_consistency_v1"), spec()
    assert after.parameters == {**before.parameters, "pipeline_variant": VARIANT,
                                "decoder_normalization": NORMALIZATION}


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_profile_registration_and_defaults(architecture, size):
    before = sanitize_template_config({"train.pipeline_variant": "next_gen4_consistency_v1",
                                      "tile_preparation.tile_size": size}, architecture=architecture)
    after = sanitize_template_config({**before, "train.pipeline_variant": VARIANT}, architecture=architecture)
    assert after == {**before, "train.pipeline_variant": VARIANT}
    template = next(item for item in initial_templates() if item["architecture"] == architecture)
    assert VARIANT in template["config_schema"]["fields"][0]["options"]
    assert template["config_schema"]["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "LayerNorm" in template["config_schema"]["pipeline_descriptions"][VARIANT]


@pytest.mark.parametrize("architecture", ["segformer_b0", "segformer_b2", "smp_unet_resnet34"])
def test_unsupported_architecture_is_rejected_before_execution(architecture):
    with pytest.raises(TrainingUIAPIError, match="SMP SegFormer"):
        sanitize_template_config({"train.pipeline_variant": VARIANT}, architecture=architecture)
    with pytest.raises(TrainingUIAPIError, match="SMP SegFormer"):
        _service._validate_training_pipeline_variant({"train.pipeline_variant": VARIANT}, architecture)
    with pytest.raises(ModelsError, match="SMP SegFormer"):
        create_model(spec(architecture=architecture))


@pytest.mark.parametrize("marker", [None, {}, {**NORMALIZATION, "epsilon": 1e-6},
                                   {**NORMALIZATION, "version": "batch_norm"}])
def test_missing_or_different_normalization_is_never_inferred(marker):
    broken = spec()
    broken.parameters["decoder_normalization"] = marker
    with pytest.raises(ModelsError, match="попиксельную нормализацию"):
        create_model(broken)


def test_per_pixel_norm_has_no_batch_spatial_or_training_mode_dependence():
    torch.manual_seed(42)
    norm = _factory._PerPixelChannelLayerNorm(8, eps=1e-5)
    image = torch.randn(1, 8, 9, 11, requires_grad=True)
    baseline = norm(image)
    other = torch.randn_like(image) * 20 + 100
    torch.testing.assert_close(norm(torch.cat((image, other)))[0], baseline[0], rtol=0, atol=0)
    altered = image.detach().clone()
    altered[..., :4, :] += torch.randn_like(altered[..., :4, :]) * 100
    torch.testing.assert_close(norm(altered)[..., 4:, :], baseline[..., 4:, :], rtol=0, atol=0)
    expected = torch.nn.functional.layer_norm(image.permute(0, 2, 3, 1), (8,), norm.weight, norm.bias, 1e-5)
    torch.testing.assert_close(baseline, expected.permute(0, 3, 1, 2))
    norm.eval()
    torch.testing.assert_close(norm(image), baseline, rtol=0, atol=0)
    baseline.square().mean().backward()
    assert torch.isfinite(image.grad).all() and torch.isfinite(norm.weight.grad).all()
    assert not list(norm.named_buffers())


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
def test_real_models_change_only_single_decoder_layer_and_leave_old_weights(architecture):
    pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    before = create_model(spec("next_gen4_consistency_v1", architecture)).model
    torch.manual_seed(42)
    after = create_model(spec(architecture=architecture)).model
    assert isinstance(before.model.decoder.fuse_stage[1], torch.nn.BatchNorm2d)
    assert isinstance(after.model.decoder.fuse_stage[1], _factory._PerPixelChannelLayerNorm)
    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in after.modules())
    previous, current = before.state_dict(), after.state_dict()
    assert set(previous) - set(current) == {
        "model.decoder.fuse_stage.1.running_mean", "model.decoder.fuse_stage.1.running_var",
        "model.decoder.fuse_stage.1.num_batches_tracked"}
    for key in current:
        torch.testing.assert_close(current[key], previous[key], rtol=0, atol=0)


def test_real_checkpoint_reload_is_offline_and_rejects_old_batchnorm_state(tmp_path, monkeypatch):
    pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(2)
    model = create_model(spec())
    model.model.eval()
    image = torch.rand(1, 4, 64, 64) * 254 + 1
    image[..., :4, :4] = 0
    with torch.no_grad():
        expected = model.model(image)
    path = tmp_path / "новые-веса.pt"
    save_checkpoint(SaveCheckpointRequest(model=model, checkpoint_uri=str(path), metadata={}))
    # Автономная загрузка не должна обращаться за предобучением.
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *args, **kwargs: pytest.fail("Сетевая загрузка"))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))
    restored.model.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.model.model(image), expected)
    assert (expected[..., :4, :4] == -1000).all() and torch.isfinite(expected).all()
    assert restored.model.spec.parameters["decoder_normalization"] == NORMALIZATION
    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"]["model.decoder.fuse_stage.1.running_mean"] = torch.zeros(256)
    torch.save(payload, path)
    with pytest.raises(ModelsError, match="загрузить веса"):
        load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))


def test_continuation_rejects_other_decoder_normalization(monkeypatch):
    expected = spec()
    loaded = SimpleNamespace(spec=expected.model_copy(deep=True))
    loaded.spec.parameters["decoder_normalization"]["epsilon"] = 1e-6
    monkeypatch.setattr(_runner, "_model_spec", lambda *args: expected)
    deps = SimpleNamespace(load_checkpoint=lambda request: SimpleNamespace(model=loaded))
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT,
        initial_checkpoint_uri="best.pt", device="cpu"))
    with pytest.raises(TrainPipelineError, match="Нормализация декодера"):
        _runner._load_or_create_model(settings, deps)


def test_unexpected_decoder_structure_is_rejected(monkeypatch):
    wrong = SimpleNamespace(Segformer=lambda **kwargs: torch.nn.Sequential(torch.nn.BatchNorm2d(4)))
    monkeypatch.setattr(_factory, "_import_smp", lambda: wrong)
    with pytest.raises(ModelsError, match="Неожиданная структура"):
        create_model(spec())


def test_exported_layer_preserves_native_probabilities(tmp_path):
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 1),
        _factory._PerPixelChannelLayerNorm(8, eps=1e-5), torch.nn.Conv2d(8, 1, 1)).eval()
    path = tmp_path / "проверка.onnx"
    _model_export._export_segmentation_mask_onnx(model=model, input_channels=4, output_channels=1,
        sample_size=8, threshold=0.9, onnx_path=path, probability_output=True)
    image = np.random.default_rng(42).normal(size=(1, 4, 8, 8)).astype("float32")
    actual = reference.ReferenceEvaluator(onnx.load(path)).run(["probabilities"], {"input": image})[0]
    with torch.no_grad():
        expected = model(torch.from_numpy(image)).sigmoid().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
