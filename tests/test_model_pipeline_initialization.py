"""Перенос всех весов, сохранение привычной калибровки и запрет несовместимых голов."""

import pytest
from types import SimpleNamespace

from mlsystem2.models.api import compatible_training_pipelines, create_model, initialize_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import CheckpointArtifact, LoadCheckpointRequest, LoadedCheckpoint, ModelSpec, ModelsError, SaveCheckpointRequest


def _spec(variant="next_gen2", channels=4, outputs=1):
    preprocessing = {"mode":"window_minmax", "epsilon":1e-6}
    if variant == "next_gen4":
        preprocessing["mask_output_nodata"] = True
    if variant == "next_gen4_global_norm":
        preprocessing = {"mode":"global_minmax", "epsilon":1e-6,"nodata":0.0,"mask_output_nodata":True,
            "low":[10.0] * channels,"high":[200.0] * channels,"source":"retained_train_windows",
            "valid_pixel_count":1000000,"window_count":100}
    return ModelSpec(name="smp_segformer_b0", input_channels=channels, output_channels=outputs,
        parameters={"pipeline_variant":variant,"preprocessing":preprocessing})


@pytest.mark.parametrize("variant,allowed,forbidden", [
    ("next_gen2","next_gen4_global_norm","next_gen4_consistency_v7"),
    ("next_gen4_consistency_v7","next_gen4_consistency_v2","next_gen2"),
    ("object_f1_v2","object_f1_v2","object_f1"),
    ("object_f1","object_f1","next_gen2"),
    ("legacy","legacy","next_gen2"),
])
def test_compatible_profiles_respect_head_meaning_and_decoder(variant, allowed, forbidden):
    profiles = compatible_training_pipelines(_spec(variant))
    assert allowed in profiles and forbidden not in profiles


def test_multiclass_legacy_preserves_semantic_output_contract():
    assert "next_gen4" in compatible_training_pipelines(_spec("legacy", outputs=3))


@pytest.fixture
def tiny_segformer(monkeypatch):
    torch = pytest.importorskip("torch")
    from mlsystem2.models import _factory

    def make(spec, **_kwargs):
        return torch.nn.Sequential(torch.nn.Conv2d(spec.input_channels, 4, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(4, 2 if spec.output_channels == 1 else spec.output_channels, 1))

    monkeypatch.setattr(_factory, "_create_smp_segformer", make)
    return torch


def test_full_weights_transfer_rebuilds_target_preprocessing_and_round_trips(tiny_segformer, tmp_path):
    torch = tiny_segformer
    source = create_model(_spec())
    target = initialize_model(source, _spec("next_gen4"))
    for key, value in source.model.model.state_dict().items():
        assert torch.equal(value, target.model.model.state_dict()[key])
    assert target.model.mask_output_nodata is True and source.model.mask_output_nodata is False
    target.model.eval()
    image = torch.randint(1, 255, (1, 4, 32, 32)).float()
    expected = target.model(image).detach()
    path = tmp_path / "новый-чекпойнт.pt"
    save_checkpoint(SaveCheckpointRequest(model=target, checkpoint_uri=str(path)))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))
    restored.model.model.eval()
    assert torch.equal(restored.model.model(image), expected)
    assert restored.model.spec.parameters["weight_initialization"]["target_pipeline_variant"] == "next_gen4"


@pytest.mark.parametrize("preserve", [True, False])
def test_global_calibration_preserved_unless_recalculation_is_explicit(tiny_segformer, preserve):
    torch = tiny_segformer
    source = create_model(_spec("next_gen4_global_norm"))
    candidate = _spec("next_gen4_global_norm")
    candidate.parameters["preprocessing"]["low"] = [80.0] * 4
    candidate.parameters["preprocessing"]["high"] = [100.0] * 4
    target = initialize_model(source, candidate, preserve_global_preprocessing=preserve)
    assert target.model.preprocess_low.flatten().tolist() == ([10.0] * 4 if preserve else [80.0] * 4)
    source.model.eval()
    target.model.eval()
    image = torch.randint(1, 255, (1, 4, 32, 32)).float()
    if preserve:
        assert torch.equal(source.model(image), target.model(image))
    for key, value in source.model.model.state_dict().items():
        assert torch.equal(value, target.model.model.state_dict()[key])


def test_incompatible_heads_and_decoders_are_rejected_before_creation(tiny_segformer):
    source = create_model(_spec())
    with pytest.raises(ModelsError, match="несовместим"):
        initialize_model(source, _spec("object_f1_v2"))
    with pytest.raises(ModelsError, match="несовместим"):
        initialize_model(source, _spec("next_gen4_consistency_v2"))


def test_pipeline_preserves_global_calibration_without_new_histogram(tiny_segformer):
    from mlsystem2.train_pipeline import _runner

    source = create_model(_spec("next_gen4_global_norm"))
    settings = SimpleNamespace(tile_preparation=SimpleNamespace(tile_size=512), train=SimpleNamespace(
        initial_checkpoint_uri="исходные-веса.pt", initial_checkpoint_weights_only=True,
        initial_checkpoint_recalculate_normalization=False, pipeline_variant="next_gen4_global_norm",
        input_channels=4, output_channels=1, pretrained=False, model_name="smp_segformer_b0", device="cpu"))
    deps = SimpleNamespace(load_checkpoint=lambda request: LoadedCheckpoint(model=source,
        artifact=CheckpointArtifact(uri=request.checkpoint_uri, format="torch_pt")), initialize_model=initialize_model)
    target = _runner._load_or_create_model(settings, deps)
    assert target.spec.parameters["preprocessing"] == source.spec.parameters["preprocessing"]
    settings.train.initial_checkpoint_recalculate_normalization = True
    with pytest.raises(ValueError, match="статистику"):
        _runner._load_or_create_model(settings, deps)


def test_real_smp_b0_transfer_keeps_every_weight_offline(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    source = create_model(_spec())
    target = initialize_model(source, _spec("next_gen4_global_norm"))
    for key, value in source.model.model.state_dict().items():
        assert torch.equal(value, target.model.model.state_dict()[key])
    path = tmp_path / "полные-веса.pt"
    save_checkpoint(SaveCheckpointRequest(model=target, checkpoint_uri=str(path)))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))
    for key, value in target.model.state_dict().items():
        assert torch.equal(value, restored.model.model.state_dict()[key])


def test_real_layer_norm_profiles_keep_inference_identical_with_preserved_calibration():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from test_next_gen4_consistency_v2 import spec

    source = create_model(spec("next_gen4_consistency_v2"))
    target_spec = spec("next_gen4_consistency_v7")
    target_spec.parameters["preprocessing"]["low"] = [80.0] * 4
    target = initialize_model(source, target_spec)
    assert target.model.allow_context_reference and target.model.supervise_context_reference
    assert target.model.preprocess_low.flatten().tolist() == [0.0] * 4
    source.model.eval()
    target.model.eval()
    image = torch.randint(1, 255, (1, 4, 64, 64)).float()
    with torch.no_grad():
        assert torch.equal(source.model(image), target.model(image))


def test_legacy_initialization_keeps_exact_spec_for_following_continuations(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    spec = ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1)
    source = create_model(spec)
    target = initialize_model(source, spec)
    assert target.spec == spec
    path = tmp_path / "legacy.pt"
    save_checkpoint(SaveCheckpointRequest(model=target, checkpoint_uri=str(path)))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), model_spec=spec, map_location="cpu"))
    assert restored.model.spec == spec


@pytest.mark.parametrize("count,volume,clipped,affected,expected", [
    (2, 100000, .9, 2, "preserve"), (10, 10000, .9, 10, "preserve"),
    (10, 1000000, .01, 10, "preserve"), (10, 1000000, .4, 2, "preserve"),
    (10, 1000000, .4, 8, "recompute"),
])
def test_normalization_advice_is_conservative_on_small_sets_and_local_outliers(count, volume, clipped, affected, expected):
    from mlsystem2.training_ui_api._continuation_normalization import _recommend

    choice, reason = _recommend({"source_count":count,"sampled_valid_pixels":100000,"estimated_valid_pixels":volume,
        "sampled_scene_count":count,"clipped_fractions":[clipped] * 4}, {"valid_pixel_count":1000000}, affected)
    assert choice == expected and reason
