"""Контракты отдельного профиля, согласованные пиксели и непрерывные временные срезы."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mlsystem2.train._consistency import context_consistency_loss
from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config

VARIANT = "next_gen4_consistency_v1"
POLICY = {"version": "confident_teacher_kl_v1", "weight": 0.25, "probability": 0.5,
    "crop_side_fractions": [0.5, 0.75], "teacher_confidence": 0.9,
    "source": "within_retained_train_window", "evaluation_mode": True, "class_balance": True}


class ContextModel(torch.nn.Module):
    def __init__(self, *, correct=True, fail=False):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(-1.0))
        self.bn = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout(0.9)
        self.seen = []
        self.correct, self.fail = correct, fail

    def forward(self, image, *, return_two_class_logits=False):
        self.seen.append(image.detach().clone())
        if self.fail and len(self.seen) == 2:
            raise RuntimeError("Проверочная ошибка второго прохода")
        self.dropout(self.bn(image))
        value = self.bias + (7 if image.shape[-1] == 128 else 0)
        if not self.correct:
            value = -value
        foreground = value.expand(len(image), 1, *image.shape[-2:])
        return torch.cat((-foreground, foreground), dim=1)


def inputs():
    image = torch.arange(128 * 128).reshape(1, 1, 128, 128).float().expand(1, 4, 128, 128).clone() + 1
    return image, torch.ones((1, 1, 128, 128)), torch.ones((1, 1, 128, 128), dtype=torch.bool)


def invoke(model, image, masks, valid, negative=None):
    # Выбрать детерминированный batch, для которого включена пара, не меняя общий RNG.
    for index in range(1, 30):
        result = context_consistency_loss(torch, model, image, masks, valid, negative,
            seed=42, epoch=1, batch_index=index)
        if model.seen:
            return result
    raise AssertionError("Не найден выбранный пакет")


def test_pair_keeps_raw_pixels_rng_bn_dropout_and_accumulates_useful_gradient():
    model = ContextModel().train()
    model.dropout.eval()  # Восстановить и неоднородные исходные флаги.
    image, masks, valid = inputs()
    original = image.clone()
    states = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    mean, variance = model.bn.running_mean.clone(), model.bn.running_var.clone()
    loss, stats = invoke(model, image, masks, valid)
    assert stats["selected_batches"] == 1 and loss.item() > 0
    assert len(model.seen) == 2
    full, crop = model.seen
    side = crop.shape[-1]
    first = int(crop[0, 0, 0, 0].item() - 1)
    top, left = divmod(first, 128)
    assert side in (64, 96) and top % 32 == left % 32 == 0
    torch.testing.assert_close(crop, image[..., top:top+side, left:left+side])
    assert stats["supervised_pixels"] == side * side
    torch.testing.assert_close(image, original)
    torch.testing.assert_close(rng, torch.get_rng_state())
    torch.testing.assert_close(mean, model.bn.running_mean)
    torch.testing.assert_close(variance, model.bn.running_var)
    assert model.bn.num_batches_tracked == 0
    assert [module.training for module in model.modules()] == states
    loss.backward()
    assert model.bias.grad.item() < 0  # Шаг спуска повышает уверенность ученика в воде.


@pytest.mark.parametrize("reason", ["wrong_teacher", "no_valid", "ambiguous_negative"])
def test_unsafe_teacher_nodata_and_ambiguous_targets_cannot_supervise(reason):
    model = ContextModel(correct=reason != "wrong_teacher").train()
    image, masks, valid = inputs()
    negative = torch.ones((1, 2, 128, 128), dtype=torch.bool) if reason == "ambiguous_negative" else None
    if reason == "no_valid":
        valid[:] = False
    for index in range(1, 10):
        loss, stats = context_consistency_loss(torch, model, image, masks, valid, negative,
            seed=42, epoch=1, batch_index=index)
        assert loss is None and stats["selected_batches"] == stats["supervised_pixels"] == 0
    assert model.bias.grad is None and model.bn.num_batches_tracked == 0


def test_modes_restored_on_failure():
    model = ContextModel(fail=True).train()
    image, masks, valid = inputs()
    with pytest.raises(RuntimeError, match="второго прохода"):
        invoke(model, image, masks, valid)
    assert all(module.training for module in model.modules())


def test_nonfinite_teacher_is_reported_instead_of_silently_disabling_penalty():
    from mlsystem2.train.contracts import TrainError
    model = ContextModel().train()
    with torch.no_grad():
        model.bias.fill_(float("nan"))
    image, masks, valid = inputs()
    with pytest.raises(TrainError, match="Неконечный прогноз учителя"):
        invoke(model, image, masks, valid)
    assert all(module.training for module in model.modules())


def test_multiclass_targets_exclude_ambiguous_pixels_and_keep_class_correspondence():
    class Multiclass(ContextModel):
        def forward(self, image, **kwargs):
            logits = super().forward(image, **kwargs)
            # Добавление отсутствующего третьего класса не меняет соответствие выбранного класса 1.
            return torch.cat((logits, torch.full_like(logits[:, :1], -30)), dim=1)
    model = Multiclass().train()
    image, masks, valid = inputs()
    negative = torch.zeros((1, 2, 128, 128), dtype=torch.bool)
    negative[..., ::2, :] = True
    loss, stats = invoke(model, image, masks[:, 0].long(), valid[:, 0], negative)
    side = model.seen[-1].shape[-1]
    assert stats["supervised_pixels"] == side * side // 2
    loss.backward()
    assert torch.isfinite(model.bias.grad)


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_registered_profile_preserves_base_defaults_and_optional_slices(architecture, size):
    before = sanitize_template_config({"train.pipeline_variant": "next_gen4_augmentation_v4", "tile_preparation.tile_size": size}, architecture=architecture)
    after = sanitize_template_config({"train.pipeline_variant": VARIANT, "tile_preparation.tile_size": size}, architecture=architecture)
    assert after == {**before, "train.pipeline_variant": VARIANT}
    sliced = sanitize_template_config({**after, "train.diagnostic_checkpoint_interval_sec": 900}, architecture=architecture)
    assert sliced == {**after, "train.diagnostic_checkpoint_interval_sec": 900}
    template = next(item for item in initial_templates() if item["architecture"] == architecture)
    assert VARIANT in template["config_schema"]["fields"][0]["options"]
    assert template["config_schema"]["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "KL" in template["config_schema"]["pipeline_descriptions"][VARIANT]


def model_spec(variant=VARIANT):
    from mlsystem2.models.contracts import ModelSpec
    return ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1, parameters={
        "pipeline_variant": variant, "context_consistency": POLICY,
        **({"decoder_normalization": {"version": "per_pixel_channel_layer_norm_v1", "epsilon": 1e-5, "affine": True}}
           if variant == "next_gen4_consistency_v2" else {}),
        "augmentation_policy": "validity_guard_v4_context_crop_v1",
        "context_crop": {"area_scale": [0.25, 1.0], "ratio": [1.0, 1.0], "probability": 0.5,
            "output_size": 512, "source": "within_retained_train_window"},
        "preprocessing": {"mode": "global_minmax", "epsilon": 1e-6, "nodata": 0.0,
            "mask_output_nodata": True, "source": "retained_train_windows", "low": [0.] * 4,
            "high": [255.] * 4, "valid_pixel_count": 4096, "window_count": 4}})


def test_checkpoint_policy_is_required_and_reloads_without_changing_forward(tmp_path, monkeypatch):
    from mlsystem2.models import _factory
    from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
    from mlsystem2.models.contracts import LoadCheckpointRequest, SaveCheckpointRequest, ModelsError
    monkeypatch.setattr(_factory, "_create_smp_segformer", lambda spec, **kwargs: torch.nn.Conv2d(4, 2, 1))
    spec = model_spec()
    broken = spec.model_copy(deep=True)
    broken.parameters["context_consistency"]["weight"] = 0.5
    with pytest.raises(ModelsError, match="политику согласованности"):
        create_model(broken)
    model = create_model(spec)
    path = tmp_path / "best.pt"
    save_checkpoint(SaveCheckpointRequest(model=model, checkpoint_uri=str(path), metadata={}))
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))
    image, _, _ = inputs()
    torch.testing.assert_close(model.model(image), loaded.model.model(image))
    assert loaded.model.spec.parameters["context_consistency"] == POLICY


def test_continuation_rejects_different_consistency_policy(monkeypatch):
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError
    spec = model_spec()
    loaded = SimpleNamespace(spec=spec.model_copy(deep=True))
    loaded.spec.parameters["context_consistency"]["weight"] = 0.5
    monkeypatch.setattr(_runner, "_model_spec", lambda *args: spec)
    deps = SimpleNamespace(load_checkpoint=lambda request: SimpleNamespace(model=loaded))
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT, initial_checkpoint_uri="best.pt", device="cpu"))
    with pytest.raises(TrainPipelineError, match="согласованности"):
        _runner._load_or_create_model(settings, deps)


@pytest.mark.parametrize("variant", [VARIANT, "next_gen4_consistency_v2"])
def test_periodic_current_snapshots_preserve_one_training_loop_and_best_selection(tmp_path, monkeypatch, variant):
    from mlsystem2.models import _factory
    from mlsystem2.models.api import create_model
    from mlsystem2.tile_preparation._adaptive_sampling import AdaptiveBackgroundSampler
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
    from mlsystem2.train import _trainer
    from mlsystem2.train.contracts import TrainRequest, TrainConfig
    torch.set_num_threads(2)
    monkeypatch.setattr(_factory, "_create_smp_segformer", lambda spec, **kwargs: torch.nn.Conv2d(4, 2, 1))
    model = create_model(model_spec(variant))
    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, index):
            index = index[0] if isinstance(index, tuple) else index
            return torch.full((4, 64, 64), 40.), torch.full((1, 64, 64), float(index < 2)), {
                "sample_index": index, "valid_pixels": torch.ones((64, 64), dtype=torch.bool)}
    sampler = AdaptiveBackgroundSampler([{"positive": index < 2, "valid_pixels": 4096,
        "foreground_pixels": 4096 if index < 2 else 0, "scene_id": "сцена", "group": "сцена"} for index in range(4)], seed=42)
    loader = torch.utils.data.DataLoader(Samples(), batch_size=1, sampler=sampler, collate_fn=_collate_tile_batch)
    loader.adaptive_sampling_controller = sampler
    ticks = iter(index / 4 for index in range(10000))
    monkeypatch.setattr(_trainer, "perf_counter", lambda: next(ticks))
    config = TrainConfig(pipeline_variant=variant, class_weights=[1., 1.], epochs=2, batch_size=1,
        device="cpu", learning_rate=1e-4, weight_decay=.01, loss="cross_entropy_tversky", threshold_mode="fixed",
        early_stopping_patience=10, diagnostic_checkpoint_interval_sec=1)
    result = _trainer.train_model(TrainRequest(model=model, train_loader=loader, val_loader=loader,
        dense_train_loader=loader, config=config, checkpoint_dir=str(tmp_path), sample_size=64))
    index = json.loads((tmp_path / "diagnostic_checkpoints.json").read_text())
    assert result.epochs_total == 2 and len(index["checkpoints"]) == 2
    assert result.diagnostics["checkpoint_selection"]["metric"] == "val_loss"
    assert result.diagnostics["dense_train_windows"]["evaluation_performed"]
    assert all(item.train_context_consistency["attempted_batches"] > 0 for item in result.history)
    for row in index["checkpoints"]:
        assert row["actual_elapsed_sec"] >= row["scheduled_elapsed_sec"]
        value = torch.load(row["uri"], map_location="cpu", weights_only=True)
        assert value["metadata"]["diagnostic_checkpoint"]["kind"] == "current_not_best"
        assert value["metadata"]["epoch"] == row["epoch"]
    assert len([item for item in result.artifacts if item.label == "diagnostic"]) == 2


def test_mlflow_keeps_current_snapshots_and_index_separate_from_best(tmp_path, monkeypatch):
    from mlsystem2.mlflow_adapter import _client
    best, snapshot = tmp_path / "best.pt", tmp_path / "diagnostic_000900s_epoch_0004.pt"
    best.write_bytes(b"best")
    snapshot.write_bytes(b"current")
    logged_files, logged_dicts = [], {}
    monkeypatch.setattr(_client, "_ensure_run_active", lambda run: SimpleNamespace())
    monkeypatch.setattr(_client, "_log_artifact", lambda path, folder: logged_files.append((str(path), folder)))
    monkeypatch.setattr(_client, "_log_dict", lambda value, path: logged_dicts.update({path: value}))
    index = {"interval_sec": 900, "checkpoints": [{"uri": str(snapshot), "kind": "current_not_best"}]}
    result = SimpleNamespace(history=[], task="binary", class_schema=[], best_threshold=.5,
        diagnostics={"diagnostic_checkpoints": index}, best_checkpoint_path=str(best), final_checkpoint_path=None,
        artifacts=[SimpleNamespace(uri=str(snapshot), label="diagnostic")])
    _client.log_training_artifacts(SimpleNamespace(active=True), result)
    assert logged_files == [(str(best), "checkpoints"), (str(snapshot), "checkpoints")]
    assert logged_dicts["reports/diagnostic_checkpoints.json"] == index
    assert set(logged_dicts["reports/checkpoint_hashes.json"]) == {best.name, snapshot.name}
