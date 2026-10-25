"""Независимые цели полного query и источника при отсоединённых K/V."""

from types import SimpleNamespace

import pytest

from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelsError, SaveCheckpointRequest
from mlsystem2.train._consistency import full_query_context_loss
from mlsystem2.train.contracts import TrainError
from test_next_gen4_consistency_v5 import inputs, spec as previous_spec

torch = pytest.importorskip("torch")

VARIANT = "next_gen4_consistency_v6"
POLICY = {
    "version": "full_query_detached_crop_kv_with_reference_gt_v1",
    "weight": .25, "probability": .5, "crop_side_fractions": [.5, .75],
    "source": "within_retained_train_window", "evaluation_mode": True,
    "class_balance": True, "reference": "detached_same_model_crop_kv_v1",
    "supervision": "query_and_reference_ground_truth_cross_entropy_v1",
    "context_reduction": "mean_of_query_and_reference",
}


def spec():
    result = previous_spec()
    result.parameters.update(pipeline_variant=VARIANT, context_consistency=dict(POLICY))
    return result


class DualContextModel(torch.nn.Module):
    def __init__(self, channels=2, fail=False, bad_protocol=False):
        super().__init__()
        self.query = torch.nn.Parameter(torch.linspace(-1., 1., channels).view(1, channels, 1, 1).expand(1, channels, 128, 128).clone())
        self.reference = torch.nn.Parameter(torch.linspace(.7, -.3, channels).view(1, channels, 1, 1).expand(1, channels, 128, 128).clone())
        self.bn = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout(.9)
        self.seen = []
        self.fail, self.bad_protocol = fail, bad_protocol

    def forward(self, image, *, return_two_class_logits=False, context_reference=None):
        self.dropout(self.bn(image))
        if context_reference is None:
            return self.query
        self.seen.append((image.detach().clone(), context_reference.detach().clone(), self.training))
        if self.fail:
            raise RuntimeError("Проверочная ошибка двухконтекстного прохода")
        side = context_reference.shape[-1]
        return self.query if self.bad_protocol else (self.query, self.reference[..., :side, :side])


def invoke(model, image, target, valid, negative=None):
    for index in range(1, 30):
        loss, stats = full_query_context_loss(
            torch, model, image, target, valid, negative,
            seed=42, epoch=1, batch_index=index, supervise_reference=True,
        )
        if model.seen or not valid.any() or (negative is not None and negative.all()):
            return loss, stats, index
    raise AssertionError("Не найден выбранный пакет")


def crop_geometry(model):
    _, crop, _ = model.seen[-1]
    top, left = divmod(int(crop[0, 0, 0, 0] - 1), 128)
    side = crop.shape[-1]
    return top, left, side


def balanced_ce(logits, target, valid):
    safe_target = target.masked_fill(~valid, 0)
    ce = torch.nn.functional.cross_entropy(logits, safe_target, reduction="none")
    return torch.stack([ce[valid & (target == label)].mean() for label in range(logits.shape[1])
                        if (valid & (target == label)).any()]).mean()


@pytest.mark.parametrize("channels", [2, 3])
def test_each_context_uses_its_own_target_balances_classes_and_weight_once(channels):
    image, target, valid = inputs()
    target.zero_()
    target[..., 20:100, 20:100] = 1
    if channels == 3:
        target[..., 62:67, 62:67] = 2
    model = DualContextModel(channels).train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    rng, original = torch.get_rng_state().clone(), image.clone()
    loss, stats, _ = invoke(model, image, target, valid)
    top, left, side = crop_geometry(model)
    region = (..., slice(top, top + side), slice(left, left + side))
    query_ce = balanced_ce(model.query, target[:, 0].long(), valid[:, 0])
    reference_ce = balanced_ce(model.reference[..., :side, :side], target[:, 0][region].long(), valid[:, 0][region])
    torch.testing.assert_close(loss, .25 * .5 * (query_ce + reference_ce))
    assert stats["query_contribution_sum"] == pytest.approx(.5 * float(query_ce.detach()))
    assert stats["reference_contribution_sum"] == pytest.approx(.5 * float(reference_ce.detach()))
    assert stats["context_loss_sum"] == pytest.approx(stats["query_contribution_sum"] + stats["reference_contribution_sum"])
    assert stats["query_supervised_pixels"] == 128 ** 2
    assert stats["reference_supervised_pixels"] == side ** 2
    assert stats["supervised_pixels"] == 128 ** 2 + side ** 2
    assert stats["half_side_selected_batches"] + stats["three_quarter_side_selected_batches"] == 1
    assert [module.training for module in model.modules()] == states
    assert model.bn.num_batches_tracked == 0 and not model.seen[-1][2]
    torch.testing.assert_close(torch.get_rng_state(), rng)
    torch.testing.assert_close(image, original)
    loss.backward()
    assert model.query.grad is not None and model.reference.grad is not None


def test_invalid_and_ambiguous_pixels_have_zero_gradient_in_both_contexts():
    image, target, valid = inputs()
    valid[..., :12, :] = False
    negative = torch.zeros(1, 2, 128, 128, dtype=torch.bool)
    negative[..., 44:76, 44:76] = True
    target[~valid | negative.any(1, keepdim=True)] = 99
    model = DualContextModel()
    loss, stats, _ = invoke(model, image, target, valid, negative)
    top, left, side = crop_geometry(model)
    allowed = valid[:, 0] & ~negative.any(1)
    source_allowed = allowed[..., top:top + side, left:left + side]
    loss.backward()
    assert torch.count_nonzero(model.query.grad.permute(0, 2, 3, 1)[~allowed]) == 0
    assert torch.count_nonzero(model.reference.grad[..., :side, :side].permute(0, 2, 3, 1)[~source_allowed]) == 0
    assert (model.query.grad[:, 1][allowed] < 0).all()
    assert (model.reference.grad[:, 1, :side, :side][source_allowed] < 0).all()
    assert stats["supervised_pixels"] == int(allowed.sum() + source_allowed.sum())


@pytest.mark.parametrize("reason", ["nodata", "ambiguous"])
def test_fully_excluded_batch_is_not_executed(reason):
    image, target, valid = inputs()
    negative = torch.ones(1, 2, 128, 128, dtype=torch.bool) if reason == "ambiguous" else None
    if reason == "nodata":
        valid.zero_()
    target.fill_(99)
    model = DualContextModel()
    loss, stats, _ = invoke(model, image, target, valid, negative)
    assert loss is None and not model.seen
    assert stats["query_supervised_pixels"] == stats["reference_supervised_pixels"] == stats["selected_batches"] == 0


def test_nonempty_query_with_empty_reference_does_not_supervise_either_context():
    image, target, valid = inputs()
    probe = DualContextModel()
    _, _, index = invoke(probe, image, target, valid)
    top, left, side = crop_geometry(probe)
    valid[..., top:top + side, left:left + side] = False
    assert valid.any()
    model = DualContextModel()
    loss, stats = full_query_context_loss(torch, model, image, target, valid, None,
        seed=42, epoch=1, batch_index=index, supervise_reference=True)
    assert loss is None and stats["selected_batches"] == 0 and not model.seen


@pytest.mark.parametrize("failure", ["runtime", "protocol", "target"])
def test_modes_are_restored_for_all_failures(failure):
    image, target, valid = inputs()
    model = DualContextModel(fail=failure == "runtime", bad_protocol=failure == "protocol").train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    if failure == "target":
        target.fill_(99)
    with pytest.raises(RuntimeError if failure == "runtime" else TrainError):
        invoke(model, image, target, valid)
    assert [module.training for module in model.modules()] == states
    assert model.bn.num_batches_tracked == 0


@pytest.mark.parametrize("field", list(POLICY))
def test_model_rejects_every_missing_policy_field(field):
    broken = spec()
    del broken.parameters["context_consistency"][field]
    with pytest.raises(ModelsError, match="политику"):
        create_model(broken)


@pytest.mark.parametrize("field,value", [("weight", .5), ("probability", 1.),
    ("version", "full_query_detached_crop_kv_gt_v1"), ("teacher_confidence", .9),
    ("reference", "wrong"), ("supervision", "wrong"), ("context_reduction", "sum")])
def test_model_rejects_policy_substitution_or_extra_fields(field, value):
    broken = spec()
    broken.parameters["context_consistency"][field] = value
    with pytest.raises(ModelsError, match="политику"):
        create_model(broken)


def test_real_smp_query_gradient_stays_detached_and_source_ce_trains_kv():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    previous = create_model(previous_spec()).model.eval()
    torch.manual_seed(123)
    model = create_model(spec()).model.eval()
    image = (torch.rand(1, 4, 128, 128) * 200 + 1).requires_grad_()
    reference = image[..., 32:96, 32:96].detach().clone().requires_grad_()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected_query = previous(image, return_two_class_logits=True, context_reference=reference)
        expected_source = model(reference, return_two_class_logits=True)
        expected_ordinary = model(image)
    query, source = model(image, return_two_class_logits=True, context_reference=reference)
    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
    attention = [module for module in model.modules() if type(module).__name__ == "Attention"]
    assert attention and all(not module.kv._forward_hooks for module in attention)
    torch.nn.functional.cross_entropy(query, torch.ones(1, 128, 128, dtype=torch.long)).backward()
    assert reference.grad is None and image.grad is not None
    assert all(module.kv.weight.grad is None for module in attention)
    assert all(module.q.weight.grad is not None for module in attention)
    model.zero_grad(set_to_none=True)
    image.grad = None
    torch.nn.functional.cross_entropy(source, torch.ones(1, 64, 64, dtype=torch.long)).backward()
    assert image.grad is None and reference.grad is not None and torch.count_nonzero(reference.grad)
    assert all(module.kv.weight.grad is not None and torch.count_nonzero(module.kv.weight.grad) for module in attention)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(image), expected_ordinary, rtol=0, atol=0)


@pytest.mark.parametrize("failure_pass", ["reference", "query"])
def test_real_smp_removes_hooks_when_either_pass_raises(failure_pass):
    model = create_model(spec()).model.eval()
    attention = [module for module in model.modules() if type(module).__name__ == "Attention"]
    calls = []
    def failure(module, inputs):
        calls.append(1)
        if len(calls) == (1 if failure_pass == "reference" else 2):
            raise RuntimeError("Проверочная ошибка K/V")
    handle = attention[-1].kv.register_forward_pre_hook(failure)
    try:
        with pytest.raises(RuntimeError, match="K/V"):
            model(torch.ones(1, 4, 64, 64), return_two_class_logits=True, context_reference=torch.ones(1, 4, 32, 32))
        assert all(not module.kv._forward_hooks for module in attention)
    finally:
        handle.remove()


def test_v6_reference_requires_two_logits_eval_and_compatible_shape():
    model = create_model(spec()).model
    image, reference = torch.ones(1, 4, 64, 64), torch.ones(1, 4, 32, 32)
    with pytest.raises(ModelsError, match="logits"):
        model(image, context_reference=reference)
    with pytest.raises(ModelsError, match="eval"):
        model(image, return_two_class_logits=True, context_reference=reference)
    model.eval()
    for shape in ((1, 3, 32, 32), (1, 4, 32, 64), (1, 4, 96, 96), (1, 4, 31, 31)):
        with pytest.raises(ModelsError, match="совместимую"):
            model(image, return_two_class_logits=True, context_reference=torch.ones(shape))


def test_checkpoint_reload_is_offline_and_ordinary_graph_matches_v5(tmp_path):
    torch.manual_seed(123)
    previous = create_model(previous_spec()).model.eval()
    torch.manual_seed(123)
    current = create_model(spec())
    current.model.eval()
    assert current.model.state_dict().keys() == previous.state_dict().keys()
    path = tmp_path / "best.pt"
    save_checkpoint(SaveCheckpointRequest(model=current, checkpoint_uri=str(path), metadata={}))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu")).model
    restored.model.eval()
    assert restored.spec.parameters["context_consistency"] == POLICY
    with torch.no_grad():
        image = torch.full((1, 4, 64, 64), 40.)
        torch.testing.assert_close(previous(image), current.model(image), rtol=0, atol=0)
        torch.testing.assert_close(current.model(image), restored.model(image), rtol=0, atol=0)
        result = restored.model(image, return_two_class_logits=True, context_reference=image[..., :32, :32])
        assert isinstance(result, tuple) and len(result) == 2


def test_pipeline_spec_changes_only_variant_and_context_policy():
    from mlsystem2.train_pipeline import _runner
    dataset = SimpleNamespace(band_histogram={"source": "retained_train_windows",
        "counts": [[1] + [0] * 254 + [1] for _ in range(4)], "valid_pixel_count": 2, "window_count": 1})
    def configured(variant):
        settings = SimpleNamespace(tile_preparation=SimpleNamespace(tile_size=512), train=SimpleNamespace(
            pipeline_variant=variant, input_channels=4, output_channels=1, pretrained=False, model_name="smp_segformer_b0"))
        return _runner._model_spec(settings, SimpleNamespace(dataset=dataset))
    before, after = configured("next_gen4_consistency_v5"), configured(VARIANT)
    assert after.parameters == {**before.parameters, "pipeline_variant": VARIANT, "context_consistency": POLICY}


@pytest.mark.parametrize("field", ["supervision", "context_reduction"])
def test_continuation_rejects_different_context_policy(monkeypatch, field):
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError
    expected = spec()
    loaded = SimpleNamespace(spec=expected.model_copy(deep=True))
    loaded.spec.parameters["context_consistency"][field] = "wrong"
    monkeypatch.setattr(_runner, "_model_spec", lambda *args: expected)
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT, initial_checkpoint_uri="best.pt", device="cpu"))
    with pytest.raises(TrainPipelineError, match="согласованности"):
        _runner._load_or_create_model(settings, SimpleNamespace(load_checkpoint=lambda request: SimpleNamespace(model=loaded)))


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_registered_profile_preserves_v5_defaults_and_continuous_stop_limits(architecture, size):
    from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config
    before = sanitize_template_config({"train.pipeline_variant": "next_gen4_consistency_v5", "tile_preparation.tile_size": size,
        "train.epochs": 60, "train.early_stopping_patience": 20, "train.max_training_time_sec": 7200,
        "train.diagnostic_checkpoint_interval_sec": 900}, architecture=architecture)
    after = sanitize_template_config({**before, "train.pipeline_variant": VARIANT}, architecture=architecture)
    assert after == {**before, "train.pipeline_variant": VARIANT}
    assert after["train.max_training_time_sec"] == 7200
    schema = next(item for item in initial_templates() if item["architecture"] == architecture)["config_schema"]
    assert VARIANT in schema["fields"][0]["options"]
    assert schema["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "K/V" in schema["pipeline_descriptions"][VARIANT]


def test_train_epoch_keeps_one_step_and_reports_decomposition_without_fake_kl(monkeypatch):
    from mlsystem2.train import _trainer
    from mlsystem2.train.contracts import TrainConfig
    model = DualContextModel()
    image, target, valid = inputs()
    batches = [(image, target, {"valid_pixels": valid, "sample_indices": [0]}) for _ in range(8)]
    observed = []
    monkeypatch.setattr(_trainer, "adaptive_controller", lambda loader: SimpleNamespace(observe_train=lambda *args: observed.append(args)))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    steps, real_step = [], optimizer.step
    def step():
        steps.append(1)
        return real_step()
    monkeypatch.setattr(optimizer, "step", step)
    config = TrainConfig(pipeline_variant=VARIANT, class_weights=[1., 1.], epochs=1, batch_size=1,
        device="cpu", learning_rate=1e-4, weight_decay=.01, loss="cross_entropy_tversky", threshold_mode="fixed", early_stopping_patience=10)
    result = _trainer._train_epoch(torch, model, batches, optimizer, "cpu", config, 1)
    stats = result["context_consistency"]
    assert len(steps) == len(batches) == len(observed)
    assert stats["attempted_batches"] == 8 and stats["selected_batches"] > 0
    assert stats["query_supervised_pixels"] == stats["selected_batches"] * 128 ** 2
    assert stats["supervised_pixels"] == stats["query_supervised_pixels"] + stats["reference_supervised_pixels"]
    assert stats["mean_context_loss"] == pytest.approx(stats["mean_query_contribution"] + stats["mean_reference_contribution"])
    assert stats["mean_context_loss"] == pytest.approx(stats["context_loss_sum"] / stats["selected_batches"])
    assert stats["weight"] == .25 and stats["probability"] == .5
    assert not any("kl" in key or "fallback" in key or "teacher" in key or "forward" in key for key in stats)
