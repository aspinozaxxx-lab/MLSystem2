"""Полный query получает train-надзор при отсоединённом K/V-контексте вырезки."""

from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec, ModelsError, SaveCheckpointRequest
from mlsystem2.train._consistency import full_query_context_loss
from mlsystem2.train.contracts import TrainError

VARIANT = "next_gen4_consistency_v5"
POLICY = {"version": "full_query_detached_crop_kv_gt_v1", "weight": .25, "probability": .5,
    "crop_side_fractions": [.5, .75], "source": "within_retained_train_window",
    "evaluation_mode": True, "class_balance": True, "reference": "detached_same_model_crop_kv_v1",
    "supervision": "full_query_ground_truth_cross_entropy_v1"}


def spec():
    return ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1, pretrained=False, parameters={
        "pipeline_variant": VARIANT, "context_consistency": dict(POLICY),
        "decoder_normalization": {"version": "per_pixel_channel_layer_norm_v1", "epsilon": 1e-5, "affine": True},
        "augmentation_policy": "validity_guard_v4_context_crop_v1",
        "context_crop": {"area_scale": [.25, 1.], "ratio": [1., 1.], "probability": .5, "output_size": 512,
                         "source": "within_retained_train_window"},
        "preprocessing": {"mode": "global_minmax", "epsilon": 1e-6, "nodata": 0., "mask_output_nodata": True,
            "source": "retained_train_windows", "low": [0.] * 4, "high": [255.] * 4, "valid_pixel_count": 4096, "window_count": 4}})


class QueryModel(torch.nn.Module):
    def __init__(self, value=0., fail=False):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.bn = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout(.9)
        self.seen = []
        self.fail = fail

    def forward(self, image, *, return_two_class_logits=False, context_reference=None):
        if context_reference is not None:
            self.seen.append((image.detach().clone(), context_reference.detach().clone(), self.training))
            if self.fail:
                raise RuntimeError("Проверочная ошибка query-прохода")
        self.dropout(self.bn(image))
        foreground = self.value.expand(len(image), 1, *image.shape[-2:])
        return torch.cat((-foreground, foreground), 1)


def inputs():
    image = torch.arange(128 * 128).reshape(1, 1, 128, 128).float().expand(1, 4, 128, 128).clone() + 1
    return image, torch.ones(1, 1, 128, 128), torch.ones(1, 1, 128, 128, dtype=torch.bool)


def invoke(model, image, target, valid, negative=None):
    for index in range(1, 30):
        loss, stats = full_query_context_loss(torch, model, image, target, valid, negative,
            seed=42, epoch=1, batch_index=index)
        if model.seen or not valid.any() or (negative is not None and negative.all()):
            return loss, stats
    raise AssertionError("Не найден выбранный пакет")


def test_full_query_target_single_weight_source_is_unchanged_and_modes_restored():
    image, target, valid = inputs()
    original = image.clone()
    model = QueryModel(-1.).train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    loss, stats = invoke(model, image, target, valid)
    full, crop, training = model.seen[0]
    assert not training and len(model.seen) == 1 and full.shape == image.shape
    side = crop.shape[-1]
    assert side in (64, 96)
    first = int(crop[0, 0, 0, 0] - 1)
    top, left = divmod(first, 128)
    torch.testing.assert_close(crop, image[..., top:top + side, left:left + side])
    expected = .25 * torch.nn.functional.cross_entropy(torch.tensor([[1., -1.]]), torch.tensor([1]))
    torch.testing.assert_close(loss, expected)
    assert stats["supervised_pixels"] == 128 ** 2 and stats["reference_valid_pixels"] == side ** 2
    assert stats["half_side_selected_batches"] + stats["three_quarter_side_selected_batches"] == 1
    assert stats["context_loss_sum"] == pytest.approx(float(loss.detach()) / .25)
    assert [module.training for module in model.modules()] == states and model.bn.num_batches_tracked == 0
    torch.testing.assert_close(image, original)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    loss.backward()
    assert model.value.grad < 0


@pytest.mark.parametrize("water_pixels", [1, 128, 128 * 128 - 1])
def test_present_classes_are_equally_weighted(water_pixels):
    image, target, valid = inputs()
    target.zero_()
    target.view(-1)[:water_pixels] = 1
    model = QueryModel(1.3)
    loss, _ = invoke(model, image, target, valid)
    logits = torch.tensor([[-1.3, 1.3]])
    expected = .25 * (torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
                      + torch.nn.functional.cross_entropy(logits, torch.tensor([1]))) / 2
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("reason", ["nodata", "ambiguous"])
def test_excluded_pixels_are_never_query_supervision(reason):
    image, target, valid = inputs()
    negative = torch.ones(1, 2, 128, 128, dtype=torch.bool) if reason == "ambiguous" else None
    if reason == "nodata":
        valid.zero_()
    target.fill_(99)
    model = QueryModel().train()
    loss, stats = invoke(model, image, target, valid, negative)
    assert loss is None and stats["supervised_pixels"] == stats["selected_batches"] == 0
    assert not model.seen and model.bn.num_batches_tracked == 0


def test_invalid_allowed_class_and_runtime_error_restore_all_modes():
    image, target, valid = inputs()
    for failure in (False, True):
        model = QueryModel(fail=failure).train()
        model.dropout.eval()
        states = [module.training for module in model.modules()]
        target.fill_(99 if not failure else 1)
        with pytest.raises(RuntimeError if failure else TrainError):
            invoke(model, image, target, valid)
        assert [module.training for module in model.modules()] == states
        assert model.bn.num_batches_tracked == 0


def test_invalid_and_ambiguous_query_logits_have_zero_gradient():
    class LogitModel(QueryModel):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 2, 128, 128))
        def forward(self, image, **kwargs):
            super().forward(image, **kwargs)
            return self.logits
    model = LogitModel()
    image, target, valid = inputs()
    valid.zero_()
    valid[..., 32:96, 32:96] = True
    target[~valid] = 99
    negative = torch.zeros(1, 2, 128, 128, dtype=torch.bool)
    negative[..., 48:64, 48:64] = True
    target[..., 48:64, 48:64] = 99
    loss, stats = invoke(model, image, target, valid, negative)
    loss.backward()
    allowed = valid[:, 0] & ~negative.any(1)
    assert stats["supervised_pixels"] == int(allowed.sum())
    assert torch.count_nonzero(model.logits.grad.permute(0, 2, 3, 1)[~allowed]) == 0
    assert (model.logits.grad[:, 1][allowed] < 0).all()


@pytest.mark.parametrize("field,value", [("reference", "wrong"), ("supervision", "wrong"), ("weight", .5), ("teacher_confidence", .9)])
def test_model_requires_exact_new_policy_before_execution(field, value):
    broken = spec()
    broken.parameters["context_consistency"][field] = value
    with pytest.raises(ModelsError, match="политику"):
        create_model(broken)


def test_real_smp_source_has_no_gradient_query_has_gradient_and_default_is_restored():
    torch.set_num_threads(2)
    model = create_model(spec()).model.eval()
    image = (torch.rand(1, 4, 128, 128) * 200 + 1).requires_grad_()
    reference = image[..., 32:96, 32:96].detach().clone().requires_grad_()
    image_before, reference_before = image.detach().clone(), reference.detach().clone()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        before = model(image, return_two_class_logits=True)
    logits = model(image, return_two_class_logits=True, context_reference=reference)
    torch.nn.functional.cross_entropy(logits, torch.ones(1, 128, 128, dtype=torch.long)).backward()
    assert reference.grad is None and image.grad is not None
    attention = [module for module in model.modules() if type(module).__name__ == "Attention"]
    assert attention and all(module.kv.weight.grad is None for module in attention)
    assert any(module.q.weight.grad is not None and torch.count_nonzero(module.q.weight.grad) for module in attention)
    assert all(not module.kv._forward_hooks for module in attention)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(image, return_two_class_logits=True), before, rtol=0, atol=0)
        torch.testing.assert_close(model(image, return_two_class_logits=True, context_reference=image), before, rtol=0, atol=0)
    torch.testing.assert_close(image, image_before)
    torch.testing.assert_close(reference, reference_before)


@pytest.mark.parametrize("fail_pass", ["reference", "query"])
def test_real_smp_hooks_are_removed_when_either_pass_fails(fail_pass):
    model = create_model(spec()).model.eval()
    attention = [module for module in model.modules() if type(module).__name__ == "Attention"]
    calls = []
    def failure(module, inputs):
        calls.append(1)
        if len(calls) == (1 if fail_pass == "reference" else 2):
            raise RuntimeError("Проверочная ошибка K/V-проекции")
    handle = attention[-1].kv.register_forward_pre_hook(failure)
    try:
        with pytest.raises(RuntimeError, match="K/V-проекции"):
            model(torch.ones(1, 4, 64, 64), context_reference=torch.ones(1, 4, 32, 32))
        assert all(not module.kv._forward_hooks for module in attention)
    finally:
        handle.remove()


def test_legacy_model_rejects_reference_and_v5_requires_eval_and_compatible_shape():
    model = create_model(spec()).model
    with pytest.raises(ModelsError, match="eval"):
        model(torch.ones(1, 4, 64, 64), context_reference=torch.ones(1, 4, 32, 32))
    model.eval()
    for shape in ((1, 3, 32, 32), (1, 4, 32, 64), (1, 4, 96, 96), (1, 4, 31, 31)):
        with pytest.raises(ModelsError, match="совместимую"):
            model(torch.ones(1, 4, 64, 64), context_reference=torch.ones(shape))
    model.allow_context_reference = False
    with pytest.raises(ModelsError, match="только для профиля"):
        model(torch.ones(1, 4, 64, 64), context_reference=torch.ones(1, 4, 32, 32))


def test_checkpoint_reload_is_offline_and_default_graph_matches_v4(tmp_path):
    from test_next_gen4_consistency_v4 import spec as previous_spec
    torch.manual_seed(123)
    previous = create_model(previous_spec()).model.eval()
    torch.manual_seed(123)
    current = create_model(spec())
    current.model.eval()
    assert current.model.state_dict().keys() == previous.state_dict().keys()
    for key, value in current.model.state_dict().items():
        torch.testing.assert_close(value, previous.state_dict()[key], rtol=0, atol=0)
    path = tmp_path / "best.pt"
    save_checkpoint(SaveCheckpointRequest(model=current, checkpoint_uri=str(path), metadata={}))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu")).model
    restored.model.eval()
    assert restored.spec.parameters["context_consistency"] == POLICY
    with torch.no_grad():
        image = torch.full((1, 4, 64, 64), 40.)
        torch.testing.assert_close(previous(image), current.model(image), rtol=0, atol=0)
        torch.testing.assert_close(current.model(image), restored.model(image), rtol=0, atol=0)


def test_pipeline_spec_changes_only_variant_and_context_policy():
    from mlsystem2.train_pipeline import _runner
    dataset = SimpleNamespace(band_histogram={"source": "retained_train_windows",
        "counts": [[1] + [0] * 254 + [1] for _ in range(4)], "valid_pixel_count": 2, "window_count": 1})
    def configured(variant):
        settings = SimpleNamespace(tile_preparation=SimpleNamespace(tile_size=512), train=SimpleNamespace(
            pipeline_variant=variant, input_channels=4, output_channels=1, pretrained=False, model_name="smp_segformer_b0"))
        return _runner._model_spec(settings, SimpleNamespace(dataset=dataset))
    before, after = configured("next_gen4_consistency_v4"), configured(VARIANT)
    assert after.parameters == {**before.parameters, "pipeline_variant": VARIANT, "context_consistency": POLICY}


def test_continuation_rejects_a_different_reference_policy(monkeypatch):
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError
    expected = spec()
    loaded = SimpleNamespace(spec=expected.model_copy(deep=True))
    loaded.spec.parameters["context_consistency"]["reference"] = "wrong"
    monkeypatch.setattr(_runner, "_model_spec", lambda *args: expected)
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT, initial_checkpoint_uri="best.pt", device="cpu"))
    with pytest.raises(TrainPipelineError, match="согласованности"):
        _runner._load_or_create_model(settings, SimpleNamespace(load_checkpoint=lambda request: SimpleNamespace(model=loaded)))


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_registered_profile_preserves_v4_defaults(architecture, size):
    from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config
    before = sanitize_template_config({"train.pipeline_variant": "next_gen4_consistency_v4", "tile_preparation.tile_size": size}, architecture=architecture)
    after = sanitize_template_config({**before, "train.pipeline_variant": VARIANT}, architecture=architecture)
    assert after == {**before, "train.pipeline_variant": VARIANT}
    schema = next(item for item in initial_templates() if item["architecture"] == architecture)["config_schema"]
    assert VARIANT in schema["fields"][0]["options"]
    assert schema["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "K/V" in schema["pipeline_descriptions"][VARIANT]


def test_train_epoch_keeps_one_step_per_batch_and_reports_only_real_context_stats(monkeypatch):
    from mlsystem2.train import _trainer
    from mlsystem2.train.contracts import TrainConfig
    model = QueryModel(.2)
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
    assert stats["supervised_pixels"] == stats["selected_batches"] * 128 ** 2
    assert stats["mean_context_loss"] == pytest.approx(stats["context_loss_sum"] / stats["selected_batches"])
    assert stats["weight"] == .25 and stats["probability"] == .5
    assert not any("kl" in key or "fallback" in key or "teacher" in key or "forward" in key for key in stats)
