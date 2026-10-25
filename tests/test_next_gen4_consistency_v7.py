"""Общие reference-геометрии: пиксели, цели, градиенты и совместимость профилей."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelsError, SaveCheckpointRequest
from mlsystem2.train._consistency import full_query_context_loss
from mlsystem2.train.contracts import TrainError
from test_next_gen4_consistency_v6 import POLICY as V6_POLICY, spec as v6_spec

torch = pytest.importorskip("torch")
GEOMETRIES = ("ordinary_crop", "padded_square", "padded_horizontal", "padded_vertical")
VARIANT = "next_gen4_consistency_v7"
POLICY = {**V6_POLICY, "version": "full_query_detached_geometry_kv_with_reference_gt_v1",
          "reference_geometries": list(GEOMETRIES), "geometry_sampling": "uniform",
          "placement": "independent_grid32_within_input"}


def spec():
    configured = v6_spec()
    configured.parameters.update(pipeline_variant=VARIANT, context_consistency=deepcopy(POLICY))
    return configured


class GeometryModel(torch.nn.Module):
    def __init__(self, channels=2, batch=1, failure=None):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(batch, channels, 128, 128))
        self.reference = torch.nn.Parameter(torch.randn(batch, channels, 128, 128))
        self.bn = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout(.9)
        self.failure = failure
        self.seen = []

    def forward(self, image, *, return_two_class_logits=False, context_reference=None):
        self.dropout(self.bn(image))
        if context_reference is None:
            return self.query
        self.seen.append((image.detach().clone(), context_reference.detach().clone(), self.training))
        if self.failure == "runtime":
            raise RuntimeError("Проверочная ошибка геометрии")
        if self.failure == "protocol":
            return self.query
        height, width = context_reference.shape[-2:]
        return self.query, self.reference[..., :height, :width]


def inputs(channels=2, batch=1):
    indices = torch.arange(128 ** 2).reshape(1, 1, 128, 128)
    image = (indices + 1).float().expand(batch, 4, 128, 128).clone()
    if batch == 2:
        image[1] += 128 ** 2
    target = (indices % channels).expand(batch, 1, 128, 128).clone().float()
    valid = torch.ones_like(target, dtype=torch.bool)
    return image, target, valid


def invoke(model, image, target, valid, negative=None, *, index=1):
    return full_query_context_loss(torch, model, image, target, valid, negative,
        seed=42, epoch=1, batch_index=index, supervise_reference=True, geometry_views=True)


def selected(model, image, target, valid, negative=None, geometry=None, half=None):
    for index in range(1, 200):
        model.seen.clear()
        loss, stats = invoke(model, image, target, valid, negative, index=index)
        if loss is not None and (geometry is None or stats[geometry + "_selected_batches"]):
            if half is None or bool(stats["half_side_selected_batches"]) == half:
                return loss, stats
    raise AssertionError("Не найдена заданная проверочная геометрия")


def mapped_targets(reference, target, allowed):
    present = reference[:, 0] != 0
    indices = (reference[:, 0].long() - 1).clamp_min(0) % (128 ** 2)
    labels = target[:, 0].flatten(1).gather(1, indices.flatten(1)).reshape_as(indices).long()
    valid = allowed.flatten(1).gather(1, indices.flatten(1)).reshape_as(indices) & present
    return labels, valid, present


def balanced(logits, labels, allowed):
    ce = torch.nn.functional.cross_entropy(logits, labels.masked_fill(~allowed, -100),
        reduction="none", ignore_index=-100)
    return torch.stack([ce[allowed & (labels == index)].mean() for index in range(logits.shape[1])
                        if (allowed & (labels == index)).any()]).mean()


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("half", [True, False])
@pytest.mark.parametrize("channels,batch", [(2, 1), (3, 2)])
def test_every_geometry_maps_own_target_and_valid_without_resizing(geometry, half, channels, batch):
    image, target, valid = inputs(channels, batch)
    valid[..., :12, :] = False
    negative = torch.zeros(batch, channels, 128, 128, dtype=torch.bool)
    negative[..., 44:76, 44:76] = True
    allowed = valid[:, 0] & ~negative.any(1)
    target[~allowed.unsqueeze(1)] = 99
    original = [value.clone() for value in (image, target, valid, negative)]
    model = GeometryModel(channels, batch).train()
    model.dropout.eval()
    states, rng = [module.training for module in model.modules()], torch.get_rng_state().clone()
    loss, stats = selected(model, image, target, valid, negative, geometry=geometry, half=half)
    query_image, reference, training = model.seen[-1]
    assert not training and [module.training for module in model.modules()] == states
    assert model.bn.num_batches_tracked == 0
    torch.testing.assert_close(query_image, image, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for before, after in zip(original, (image, target, valid, negative)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    labels, reference_valid, present = mapped_targets(reference, target, allowed)
    side = 64 if half else 96
    crop_height = 128 if geometry == "padded_vertical" else side
    crop_width = 128 if geometry == "padded_horizontal" else side
    assert reference.shape[-2:] == ((side, side) if geometry == "ordinary_crop" else (128, 128))
    assert present.sum() == batch * crop_height * crop_width
    ys, xs = present[0].nonzero(as_tuple=True)
    assert int(ys.min()) % 32 == int(xs.min()) % 32 == 0
    assert int(ys.max() - ys.min() + 1) == crop_height
    assert int(xs.max() - xs.min() + 1) == crop_width
    origin = int(reference[0, 0][present[0]][0] - 1)
    top, left = divmod(origin, 128)
    assert top % 32 == left % 32 == 0
    torch.testing.assert_close(reference[..., int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1],
        image[..., top:top + crop_height, left:left + crop_width], rtol=0, atol=0)
    h, w = reference.shape[-2:]
    query_ce = balanced(model.query, target[:, 0].long(), allowed)
    reference_ce = balanced(model.reference[..., :h, :w], labels, reference_valid)
    torch.testing.assert_close(loss, .125 * (query_ce + reference_ce))
    assert stats["query_supervised_pixels"] == int(allowed.sum())
    assert stats["reference_supervised_pixels"] == int(reference_valid.sum())
    assert stats["supervised_pixels"] == int(allowed.sum() + reference_valid.sum())
    assert sum(stats[name + "_selected_batches"] for name in GEOMETRIES) == 1
    assert stats["context_loss_sum"] == pytest.approx(stats["query_contribution_sum"] + stats["reference_contribution_sum"])
    loss.backward()
    assert not model.query.grad.masked_select((~allowed).unsqueeze(1).expand_as(model.query)).any()
    source_gradient = model.reference.grad[..., :h, :w]
    assert not source_gradient.masked_select((~reference_valid).unsqueeze(1).expand_as(source_gradient)).any()
    assert torch.count_nonzero(source_gradient[reference_valid.unsqueeze(1).expand_as(source_gradient)])


def test_placement_is_independent_from_extraction_and_reproducible():
    image, target, valid = inputs()
    model = GeometryModel().eval()
    shifted, positions = set(), {name: set() for name in GEOMETRIES[1:]}
    for index in range(1, 80):
        loss, stats = invoke(model, image, target, valid, index=index)
        if loss is None:
            continue
        geometry = next(name for name in GEOMETRIES if stats[name + "_selected_batches"])
        if geometry == "ordinary_crop":
            continue
        reference = model.seen[-1][1]
        ys, xs = (reference[0, 0] != 0).nonzero(as_tuple=True)
        source = divmod(int(reference[0, 0, ys[0], xs[0]] - 1), 128)
        destination = (int(ys.min()), int(xs.min()))
        positions[geometry].add(destination)
        if source != destination:
            shifted.add(geometry)
        again, repeated = invoke(model, image, target, valid, index=index)
        assert repeated == stats
        torch.testing.assert_close(again, loss, rtol=0, atol=0)
        torch.testing.assert_close(model.seen[-1][1], reference, rtol=0, atol=0)
    assert shifted == set(GEOMETRIES[1:])
    assert all(len(value) >= 2 for value in positions.values())


@pytest.mark.parametrize("failure", ["runtime", "protocol", "target"])
def test_modes_restored_after_failure(failure):
    image, target, valid = inputs()
    if failure == "target":
        target.fill_(99)
    model = GeometryModel(failure=failure).train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError if failure == "runtime" else TrainError):
        selected(model, image, target, valid)
    assert [module.training for module in model.modules()] == states
    assert model.bn.num_batches_tracked == 0


@pytest.mark.parametrize("empty", ["valid", "negative"])
def test_empty_targets_never_run_model(empty):
    image, target, valid = inputs()
    if empty == "valid":
        valid.zero_()
    negative = torch.ones(1, 2, 128, 128, dtype=torch.bool) if empty == "negative" else None
    model = GeometryModel()
    for index in range(1, 12):
        loss, stats = invoke(model, image, target, valid, negative, index=index)
        assert loss is None and stats["selected_batches"] == 0
        assert sum(stats[name + "_selected_batches"] for name in GEOMETRIES) == 0
    assert not model.seen


def test_geometry_requires_compatible_full_input_and_source_supervision():
    image, target, valid = inputs()
    with pytest.raises(TrainError, match="квадратный"):
        invoke(GeometryModel(), image[..., :96], target[..., :96], valid[..., :96])
    with pytest.raises(TrainError, match="надзор"):
        full_query_context_loss(torch, GeometryModel(), image, target, valid, None,
            seed=42, epoch=1, batch_index=1, geometry_views=True)


@pytest.mark.parametrize("field", list(POLICY))
def test_every_policy_field_required(field):
    configured = spec()
    del configured.parameters["context_consistency"][field]
    with pytest.raises(ModelsError, match="политику"):
        create_model(configured)


@pytest.mark.parametrize("field,value", [
    ("reference_geometries", list(reversed(GEOMETRIES))), ("geometry_sampling", "weighted"),
    ("placement", "fixed_corner"), ("weight", .5), ("supervision", "query_only"),
    ("version", V6_POLICY["version"]), ("unknown", True),
])
def test_policy_changes_rejected(field, value):
    configured = spec()
    configured.parameters["context_consistency"][field] = value
    with pytest.raises(ModelsError, match="политику"):
        create_model(configured)


def test_real_smp_full_canvas_source_kv_gradients_offline_reload_and_ordinary_abi(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    previous = create_model(v6_spec()).model.eval()
    torch.manual_seed(123)
    current = create_model(spec())
    current.model.eval()
    image = torch.full((1, 4, 128, 128), 40.)
    source = torch.zeros_like(image)
    source[..., 32:96, :] = image[..., 32:96, :]
    before = {name: value.detach().clone() for name, value in current.model.state_dict().items()}
    query, reference = current.model(image, return_two_class_logits=True, context_reference=source)
    attention = [module for module in current.model.modules() if type(module).__name__ == "Attention"]
    query_loss = torch.nn.functional.cross_entropy(query, torch.ones(1, 128, 128, dtype=torch.long))
    query_gradients = torch.autograd.grad(query_loss, [module.kv.weight for module in attention], allow_unused=True)
    assert all(value is None for value in query_gradients)
    reference_loss = torch.nn.functional.cross_entropy(reference[..., 32:96, :], torch.ones(1, 64, 128, dtype=torch.long))
    reference_loss.backward()
    assert all(module.kv.weight.grad is not None and torch.count_nonzero(module.kv.weight.grad) for module in attention)
    assert all(not module.kv._forward_hooks for module in attention)
    for name, value in current.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(current.model(image), previous(image), rtol=0, atol=0)
    path = tmp_path / "v7.pt"
    save_checkpoint(SaveCheckpointRequest(model=current, checkpoint_uri=str(path), metadata={}))
    restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu")).model
    restored.model.eval()
    assert restored.spec.parameters["context_consistency"] == POLICY
    with torch.no_grad():
        torch.testing.assert_close(restored.model(image), current.model(image), rtol=0, atol=0)


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_defaults_keep_continuous_7200_seconds(architecture, size):
    from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config
    before = sanitize_template_config({"train.pipeline_variant": "next_gen4_consistency_v6", "tile_preparation.tile_size": size,
        "train.epochs": 60, "train.early_stopping_patience": 20, "train.max_training_time_sec": 7200,
        "train.diagnostic_checkpoint_interval_sec": 900}, architecture=architecture)
    after = sanitize_template_config({**before, "train.pipeline_variant": VARIANT}, architecture=architecture)
    assert after == {**before, "train.pipeline_variant": VARIANT}
    schema = next(item for item in initial_templates() if item["architecture"] == architecture)["config_schema"]
    assert schema["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "K/V" in schema["pipeline_descriptions"][VARIANT]


def test_pipeline_changes_only_version_and_context_policy():
    from mlsystem2.train_pipeline import _runner
    dataset = SimpleNamespace(band_histogram={"source": "retained_train_windows",
        "counts": [[1] + [0] * 254 + [1] for _ in range(4)], "valid_pixel_count": 2, "window_count": 1})
    def configured(variant):
        settings = SimpleNamespace(tile_preparation=SimpleNamespace(tile_size=1024), train=SimpleNamespace(
            pipeline_variant=variant, input_channels=4, output_channels=1, pretrained=False, model_name="smp_segformer_b0"))
        return _runner._model_spec(settings, SimpleNamespace(dataset=dataset))
    before, after = configured("next_gen4_consistency_v6"), configured(VARIANT)
    assert after.parameters == {**before.parameters, "pipeline_variant": VARIANT, "context_consistency": POLICY}


@pytest.mark.parametrize("field", ["reference_geometries", "geometry_sampling", "placement"])
def test_continuation_rejects_geometry_substitution(monkeypatch, field):
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError
    expected = spec()
    loaded = SimpleNamespace(spec=expected.model_copy(deep=True))
    loaded.spec.parameters["context_consistency"][field] = "wrong"
    monkeypatch.setattr(_runner, "_model_spec", lambda *args: expected)
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT, initial_checkpoint_uri="best.pt", device="cpu"))
    with pytest.raises(TrainPipelineError, match="согласованности"):
        _runner._load_or_create_model(settings, SimpleNamespace(load_checkpoint=lambda request: SimpleNamespace(model=loaded)))


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_three_dimensional_masks_and_optional_valid(geometry):
    image, target, valid = inputs()
    loss, stats = selected(GeometryModel(), image, target[:, 0], valid[:, 0], geometry=geometry)
    assert torch.isfinite(loss) and stats["selected_batches"] == 1
    loss, stats = selected(GeometryModel(), image, target[:, 0], None, geometry=geometry)
    assert torch.isfinite(loss) and stats["query_supervised_pixels"] == 128 ** 2


def test_loss_description_does_not_call_v7_a_kl_penalty():
    from mlsystem2.mlflow_adapter._metric_descriptions import describe_metric
    meaning = describe_metric("train/loss", {"pipeline_variant": VARIANT})["meaning"]
    assert "полоса" in meaning and "пустая область исключена" in meaning
    assert "Включает дополнительный KL" not in meaning


def test_one_optimizer_step_per_batch_and_geometry_counters(monkeypatch):
    from mlsystem2.train import _trainer
    from mlsystem2.train.contracts import TrainConfig
    image, target, valid = inputs()
    model = GeometryModel()
    batches = [(image, target, {"valid_pixels": valid, "sample_indices": [0]}) for _ in range(12)]
    monkeypatch.setattr(_trainer, "adaptive_controller", lambda loader: SimpleNamespace(observe_train=lambda *args: None))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    steps, original = [], optimizer.step
    def step():
        steps.append(1)
        return original()
    monkeypatch.setattr(optimizer, "step", step)
    config = TrainConfig(pipeline_variant=VARIANT, class_weights=[1., 1.], epochs=1, batch_size=1,
        device="cpu", learning_rate=1e-4, weight_decay=.01, loss="cross_entropy_tversky",
        threshold_mode="fixed", early_stopping_patience=10)
    result = _trainer._train_epoch(torch, model, batches, optimizer, "cpu", config, 1)
    stats = result["context_consistency"]
    assert len(steps) == 12
    assert sum(stats[name + "_selected_batches"] for name in GEOMETRIES) == stats["selected_batches"] > 0
    assert stats["mean_context_loss"] == pytest.approx(stats["mean_query_contribution"] + stats["mean_reference_contribution"])
    assert not any("kl" in key or "teacher" in key or "fallback" in key for key in stats)
