"""Двунаправленный штраф исправляет полное окно, не обучая правильного учителя."""

from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from mlsystem2.train._consistency import context_consistency_loss

VARIANT = "next_gen4_consistency_v3"
POLICY = {"version": "bidirectional_confident_teacher_kl_v1", "weight": .25, "probability": .5,
    "crop_side_fractions": [.5, .75], "teacher_confidence": .9,
    "source": "within_retained_train_window", "evaluation_mode": True, "class_balance": True,
    "direction_reduction": "mean_of_active_directions"}


class ContextModel(torch.nn.Module):
    def __init__(self, full=-.5, crop=4., fail=False):
        super().__init__()
        self.full = torch.nn.Parameter(torch.tensor(float(full)))
        self.crop = torch.nn.Parameter(torch.tensor(float(crop)))
        self.bn = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout(.9)
        self.seen = []
        self.fail = fail

    def forward(self, image, **kwargs):
        self.seen.append(image.detach().clone())
        if self.fail and len(self.seen) == 2:
            raise RuntimeError("Проверочная ошибка вырезки")
        self.dropout(self.bn(image))
        bias = self.full if image.shape[-1] == 128 else self.crop
        foreground = bias.expand(len(image), 1, *image.shape[-2:])
        return torch.cat((-foreground, foreground), 1)


def invoke(model, reason=None, bidirectional=True):
    images = torch.arange(128*128).reshape(1,1,128,128).float().expand(1,4,128,128).clone()+1
    masks = torch.ones((1,1,128,128))
    valid = torch.ones_like(masks, dtype=torch.bool)
    negative = torch.ones((1,2,128,128), dtype=torch.bool) if reason == "ambiguous" else None
    if reason == "nodata":
        valid[:] = False
    for index in range(1, 30):
        loss, stats = context_consistency_loss(torch, model, images, masks, valid, negative,
            seed=42, epoch=1, batch_index=index, bidirectional=bidirectional)
        if model.seen or reason in ("nodata", "ambiguous"):
            return loss, stats, images
    raise AssertionError("Не найден выбранный пакет")


def test_reverse_teacher_supplies_missing_full_window_gradient_without_teaching_teacher():
    model = ContextModel().train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    loss, stats, images = invoke(model)
    assert stats["forward_supervised_pixels"] == 0
    side = model.seen[1].shape[-1]
    assert stats["reverse_supervised_pixels"] == side*side
    assert stats["supervised_pixels"] == side*side
    assert stats["selected_batches"] == stats["reverse_selected_batches"] == 1
    assert len(model.seen) == 2 and side in (64,96)
    first = int(model.seen[1][0,0,0,0]-1)
    top, left = divmod(first,128)
    torch.testing.assert_close(model.seen[1], images[...,top:top+side,left:left+side])
    torch.testing.assert_close(rng, torch.get_rng_state())
    assert model.bn.num_batches_tracked == 0
    assert [module.training for module in model.modules()] == states
    loss.backward()
    assert model.full.grad < 0 and model.crop.grad is None


def test_historical_one_way_rule_still_ignores_uncertain_full_teacher():
    model = ContextModel()
    loss, stats, _ = invoke(model, bidirectional=False)
    assert loss is None and stats["supervised_pixels"] == 0
    assert model.full.grad is model.crop.grad is None


def test_forward_direction_stays_useful_and_only_student_gets_gradient():
    model = ContextModel(full=4., crop=-1.)
    loss, stats, _ = invoke(model)
    assert stats["forward_supervised_pixels"] > 0 and stats["reverse_supervised_pixels"] == 0
    loss.backward()
    assert model.full.grad is None and model.crop.grad < 0


def test_two_active_directions_are_averaged_not_doubled():
    model = ContextModel(full=4., crop=3.)
    loss, stats, _ = invoke(model)
    full, crop = torch.tensor([-4.,4.]), torch.tensor([-3.,3.])
    forward = torch.nn.functional.kl_div(crop.log_softmax(0), full.softmax(0), reduction="sum")
    reverse = torch.nn.functional.kl_div(full.log_softmax(0), crop.softmax(0), reduction="sum")
    torch.testing.assert_close(loss, .25*(forward+reverse)/2, rtol=1e-4, atol=1e-7)
    assert stats["forward_selected_batches"] == stats["reverse_selected_batches"] == 1


@pytest.mark.parametrize("reason", ["nodata", "ambiguous", "both_uncertain", "both_wrong"])
def test_no_unsafe_supervision(reason):
    value = 0. if reason == "both_uncertain" else -4. if reason == "both_wrong" else 4.
    model = ContextModel(full=value,crop=value).train()
    loss, stats, _ = invoke(model, reason)
    assert loss is None and stats["supervised_pixels"] == 0
    assert model.bn.num_batches_tracked == 0
    assert all(module.training for module in model.modules())


def test_modes_restored_after_reverse_pass_failure():
    model = ContextModel(fail=True).train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError,match="ошибка вырезки"):
        invoke(model)
    assert [module.training for module in model.modules()] == states


def spec():
    from mlsystem2.models.contracts import ModelSpec
    return ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1, parameters={
        "pipeline_variant": VARIANT, "context_consistency": dict(POLICY),
        "decoder_normalization": {"version":"per_pixel_channel_layer_norm_v1","epsilon":1e-5,"affine":True},
        "augmentation_policy":"validity_guard_v4_context_crop_v1",
        "context_crop":{"area_scale":[.25,1.],"ratio":[1.,1.],"probability":.5,"output_size":512,"source":"within_retained_train_window"},
        "preprocessing":{"mode":"global_minmax","epsilon":1e-6,"nodata":0.,"mask_output_nodata":True,
            "source":"retained_train_windows","low":[0.]*4,"high":[255.]*4,"valid_pixel_count":4096,"window_count":4}})


@pytest.mark.parametrize("architecture",[f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size",[512,768,1024,1536])
def test_profile_only_changes_variant_not_training_defaults(architecture,size):
    from mlsystem2.training_ui_api._templates import initial_templates,sanitize_template_config
    before=sanitize_template_config({"train.pipeline_variant":"next_gen4_consistency_v2","tile_preparation.tile_size":size},architecture=architecture)
    after=sanitize_template_config({"train.pipeline_variant":VARIANT,"tile_preparation.tile_size":size},architecture=architecture)
    assert after=={**before,"train.pipeline_variant":VARIANT}
    schema=next(item for item in initial_templates() if item["architecture"]==architecture)["config_schema"]
    assert VARIANT in schema["fields"][0]["options"]
    assert schema["pipeline_defaults"][VARIANT]["train.pipeline_variant"]==VARIANT


def test_checkpoint_policy_required_and_autonomous_load_keeps_forward(tmp_path,monkeypatch):
    from mlsystem2.models import _factory
    from mlsystem2.models.api import create_model,load_checkpoint,save_checkpoint
    from mlsystem2.models.contracts import ModelsError,LoadCheckpointRequest,SaveCheckpointRequest
    monkeypatch.setattr(_factory,"_create_smp_segformer",lambda *args,**kwargs:torch.nn.Conv2d(4,2,1))
    broken=spec().model_copy(deep=True)
    broken.parameters["context_consistency"]["version"]="confident_teacher_kl_v1"
    with pytest.raises(ModelsError,match="политику согласованности"):
        create_model(broken)
    model=create_model(spec())
    path=tmp_path/"best.pt"
    save_checkpoint(SaveCheckpointRequest(model=model,checkpoint_uri=str(path),metadata={}))
    loaded=load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path),map_location="cpu"))
    image=torch.full((1,4,64,64),40.)
    torch.testing.assert_close(model.model(image),loaded.model.model(image))
    assert loaded.model.spec.parameters["context_consistency"]==POLICY


def test_continuation_rejects_missing_reverse_policy(monkeypatch):
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError
    model_spec=spec()
    loaded=SimpleNamespace(spec=model_spec.model_copy(deep=True))
    loaded.spec.parameters["context_consistency"]["version"]="confident_teacher_kl_v1"
    monkeypatch.setattr(_runner,"_model_spec",lambda *args:model_spec)
    settings=SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT,initial_checkpoint_uri="best.pt",device="cpu"))
    with pytest.raises(TrainPipelineError,match="согласованности"):
        _runner._load_or_create_model(settings,SimpleNamespace(load_checkpoint=lambda request:SimpleNamespace(model=loaded)))


def test_model_graph_and_initial_weights_are_identical_to_v2():
    from mlsystem2.models.api import create_model
    previous=spec().model_copy(deep=True)
    previous.parameters["pipeline_variant"]="next_gen4_consistency_v2"
    previous.parameters["context_consistency"]["version"]="confident_teacher_kl_v1"
    del previous.parameters["context_consistency"]["direction_reduction"]
    torch.manual_seed(123)
    before=create_model(previous).model.eval()
    torch.manual_seed(123)
    after=create_model(spec()).model.eval()
    assert before.state_dict().keys()==after.state_dict().keys()
    for key,value in before.state_dict().items():
        torch.testing.assert_close(value,after.state_dict()[key],rtol=0,atol=0)
    with torch.inference_mode():
        image=torch.full((1,4,128,128),40.)
        torch.testing.assert_close(before(image),after(image),rtol=0,atol=0)
