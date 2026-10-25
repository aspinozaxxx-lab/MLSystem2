"""Надзор по train-эталону закрывает пиксели без обоих уверенных учителей."""

from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from mlsystem2.train._consistency import context_consistency_loss, _ground_truth_fallback_loss
from mlsystem2.train.contracts import TrainError

VARIANT = "next_gen4_consistency_v4"
POLICY = {"version":"bidirectional_confident_teacher_kl_with_gt_fallback_v1","weight":.25,"probability":.5,
    "crop_side_fractions":[.5,.75],"teacher_confidence":.9,"source":"within_retained_train_window",
    "evaluation_mode":True,"class_balance":True,"direction_reduction":"mean_of_active_directions",
    "uncovered_supervision":"ground_truth_cross_entropy_v1"}


class ViewModel(torch.nn.Module):
    def __init__(self, full=0., crop=0., fail=False):
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
        value = self.full if image.shape[-1] == 128 else self.crop
        foreground = value.expand(len(image),1,*image.shape[-2:])
        return torch.cat((-foreground,foreground),1)


def invoke(model, *, reason=None, fallback=True):
    image = torch.arange(128*128).reshape(1,1,128,128).float().expand(1,4,128,128).clone()+1
    target = torch.full((1,1,128,128),99. if reason=="invalid_class" else 1.)
    valid = torch.ones_like(target,dtype=torch.bool)
    if reason=="nodata":
        valid[:] = False
        target[:] = 99
    negative = torch.ones((1,2,128,128),dtype=torch.bool) if reason=="ambiguous" else None
    for index in range(1,30):
        loss,stats = context_consistency_loss(torch,model,image,target,valid,negative,
            seed=42,epoch=1,batch_index=index,bidirectional=True,ground_truth_fallback=fallback)
        if model.seen or reason in ("nodata","ambiguous"):
            return loss,stats,image
    raise AssertionError("Не найден выбранный пакет")


@pytest.mark.parametrize("value",[0.,-4.])
def test_unqualified_views_receive_correct_target_gradient_without_confident_teacher(value):
    model = ViewModel(full=value,crop=value).train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    loss,stats,image = invoke(model)
    side = model.seen[1].shape[-1]
    expected = .25*torch.nn.functional.cross_entropy(torch.tensor([[-value,value]]),torch.tensor([1]))
    torch.testing.assert_close(loss,expected)
    assert stats["fallback_selected_batches"] == stats["selected_batches"] == 1
    assert stats["fallback_supervised_pixels"] == stats["supervised_pixels"] == 2*side*side
    assert stats["forward_selected_batches"] == stats["reverse_selected_batches"] == 0
    assert stats["kl_sum"] == stats["teacher_contribution_sum"] == 0
    assert stats["context_loss_sum"] == pytest.approx(stats["fallback_ce_contribution_sum"])
    assert len(model.seen)==2 and side in (64,96)
    first = int(model.seen[1][0,0,0,0]-1)
    top,left = divmod(first,128)
    torch.testing.assert_close(model.seen[1],image[...,top:top+side,left:left+side])
    assert [module.training for module in model.modules()] == states
    assert model.bn.num_batches_tracked == 0
    torch.testing.assert_close(rng,torch.get_rng_state())
    loss.backward()
    assert model.full.grad < 0 and model.crop.grad < 0


@pytest.mark.parametrize("full,crop",[(-.5,4.),(4.,-1.),(4.,3.)])
def test_without_uncovered_pixels_v4_keeps_exact_v3_loss_and_gradients(full,crop):
    before,after = ViewModel(full,crop),ViewModel(full,crop)
    loss3,stats3,_ = invoke(before,fallback=False)
    loss4,stats4,_ = invoke(after)
    torch.testing.assert_close(loss3,loss4,rtol=0,atol=0)
    assert stats4["fallback_supervised_pixels"] == 0
    assert stats4["supervised_pixels"] == stats3["supervised_pixels"]
    loss3.backward()
    loss4.backward()
    for old,new in ((before.full.grad,after.full.grad),(before.crop.grad,after.crop.grad)):
        if old is None:
            assert new is None or torch.count_nonzero(new)==0
        else:
            torch.testing.assert_close(old,new,rtol=0,atol=0)


@pytest.mark.parametrize("reason",["nodata","ambiguous"])
def test_invalid_or_ambiguous_pixels_do_not_get_fallback(reason):
    model = ViewModel().train()
    loss,stats,_ = invoke(model,reason=reason)
    assert loss is None and stats["supervised_pixels"] == stats["fallback_supervised_pixels"] == 0
    assert model.full.grad is model.crop.grad is None
    assert model.bn.num_batches_tracked == 0


def test_invalid_valid_target_restores_modes_and_fails_explicitly():
    model = ViewModel().train()
    model.dropout.eval()
    states = [module.training for module in model.modules()]
    with pytest.raises(TrainError,match="Недопустимый класс"):
        invoke(model,reason="invalid_class")
    assert [module.training for module in model.modules()] == states
    assert model.bn.num_batches_tracked == 0


def test_fallback_requires_bidirectional_policy():
    with pytest.raises(TrainError,match="двух направлений"):
        context_consistency_loss(torch,ViewModel(),torch.ones(1,4,128,128),torch.ones(1,1,128,128),
            None,None,seed=42,epoch=1,batch_index=1,ground_truth_fallback=True)


def test_ignored_invalid_labels_and_pixels_have_zero_gradient():
    full = torch.zeros(1,2,4,4,requires_grad=True)
    crop = torch.zeros_like(full,requires_grad=True)
    valid = torch.zeros(1,4,4,dtype=torch.bool)
    valid[:,1:3,1:3] = True
    target = torch.full((1,4,4),99,dtype=torch.long)
    target[valid] = 1
    loss,stats = _ground_truth_fallback_loss(torch,full,crop,[full.detach().softmax(1),crop.detach().softmax(1)],
        target,valid,{"supervised_pixels":0})
    loss.backward()
    assert stats["fallback_supervised_pixels"] == 8
    assert (full.grad[:,1][valid] < 0).all() and (crop.grad[:,1][valid] < 0).all()
    assert torch.count_nonzero(full.grad.permute(0,2,3,1)[~valid]) == 0
    assert torch.count_nonzero(crop.grad.permute(0,2,3,1)[~valid]) == 0


@pytest.mark.parametrize("water_pixels",[1,8,15])
def test_fallback_balances_present_classes_not_pixel_counts(water_pixels):
    full = torch.zeros(1,2,4,4)
    full[:,1] = -1
    full.requires_grad_(True)
    crop = full.detach().clone().requires_grad_(True)
    target = torch.zeros(1,4,4,dtype=torch.long)
    target.view(-1)[:water_pixels] = 1
    loss,_ = _ground_truth_fallback_loss(torch,full,crop,[full.detach().softmax(1),crop.detach().softmax(1)],
        target,torch.ones_like(target,dtype=torch.bool),{"supervised_pixels":0})
    expected = .25*(-torch.tensor([0.,-1.]).log_softmax(0).mean())
    torch.testing.assert_close(loss,expected)


def spec():
    from mlsystem2.models.contracts import ModelSpec
    return ModelSpec(name="smp_segformer_b0",input_channels=4,output_channels=1,pretrained=False,parameters={
        "pipeline_variant":VARIANT,"context_consistency":dict(POLICY),
        "decoder_normalization":{"version":"per_pixel_channel_layer_norm_v1","epsilon":1e-5,"affine":True},
        "augmentation_policy":"validity_guard_v4_context_crop_v1",
        "context_crop":{"area_scale":[.25,1.],"ratio":[1.,1.],"probability":.5,"output_size":512,"source":"within_retained_train_window"},
        "preprocessing":{"mode":"global_minmax","epsilon":1e-6,"nodata":0.,"mask_output_nodata":True,
            "source":"retained_train_windows","low":[0.]*4,"high":[255.]*4,"valid_pixel_count":4096,"window_count":4}})


@pytest.mark.parametrize("architecture",[f"smp_segformer_b{index}" for index in range(4)])
@pytest.mark.parametrize("size",[512,768,1024,1536])
def test_profile_preserves_all_v3_defaults_except_variant(architecture,size):
    from mlsystem2.training_ui_api._templates import initial_templates,sanitize_template_config
    before = sanitize_template_config({"train.pipeline_variant":"next_gen4_consistency_v3","tile_preparation.tile_size":size},architecture=architecture)
    after = sanitize_template_config({"train.pipeline_variant":VARIANT,"tile_preparation.tile_size":size},architecture=architecture)
    assert after == {**before,"train.pipeline_variant":VARIANT}
    schema = next(item for item in initial_templates() if item["architecture"]==architecture)["config_schema"]
    assert VARIANT in schema["fields"][0]["options"]
    assert schema["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT


@pytest.mark.parametrize("change",["missing_marker","old_policy","wrong_reduction"])
def test_checkpoint_rejects_incompatible_fallback_policy(change):
    from mlsystem2.models.api import create_model
    from mlsystem2.models.contracts import ModelsError
    broken = spec().model_copy(deep=True)
    if change=="missing_marker":
        del broken.parameters["context_consistency"]["uncovered_supervision"]
    elif change=="old_policy":
        broken.parameters["context_consistency"]["version"] = "bidirectional_confident_teacher_kl_v1"
    else:
        broken.parameters["context_consistency"]["direction_reduction"] = "sum"
    with pytest.raises(ModelsError,match="политику согласованности"):
        create_model(broken)


def test_continuation_rejects_missing_fallback_policy(monkeypatch):
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError
    expected = spec()
    loaded = SimpleNamespace(spec=expected.model_copy(deep=True))
    del loaded.spec.parameters["context_consistency"]["uncovered_supervision"]
    monkeypatch.setattr(_runner,"_model_spec",lambda *args:expected)
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT,initial_checkpoint_uri="best.pt",device="cpu"))
    with pytest.raises(TrainPipelineError,match="согласованности"):
        _runner._load_or_create_model(settings,SimpleNamespace(load_checkpoint=lambda request:SimpleNamespace(model=loaded)))


def test_mixed_kl_and_target_ce_have_one_budget_and_detached_teacher():
    full_p = torch.tensor([[[.97,.97,.65,.65],[.97,.97,.65,.65]]])
    crop_p = torch.full_like(full_p,.2)
    full = torch.stack([torch.log1p(-full_p),full_p.log()],1).requires_grad_(True)
    crop = torch.stack([torch.log1p(-crop_p),crop_p.log()],1).requires_grad_(True)
    target = torch.ones(1,2,4,dtype=torch.long)
    qualified = torch.zeros_like(target,dtype=torch.bool)
    qualified[:,:,:2] = True
    uncovered = ~qualified
    loss,stats = _ground_truth_fallback_loss(torch,full,crop,[full.detach().softmax(1),crop.detach().softmax(1)],
        target,torch.ones_like(qualified),{"supervised_pixels":0})
    kl = torch.nn.functional.kl_div(crop.log_softmax(1),full.detach().softmax(1),reduction="none").sum(1)
    ce_crop = torch.nn.functional.cross_entropy(crop,target,reduction="none")
    ce_full = torch.nn.functional.cross_entropy(full,target,reduction="none")
    expected = .25*(torch.where(uncovered,ce_crop,kl).mean()+ce_full[uncovered].mean())/2
    torch.testing.assert_close(loss,expected)
    assert stats["fallback_supervised_pixels"] == 8 and stats["supervised_pixels"] == 12
    assert stats["context_loss_sum"] == pytest.approx(stats["teacher_contribution_sum"]+stats["fallback_ce_contribution_sum"],abs=1e-6)
    loss.backward()
    assert torch.count_nonzero(full.grad[:,1][qualified]) == 0
    assert (full.grad[:,1][uncovered] < 0).all() and (crop.grad[:,1] < 0).all()


def test_model_graph_initial_weights_and_checkpoint_forward_match_v3(tmp_path):
    from mlsystem2.models.api import create_model,load_checkpoint,save_checkpoint
    from mlsystem2.models.contracts import LoadCheckpointRequest,SaveCheckpointRequest
    previous = spec().model_copy(deep=True)
    previous.parameters["pipeline_variant"] = "next_gen4_consistency_v3"
    previous.parameters["context_consistency"]["version"] = "bidirectional_confident_teacher_kl_v1"
    del previous.parameters["context_consistency"]["uncovered_supervision"]
    torch.manual_seed(123)
    before = create_model(previous)
    torch.manual_seed(123)
    after = create_model(spec())
    assert before.model.state_dict().keys() == after.model.state_dict().keys()
    for key,value in before.model.state_dict().items():
        torch.testing.assert_close(value,after.model.state_dict()[key],rtol=0,atol=0)
    path = tmp_path/"best.pt"
    save_checkpoint(SaveCheckpointRequest(model=after,checkpoint_uri=str(path),metadata={}))
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path),map_location="cpu"))
    assert loaded.model.spec.parameters["context_consistency"] == POLICY
    before.model.eval()
    after.model.eval()
    loaded.model.model.eval()
    with torch.inference_mode():
        image = torch.full((1,4,128,128),40.)
        torch.testing.assert_close(before.model(image),after.model(image),rtol=0,atol=0)
        torch.testing.assert_close(after.model(image),loaded.model.model(image),rtol=0,atol=0)
