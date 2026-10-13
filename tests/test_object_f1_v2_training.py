"""Обучение трёх классов, независимая калибровка и переносимость checkpoint."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlsystem2.inference.contracts import ObjectSeparationConfig
from mlsystem2.models.api import create_model, load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec, ModelsError
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainRequest
from mlsystem2.train._object_f1 import object_loss, validate_objects


def _config(**kwargs):
    return TrainConfig(pipeline_variant="object_f1_v2", task="binary", quality_metric="objects",
        class_weights=[1, 1, 3], epochs=2, batch_size=1, device="cpu", learning_rate=1e-4,
        weight_decay=.01, loss="cross_entropy_tversky", early_stopping_patience=10,
        object_separation=ObjectSeparationConfig(profile="object_f1_v2"), **kwargs)


def _spec():
    return ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=1, parameters={
        "pipeline_variant": "object_f1_v2", "output_layout": "background_interior_boundary",
        "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}, "boundary_width": 8,
        "object_separation": ObjectSeparationConfig(profile="object_f1_v2").model_dump()})


def _batch(torch):
    ids = np.zeros((64, 64), np.int32)
    ids[8:56, 8:32] = 1
    ids[8:56, 32:56] = 2
    boundary = np.zeros_like(ids, bool)
    boundary[8:56, 31:33] = True
    image = torch.from_numpy(np.random.default_rng(42).uniform(0, 255, (1, 3, 64, 64)).astype(np.float32))
    masks = torch.from_numpy((ids > 0).astype(np.float32))[None, None]
    meta = {"object_instances": torch.from_numpy(ids)[None], "valid_pixels": torch.ones((1, 64, 64), dtype=torch.bool),
            "boundary_target": torch.from_numpy(boundary)[None].float(),
            "boundary_valid": torch.ones((1, 64, 64), dtype=torch.bool),
            "scene_ids": ["участки"], "scene_shapes": [{"width": 64, "height": 64}],
            "windows": [{"x": 0, "y": 0}], "overlap_pixels": 0}
    return image, masks, meta


def test_v2_loss_ignores_invalid_and_empty_supervision():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    _, masks, meta = _batch(torch)
    valid = meta["valid_pixels"][:, None].clone()
    valid[:, :, :6] = False
    logits = torch.zeros((1, 3, 64, 64), requires_grad=True)
    loss, _, _ = object_loss(torch, logits, masks, valid, meta, _config())
    loss.backward()
    assert torch.isfinite(loss) and not logits.grad[:, :, :6].any()
    assert logits.grad[:, :, 20, 31].abs().sum() > 0
    empty = torch.randn((1, 3, 64, 64), requires_grad=True)
    loss, _, _ = object_loss(torch, empty, masks, torch.zeros_like(valid), meta, _config())
    loss.backward()
    assert loss.item() == 0 and not empty.grad.any()


def test_validation_calibrates_boundary_but_test_keeps_saved_threshold():
    torch = pytest.importorskip("torch")
    image, masks, meta = _batch(torch)
    probabilities = np.full((1, 3, 64, 64), .05, np.float32)
    probabilities[:, 0] = .9
    probabilities[:, 0, 8:56, 8:56] = .1
    probabilities[:, 1, 8:56, 8:56] = .85
    probabilities[:, 1, 8:56, 31:33] = .5
    probabilities[:, 2, 8:56, 31:33] = .4

    class Model(torch.nn.Module):
        def forward(self, images):
            return torch.from_numpy(np.log(probabilities))

    config = _config()
    result = validate_objects(torch, Model(), [(image, masks, meta)], "cpu", config, 1, None)
    assert result["quality_f1"] == 1
    assert result["object_separation"]["boundary_threshold"] == .35
    assert len(result["object_threshold_sweep"]) == 3
    assert result["per_scene_metrics"][0]["merged_predictions"] == 0
    fixed = config.model_copy(update={"calibrate_object_threshold": False})
    test = validate_objects(torch, Model(), [(image, masks, meta)], "cpu", fixed, 1, None)
    assert len(test["object_threshold_sweep"]) == 1
    assert test["object_separation"]["boundary_threshold"] == .5
    assert test["quality_f1"] == pytest.approx(2 / 3)
    assert test["per_scene_metrics"][0]["merged_predictions"] == 1


def test_real_b0_training_checkpoint_and_onnx_roundtrip(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    from mlsystem2.training_ui_api._model_export import _export_segmentation_mask_onnx
    from mlsystem2.training_ui_api._object_inference import torch_object_predictor

    torch.set_num_threads(2)
    torch.manual_seed(42)
    batch = _batch(torch)
    handle = create_model(_spec())
    result = train_model(TrainRequest(model=handle, train_loader=[batch], val_loader=[batch], test_loader=[batch],
        config=_config(), checkpoint_dir=str(tmp_path / "модель"), sample_size=64))
    # Синтетический smoke проверяет проводку, не оценивает обобщение на территории.
    assert result.epochs_total == 2
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=result.best_checkpoint_path, map_location="cpu"))
    assert loaded.model.spec.parameters["output_layout"] == "background_interior_boundary"
    chosen = loaded.model.spec.parameters["object_separation"]
    assert chosen == loaded.artifact.metadata["object_separation"]
    assert chosen == result.diagnostics["test_metrics"]["object_separation"]
    assert loaded.artifact.metadata["checkpoint_selection_metric"] == "val/object_f1"
    assert all(len(epoch.val_object_threshold_sweep) == 3 for epoch in result.history)
    original = create_model(loaded.model.spec)
    original.model.load_state_dict(torch.load(result.best_checkpoint_path, map_location="cpu")["model_state_dict"])
    original.model.eval()
    loaded.model.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(original.model(batch[0]), loaded.model.model(batch[0]), rtol=0, atol=0)
    path = tmp_path / "модель.onnx"
    _export_segmentation_mask_onnx(model=loaded.model.model, input_channels=3, output_channels=1,
        sample_size=64, threshold=.5, onnx_path=path, probability_output=True,
        object_output=True, object_profile="object_f1_v2")
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    actual = reference.ReferenceEvaluator(graph).run(["probabilities"], {"input": batch[0].numpy()})[0]
    expected = torch_object_predictor(torch, loaded.model.model, "cpu", 3, profile="object_f1_v2")(batch[0].numpy())
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


def test_v2_rejects_v1_layout_and_two_class_weights():
    wrong = _spec().model_copy(update={"parameters": {**_spec().parameters, "output_layout": "background_foreground_boundary"}})
    with pytest.raises(ModelsError, match="схему"):
        create_model(wrong)
    raw = _config().model_dump()
    raw["class_weights"] = [1, 1]
    with pytest.raises(ValueError, match="веса"):
        TrainConfig.model_validate(raw)
    raw = _config().model_dump()
    raw["object_separation"] = ObjectSeparationConfig().model_dump()
    with pytest.raises(ValueError, match="совпадать"):
        TrainConfig.model_validate(raw)


def test_boundary_diagnostics_do_not_depend_on_window_order():
    from mlsystem2.tile_preparation._object_targets import boundary_targets_v2
    from mlsystem2.train._object_f1 import _merge_boundary_target

    truth = np.zeros((64, 64), np.int32)
    truth[6:58, 6:46] = 1
    windows = [(x, y) for y in (0, 16, 32) for x in (0, 16, 32)]
    results = []
    for order in (windows, list(reversed(windows))):
        edges, valid, rank = np.zeros_like(truth, bool), np.zeros_like(truth, bool), np.zeros_like(truth, np.uint16)
        for x, y in order:
            target, usable = boundary_targets_v2(truth[y:y+32, x:x+32], np.ones((32, 32), bool))
            _merge_boundary_target(edges, valid, rank, x, y, 32, target > 0, usable)
        results.append((edges, valid, rank))
    for forward, backward in zip(*results):
        np.testing.assert_array_equal(forward, backward)


def test_mlflow_keeps_structured_test_reports_out_of_scalar_metrics(monkeypatch):
    from mlsystem2.mlflow_adapter import _client
    from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
    from mlsystem2.train.contracts import TrainResult

    logged = {}
    monkeypatch.setattr(_client, "_ensure_run_active", lambda run: SimpleNamespace(log_metric=lambda key, value: logged.update({key: value})))
    result = TrainResult(history=[], epochs_total=0, training_time_sec=0, best_checkpoint_path="best.pt",
                         final_checkpoint_path=None, artifacts=[], diagnostics={"test_metrics": {
                             "object_f1": .8, "object_separation": {"boundary_threshold": .35},
                             "per_scene_metrics": [{"scene_id": "сцена", "merged_predictions": 0}]}})
    run = MLflowRunRef(run_id="v2", experiment_name="проверка", tracking_uri="http://localhost", active=True)
    _client.log_training_metrics(run, result)
    assert logged["test/object_f1"] == .8
    assert "test/object_separation" not in logged
