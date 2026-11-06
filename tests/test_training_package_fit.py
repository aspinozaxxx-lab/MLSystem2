"""Проверки полного пакета, защиты начальных весов и рабочего оценщика."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from mlsystem2.models.contracts import ModelHandle, ModelSpec, LoadedCheckpoint, CheckpointArtifact
from mlsystem2.tile_preparation import _dataloader
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest, TileSceneSource
from mlsystem2.train import _trainer
from mlsystem2.train.contracts import TrainConfig, TrainRequest
from mlsystem2.training_ui_api import _training_package_fit, _pseudo_runner


def _values(f1, precision=0.8):
    return {
        "quality_f1": f1, "quality_precision": precision, "quality_recall": f1,
        "threshold": 0.9, "pixel_f1": f1, "pixel_precision": precision,
        "pixel_recall": f1, "object_f1": 0.3, "object_precision": 0.4,
        "object_recall": 0.2,
        "package_evaluation": {"scope": "training_package", "independent": False,
                               "quality_counts": {"fp": 10, "fn": 20}},
    }


@pytest.mark.parametrize("scores,selected", [([0.8, 0.7, 0.75], 0), ([0.8, 0.85, 0.83], 1)])
@pytest.mark.parametrize("variant", ["legacy", "next_gen2", "next_gen3"])
def test_best_protects_baseline_and_selects_package_score(tmp_path, monkeypatch, scores, selected, variant):
    torch = pytest.importorskip("torch")
    if variant != "legacy":
        pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(2)
    class Trainable(torch.nn.Conv2d):
        def forward(self, images, return_two_class_logits=False):
            return super().forward(images)

    core = Trainable(4, 1 if variant == "legacy" else 2, 1)
    model = ModelHandle(spec=ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1,
                                      parameters={"pipeline_variant": variant}), model=core)
    images = torch.ones(2, 4, 8, 8)
    masks = torch.zeros(2, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 1
    snapshots, calls, events = {}, [], []

    def evaluate(handle, epoch, pause):
        pause()
        assert handle is model
        calls.append(epoch)
        snapshots[epoch] = {key: value.clone() for key, value in core.state_dict().items()}
        return _values(scores[epoch])

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("Внутренняя validation не должна вызываться")

    monkeypatch.setattr(_trainer, "_validate_epoch", unexpected_validation)
    result = _trainer.train_model(TrainRequest(
        model=model, train_loader=[(images, masks, {"valid_pixels": torch.ones_like(masks, dtype=torch.bool)})],
        val_loader=None, package_evaluator=evaluate, checkpoint_dir=str(tmp_path),
        config=TrainConfig(epochs=2, batch_size=2, device="cpu", learning_rate=1e-3,
                           weight_decay=0, loss="bce_dice" if variant == "legacy" else "cross_entropy_tversky",
                           early_stopping_patience=5, pipeline_variant=variant,
                           class_weights=[] if variant == "legacy" else [1, 1]),
    ), progress_sink=events.append)
    assert calls == [0, 1, 2]
    assert result.epochs_total == 2 and len(result.history) == 3
    assert all(item.val_loss is None for item in result.history)
    assert result.diagnostics["checkpoint_selection"]["epoch"] == selected
    assert result.diagnostics["checkpoint_selection"]["baseline_selected"] == (selected == 0)
    best = torch.load(result.best_checkpoint_path, map_location="cpu", weights_only=False)
    final = torch.load(result.final_checkpoint_path, map_location="cpu", weights_only=False)
    assert best["metadata"]["epoch"] == selected
    assert best["metadata"]["confidence_threshold"] == 0.9
    assert best["metadata"]["evaluation_scope"] == "training_package"
    for key, value in best["model_state_dict"].items():
        assert torch.equal(value, snapshots[selected][key])
    assert any(not torch.equal(value, snapshots[0][key]) for key, value in final["model_state_dict"].items())
    assert events[0].epoch == 0 and events[0].metrics.package_evaluation is not None


def _scene(tmp_path, index):
    path = tmp_path / f"снимок_{index}.tif"
    valid = np.full((32, 32), 255, np.uint8)
    valid[:4] = 0
    with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=4, ALPHA="NO",
                       dtype="uint8", crs="EPSG:3857", transform=from_origin(index * 40, 32, 1, 1)) as dataset:
        image = np.full((4, 32, 32), 40 + index, np.uint8)
        image[:, 8:24, 8:24] = 180
        dataset.write(image)
        dataset.write_mask(valid)
        dataset.descriptions = ("RED", "GRN", "BLU", "NIR")
        dataset.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                               rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.undefined)
    annotation = tmp_path / f"снимок_{index}.geojson"
    annotation.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    return TileSceneSource(scene_id=str(index), image_path=path, annotation_file=annotation)


@pytest.mark.parametrize("tile_size", [32, 64])
def test_exhaustive_loader_covers_every_window_each_epoch_and_masks_nodata(tmp_path, monkeypatch, tile_size):
    pytest.importorskip("torch")
    settings = SimpleNamespace(tile_size=tile_size, context=0, stride=tile_size // 2, seed=42, augmentation_level=0,
        positive_factor=0.5, hard_negative_factor=0, background_factor=0.5, class_balance=False, num_workers=8)
    monkeypatch.setattr(_dataloader, "get_settings", lambda: SimpleNamespace(tile_preparation=settings))
    loader = _dataloader.create_tile_dataloader(TileDataloaderRequest(
        scenes=[_scene(tmp_path, i) for i in range(6)], batch_size=4, mode="train",
        input_channels=4, pipeline_variant="next_gen2", exhaustive_training=True,
    ))
    assert loader.num_workers == 0 and len(loader.dataset) == 6
    assert loader.dataset.tile_split_manifest["strategy"] == "fit_package"
    orders = []
    for epoch in (1, 2, 1):
        loader.sampler.set_epoch(epoch)
        order = list(loader.sampler)
        assert sorted(index for index, _, _ in order) == list(range(6))
        orders.append(order)
        batches = list(loader)
        assert sum(batch[0].shape[0] for batch in batches) == 6
        for _, _, metadata in batches:
            assert not metadata["valid_pixels"][..., :4, :].any()
            if tile_size > 32:
                assert not metadata["valid_pixels"][..., 32:, :].any()
                assert not metadata["valid_pixels"][..., :, 32:].any()
    assert orders[0] == orders[2] and orders[0] != orders[1]


def test_evaluator_uses_live_production_inference_with_frozen_profile(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    scene = _scene(tmp_path, 0)
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 1
    mask_path = tmp_path / "эталон.png"
    Image.fromarray(mask).save(mask_path)

    class Predict(torch.nn.Module):
        def forward(self, images, return_two_class_logits=False):
            probability = (images[:, :1] > 100).float()
            return torch.cat([5 - 10 * probability, 10 * probability - 5], dim=1)

    handle = ModelHandle(spec=ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1,
                                       parameters={"pipeline_variant": "next_gen2"}), model=Predict())
    loaded = LoadedCheckpoint(model=handle, artifact=CheckpointArtifact(uri="исходный.pt", format="torch_pt",
                              metadata={"confidence_threshold": 0.5, "task": "binary", "sample_size": 64,
                                        "inference_core_size": 64}))
    monkeypatch.setattr(_training_package_fit, "load_checkpoint", lambda _request: loaded)
    config = {"initial_checkpoint_uri": "исходный.pt", "pipeline_variant": "next_gen2", "tile_size": 32,
        "stride": 16, "context": 0, "device": "cpu", "input_channels": 4, "quality_metric": "pixel",
        "test_sample_id": "пакет", "test_sample_revision": 7, "inference_config_hash": "зафиксирован",
        "inference_template_id": None, "inference_template_version": None, "postprocess_config": {},
        "evaluations_root": str(tmp_path / "оценки"), "tiles": [{"index": 1, "image_path": str(scene.image_path),
        "mask_path": str(mask_path), "geojson_path": str(scene.annotation_file), "clip_to_valid_data": True}]}
    path = tmp_path / "пакет.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    evaluate = _training_package_fit.create_evaluator(path)
    monkeypatch.setattr(_pseudo_runner, "load_checkpoint", lambda *_args: pytest.fail("Повторная загрузка модели"))
    pauses = []
    metrics = evaluate(handle, 0, lambda: pauses.append(True))
    loaded.artifact.metadata.update(sample_size=32, inference_core_size=32)
    direct = _pseudo_runner.run_test_sample_f1({**config, "run_root": str(tmp_path / "эталонная_оценка")},
                                               loaded_checkpoint=loaded)
    report = metrics["package_evaluation"]["report"]
    for key in ("true_positive", "false_positive", "false_negative", "threshold"):
        assert report[key] == direct[key]
    assert metrics["threshold"] == 0.9 and metrics["pixel_f1"] == 1
    assert pauses and metrics["package_evaluation"]["test_sample_revision"] == 7


def test_fit_metric_descriptions_are_explicit_about_training_scope():
    from mlsystem2.mlflow_adapter._metric_descriptions import describe_metric
    for name in ("fit/pixel_f1", "fit/object_precision", "fit/quality_precision", "fit/threshold",
                 "fit/false_positive", "fit/false_negative"):
        entry = describe_metric(name, {})
        assert entry["meaning"] and entry["purpose"]
    assert "не независимая" in describe_metric("fit/pixel_f1", {})["purpose"]


@pytest.mark.parametrize("fitting", [True, False])
def test_fitting_preserves_batchnorm_and_disables_dropout_while_weights_learn(fitting):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.BatchNorm2d(4),
                                torch.nn.Dropout2d(0.9), torch.nn.Conv2d(4, 1, 1))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    images, target = torch.randn(2, 4, 8, 8) + 10, torch.ones(2, 1, 8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainConfig(epochs=1, batch_size=2, device="cpu", learning_rate=1e-3,
                         weight_decay=0, loss="bce_dice", early_stopping_patience=2)
    _trainer._train_epoch(torch, model, [(images, target)], optimizer, torch.device("cpu"),
                          config, 1, fitting_package=fitting)
    assert model.training is not fitting and model[2].training is not fitting
    assert not torch.equal(before["0.weight"], model[0].weight)
    assert model[0].weight.grad is not None
    if fitting:
        for key in ("1.running_mean", "1.running_var", "1.num_batches_tracked"):
            assert torch.equal(before[key], model.state_dict()[key])
    else:
        assert model[1].num_batches_tracked == 1


def test_fitting_keeps_learning_when_loss_improves_despite_flat_f1(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    core = torch.nn.Conv2d(4, 1, 1)
    model = ModelHandle(spec=ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1), model=core)
    images, targets = torch.ones(2, 4, 8, 8), torch.ones(2, 1, 8, 8)
    result = _trainer.train_model(TrainRequest(
        model=model, train_loader=[(images, targets)], val_loader=None,
        package_evaluator=lambda *_args: _values(0.8), checkpoint_dir=str(tmp_path),
        config=TrainConfig(epochs=8, batch_size=2, device="cpu", learning_rate=0.01,
                           weight_decay=0, loss="bce_dice", early_stopping_patience=1),
    ))
    assert result.epochs_total == 8
    assert result.history[-1].train_loss < result.history[1].train_loss
    assert all(item.learning_rate == 0.01 for item in result.history)
    assert result.diagnostics["checkpoint_selection"]["baseline_selected"] is True


@pytest.mark.parametrize("precisions,selected", [([0.8, 0.8, 0.7], 0), ([0.8, 0.9, 0.7], 1)])
def test_mlflow_roundtrip_selects_same_package_checkpoint_including_epoch_zero(tmp_path, monkeypatch, precisions, selected):
    import mlflow
    from mlsystem2.mlflow_adapter import api
    from mlsystem2.mlflow_adapter.contracts import MLflowStartRunRequest, MLflowRunStatus
    from mlsystem2.train._package_fit import package_epoch_metrics
    from mlsystem2.train.contracts import TrainResult

    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    monkeypatch.delenv("MLSYSTEM2_MLFLOW_RUN_ID_FILE", raising=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    name = "Подгонка размеченного пакета"
    client.create_experiment(name, artifact_location=(tmp_path / "артефакты").as_uri())
    try:
        run = api.start_run(MLflowStartRunRequest(enabled=True, tracking_uri=uri, experiment_name=name,
            dataset="Учебный пакет", tags={"pipeline_variant": "next_gen2", "evaluation_scope": "training_package",
            "checkpoint_selection_metric": "fit/pixel_f1"}))
        history = []
        for epoch, precision in enumerate(precisions):
            metric = package_epoch_metrics(SimpleNamespace(config=SimpleNamespace(quality_metric="pixel")),
                                            _values(0.8 if epoch < 2 else 0.7, precision), epoch, 0.2, 1e-5, 1)
            api.log_training_epoch(run, metric)
            history.append(metric)
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes("Проверка".encode("utf-8"))
        api.log_training_artifacts(run, TrainResult(history=history, epochs_total=2, training_time_sec=3,
                                                    best_checkpoint_path=str(checkpoint), best_threshold=0.9))
        api.end_run(run, MLflowRunStatus.FINISHED)
        best = api.get_usable_training_checkpoint(uri, run.run_id)
        assert (best.epoch, best.f1_score, best.threshold) == (selected, 0.8, 0.9)
        metrics = client.get_run(run.run_id).data.metrics
        assert not any(name.startswith("val/") for name in metrics)
        assert [item.step for item in client.get_metric_history(run.run_id, "train/loss")] == [1, 2]
        assert [item.step for item in client.get_metric_history(run.run_id, "fit/pixel_f1")] == [0, 1, 2]
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)
