"""Автобаланс фона: квоты, отсутствие утечки и сквозное обучение."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from mlsystem2.tile_preparation._adaptive_sampling import AdaptiveBackgroundSampler
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import AdaptiveSamplingController, SamplingFeedback, TilePreparationError, TileSceneSource, TileSplitRequest
from mlsystem2.train._sampling_feedback import BackgroundErrors, adaptive_controller
from mlsystem2.training_ui_api._templates import NEXT_GEN3_DEFAULT_CONFIG, initial_templates, sanitize_template_config
from mlsystem2.training_ui_api import _service, _pseudo_runner


def tiles(positive=20, background=2000):
    return [
        {"positive": i < positive, "valid_pixels": 1024, "scene_id": str(i % 5),
         "group": str(i % 17), "hard_negative": i % 19 == 0}
        for i in range(positive + background)
    ]


def feedback(fp=100, tp=90, empty=100000):
    return SamplingFeedback(empty_tiles=100, false_positive_tiles=20 if fp else 0,
                            empty_pixels=empty, false_positive_pixels=fp,
                            positive_pixels=100, true_positive_pixels=tp)


def test_budget_repeats_rotation_and_reproducibility():
    a = AdaptiveBackgroundSampler(tiles(), seed=42)
    b = AdaptiveBackgroundSampler(tiles(), seed=42)
    assert isinstance(a, AdaptiveSamplingController)
    assert len(a) == 276 < 2020
    previous = None
    for epoch in range(1, 7):
        a.set_epoch(epoch)
        b.set_epoch(epoch)
        draws = list(a)
        assert draws == list(b) == list(a)
        assert len(draws) == len(a)
        counts = Counter(i for i, _, _ in draws)
        assert all(1 <= counts[i] <= 4 for i in range(20))
        assert all(v == 1 for i, v in counts.items() if i >= 20)
        assert {e for _, e, _ in draws} == {epoch}
        assert len({d for _, _, d in draws}) == len(draws)
        selected = {i for i, _, _ in draws if i >= 20}
        if previous is not None:
            assert selected != previous
        previous = selected
        a.update_feedback(feedback())
        b.update_feedback(feedback())
    assert a.snapshot(include_state=True) == b.snapshot(include_state=True)
    assert a.snapshot()["current"]["positive_repeats_max_observed"] <= 4


def test_feedback_warmup_fp_growth_recall_guard_and_missing_data():
    sampler = AdaptiveBackgroundSampler(tiles(100, 10000), seed=42)
    for epoch in (1, 2):
        sampler.set_epoch(epoch)
        sampler.update_feedback(feedback(fp=0))
        assert sampler.snapshot()["current"]["next_background_draws"] == 500
    sampler.set_epoch(3)
    sampler.update_feedback(feedback(fp=1000))
    current = sampler.snapshot()["current"]
    assert current["background_draws"] == 500
    assert current["next_background_draws"] == 545
    sampler.set_epoch(4)
    sampler.update_feedback(feedback(fp=1000, tp=0))
    assert sampler.snapshot()["current"]["next_background_draws"] == 500
    assert "защита полноты" in sampler.snapshot()["current"]["reason"]
    sampler.set_epoch(5)
    sampler.update_feedback(SamplingFeedback())
    assert sampler.snapshot()["current"]["next_background_draws"] == 500
    assert "недостаточно" in sampler.snapshot()["current"]["reason"]


def test_all_foreground_warmup_does_not_become_recall_reference():
    sampler = AdaptiveBackgroundSampler(tiles(100, 10000), seed=42)
    for epoch, fp, tp in ((1, 50000, 99), (2, 30000, 80), (3, 20000, 40)):
        sampler.set_epoch(epoch)
        sampler.update_feedback(feedback(fp=fp, tp=tp))
    current = sampler.snapshot()["current"]
    assert current["next_background_draws"] == 545
    assert "ложные срабатывания" in current["reason"]
    assert sampler.snapshot(include_state=True)["state"]["best_recall"] is None


def test_small_pools_no_positives_nodata_and_partial_negatives():
    sampler = AdaptiveBackgroundSampler(tiles(20, 3), seed=42)
    sampler.set_epoch(1)
    assert {i for i, _, _ in sampler} == set(range(23))
    assert len(sampler) == 23
    with pytest.raises(TilePreparationError, match="нет ни одного"):
        AdaptiveBackgroundSampler(tiles(0, 100), seed=42)
    pool = tiles(2, 3)
    pool[0]["valid_pixels"] = 0
    pool[3]["partial_negative"] = True
    sampler = AdaptiveBackgroundSampler(pool, seed=42)
    sampler.set_epoch(1)
    assert 0 not in {i for i, _, _ in sampler}
    assert 3 in sampler.informative and 3 not in sampler.background
    assert adaptive_controller(SimpleNamespace(loader=SimpleNamespace(sampler=sampler))) is sampler


@pytest.mark.parametrize("positive", [1, 2, 8, 20, 100])
@pytest.mark.parametrize("background", [0, 1, 3, 20, 1000])
def test_extreme_pools_keep_foreground_background_and_repeat_bounds(positive, background):
    sampler = AdaptiveBackgroundSampler(tiles(positive, background), seed=42)
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        plan = Counter(i for i, _, _ in sampler)
        assert all(1 <= plan[i] <= 4 for i in range(positive))
        assert max((v for i, v in plan.items() if i >= positive), default=0) <= 1
        if background:
            assert any(i >= positive for i in plan)
        assert len(list(sampler)) <= positive + background
        sampler.update_feedback(feedback(fp=0))


def test_hard_mining_uses_train_indices_only_and_random_majority():
    sampler = AdaptiveBackgroundSampler(tiles(), seed=42)
    sampler.observe_train([100, 999999, 1], [0.9, 1.0, 0.8])
    assert sampler.snapshot(include_state=True)["state"]["train_scores"] == {100: 0.9}
    sampler.set_epoch(1)
    assert 100 in {i for i, _, _ in sampler}
    assert sampler.snapshot()["current"]["hard_background_draws"] <= sampler.snapshot()["current"]["background_draws"] * 0.25
    with pytest.raises(TilePreparationError):
        sampler.observe_train([100], [float("nan")])


def test_feedback_is_normalized_and_excludes_nodata_and_partial_labels():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(3, 2, 4, 4)
    logits[:, 1] = 10
    target = torch.zeros(3, 1, 4, 4)
    target[2] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1] = False
    errors = BackgroundErrors()
    assert errors.observe(torch, logits, target, valid) == [1, None, None]
    assert errors.feedback() == SamplingFeedback(empty_tiles=1, false_positive_tiles=1,
        empty_pixels=16, false_positive_pixels=16, positive_pixels=16, true_positive_pixels=16)
    partial = torch.ones(3, 1, 4, 4, dtype=torch.bool)
    errors = BackgroundErrors()
    errors.observe(torch, logits, target[:, 0].long(), valid[:, 0], partial)
    assert errors.feedback() == SamplingFeedback()


def test_masked_loss_has_zero_gradient_outside_valid_pixels():
    torch = pytest.importorskip("torch")
    from mlsystem2.train import _trainer
    from mlsystem2.train.contracts import TrainConfig
    config = TrainConfig(pipeline_variant="next_gen3", epochs=1, batch_size=2, device="cpu",
        learning_rate=1e-4, weight_decay=0.01, early_stopping_patience=3,
        loss="cross_entropy_tversky", class_weights=[0.5, 1.5])
    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    masks = torch.ones(2, 1, 4, 4)
    valid = torch.ones_like(masks, dtype=torch.bool)
    valid[..., :2] = False
    loss = _trainer._loss(torch, logits, masks, config, valid_pixels=valid)
    loss.backward()
    assert torch.isfinite(loss) and not logits.grad[..., :2].any()
    zero = _trainer._loss(torch, logits, masks, config, valid_pixels=torch.zeros_like(valid))
    assert zero.item() == 0


def scene(root):
    image, annotation = root / "Снимок.tif", root / "Снимок.geojson"
    pixels = np.random.default_rng(7).integers(1, 255, (4, 100, 100), dtype=np.uint8)
    pixels[:, :16, :16] = 0
    with rasterio.open(image, "w", driver="GTiff", width=100, height=100, count=4,
                       dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 100, 1, 1), nodata=0) as out:
        out.write(pixels)
        out.colorinterp = (rasterio.enums.ColorInterp.gray, *(rasterio.enums.ColorInterp.undefined,) * 3)
    annotation.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(44, 55, 45, 56))}]}), encoding="utf-8")
    return TileSceneSource(scene_id="Сцена", image_path=image, annotation_file=annotation)


def test_tiny_positive_mask_augmentation_draw_seed_and_full_heldout(tmp_path):
    source = scene(tmp_path)
    kwargs = dict(scenes=[source], tile_size=32, stride=16, seed=42,
                  augmentation_level=3, pipeline_variant="next_gen3")
    data = TileDataset(mode="train", **kwargs)
    assert any(t["positive"] for t in data.adaptive_sampling_tiles())
    assert min(data._notebook_positive_ratios[data._notebook_positive_ratios > 0]) < 0.001
    a, b = data[(6, 1, 0)], data[(6, 1, 0)]
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert any(not np.array_equal(data[(6, 2, d)][0], a[0]) for d in range(5))
    assert "valid_pixels" in a[2] and a[2]["sample_index"] == 6
    split = TileSplitRequest(val_fraction=0.2, test_fraction=0.2)
    datasets = {mode: TileDataset(mode=mode, tile_split=split, **kwargs) for mode in ("train", "val", "test")}
    manifest = datasets["train"].tile_split_manifest
    assert set(manifest["indices"]["train"]).isdisjoint(manifest["indices"]["val"])
    assert datasets["val"].adaptive_sampling_tiles() == datasets["test"].adaptive_sampling_tiles() == []
    assert len(datasets["val"]) == len(manifest["indices"]["val"])
    data.close()
    for dataset in datasets.values():
        dataset.close()


def test_parallel_preparation_keeps_counts_and_sampler_plan(tmp_path):
    source = scene(tmp_path)
    options = dict(scenes=[source], tile_size=32, stride=16, seed=42,
                   augmentation_level=0, pipeline_variant="next_gen3", mode="train")
    serial = TileDataset(**options, statistics_workers=0)
    parallel = TileDataset(**options, statistics_workers=2)
    assert serial.adaptive_sampling_tiles() == parallel.adaptive_sampling_tiles()
    assert serial.notebook_class_weights == parallel.notebook_class_weights
    a = AdaptiveBackgroundSampler(serial.adaptive_sampling_tiles(), seed=42)
    b = AdaptiveBackgroundSampler(parallel.adaptive_sampling_tiles(), seed=42)
    assert list(a) == list(b)
    serial.close()
    parallel.close()


def test_draw_augmentations_do_not_depend_on_worker_assignment(tmp_path):
    torch = pytest.importorskip("torch")
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch, _seed_tile_worker
    data = TileDataset(scenes=[scene(tmp_path)], tile_size=32, stride=16, seed=42,
                       augmentation_level=3, pipeline_variant="next_gen3", mode="train")
    plans = []
    for workers in (0, 2):
        sampler = AdaptiveBackgroundSampler(data.adaptive_sampling_tiles(), seed=42)
        sampler.set_epoch(1)
        loader = torch.utils.data.DataLoader(data, batch_size=4, sampler=sampler,
            num_workers=workers, collate_fn=_collate_tile_batch, worker_init_fn=_seed_tile_worker,
            **({"multiprocessing_context": "spawn"} if workers else {}))
        plans.append([(image, mask, meta["valid_pixels"]) for image, mask, meta in loader])
    for a, b in zip(*plans):
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    data.close()


def test_ui_profile_and_inference_keep_variant():
    for architecture in ("segformer_b0", "smp_segformer_b0", "smp_segformer_b2"):
        config = sanitize_template_config({"train.pipeline_variant": "next_gen3"}, architecture=architecture)
        assert config["train.pipeline_variant"] == "next_gen3"
        config.update({"dataset.task": "binary", "dataset.imagery_type": "kanopus", "train.input_channels": 4})
        _service._validate_training_pipeline_variant(config, architecture)
        if architecture.startswith("smp_"):
            template = next(t for t in initial_templates() if t["architecture"] == architecture)
            assert "next_gen3" in template["config_schema"]["fields"][0]["options"]
            assert template["default_config"]["train.pipeline_variant"] != "next_gen3"
    loaded = SimpleNamespace(model=SimpleNamespace(spec=SimpleNamespace(parameters={"pipeline_variant": "next_gen3"})),
                             artifact=SimpleNamespace(metadata={"sample_size": 512, "pipeline_variant": "next_gen3"}))
    config = _pseudo_runner._native_inference_config(loaded, {"tile_size": 512})
    assert config["pipeline_variant"] == "next_gen3"
    assert config["threshold"] == 0.9


def test_public_contracts_and_mlflow_sampling_artifacts(monkeypatch):
    from mlsystem2.mlflow_adapter import _client
    from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
    from mlsystem2.train.contracts import EpochMetrics
    sampler = AdaptiveBackgroundSampler(tiles(), seed=42)
    sampler.set_epoch(1)
    sampler.update_feedback(feedback())
    recorded, artifacts = [], []
    monkeypatch.setattr(_client, "_ensure_run_active", lambda run: SimpleNamespace(log_metric=lambda *a, **k: recorded.append(a), set_tag=lambda *a: None))
    monkeypatch.setattr(_client, "_log_dict", lambda data, path: artifacts.append((path, deepcopy(data))))
    metrics = EpochMetrics(epoch=1, train_loss=0.2, val_loss=0.3, epoch_time_sec=1,
        adaptive_sampling=sampler.snapshot())
    _client.log_training_epoch(MLflowRunRef(run_id="проверка", experiment_name="Проверка", tracking_uri="file:///unused", active=True), metrics)
    assert any(a[0] == "sampling/background_fraction" for a in recorded)
    assert artifacts[0][0] == "reports/sampling/epoch_0001.json"
    assert Path("frontend/public/news/next-gen3-adaptive-background.svg").is_file()
    assert NEXT_GEN3_DEFAULT_CONFIG["train.pipeline_variant"] == "next_gen3"


@pytest.mark.parametrize("architecture", ["smp_segformer_b0", "segformer_b0"])
def test_four_epochs_checkpoint_offline_and_onnx(tmp_path, monkeypatch, architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("transformers")
    onnx = pytest.importorskip("onnx")
    from mlsystem2.models.api import create_model, load_checkpoint
    from mlsystem2.models.contracts import ModelSpec, LoadCheckpointRequest
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
    from mlsystem2.train.api import train_model
    from mlsystem2.train.contracts import TrainConfig, TrainRequest
    from mlsystem2.training_ui_api._model_export import _export_segmentation_mask_onnx
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = scene(tmp_path)
    datasets = {mode: TileDataset(scenes=[source], tile_size=32, stride=16, mode=mode,
        seed=42, augmentation_level=0, pipeline_variant="next_gen3",
        tile_split=TileSplitRequest(val_fraction=0.2, test_fraction=0.2)) for mode in ("train", "val", "test")}
    sampler = AdaptiveBackgroundSampler(datasets["train"].adaptive_sampling_tiles(), seed=42)
    loaders = {mode: torch.utils.data.DataLoader(data, batch_size=4,
        sampler=sampler if mode == "train" else None, collate_fn=_collate_tile_batch) for mode, data in datasets.items()}
    spec = ModelSpec(name=architecture, input_channels=4, output_channels=1,
        parameters={"pipeline_variant": "next_gen3", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6},
                    "sampler": sampler.snapshot()["policy"]})
    handle = create_model(spec)
    result = train_model(TrainRequest(model=handle, train_loader=loaders["train"], val_loader=loaders["val"],
        test_loader=loaders["test"], checkpoint_dir=str(tmp_path / "checkpoints"), sample_size=32,
        config=TrainConfig(pipeline_variant="next_gen3", epochs=4, batch_size=4, device="cpu",
                           learning_rate=1e-4, weight_decay=0.01, early_stopping_patience=10,
                           loss="cross_entropy_tversky", class_weights=datasets["train"].notebook_class_weights)))
    assert result.epochs_total == 4
    assert len(result.diagnostics["adaptive_sampling"]["history"]) == 4
    assert result.diagnostics["test_metrics"]["checkpoint_epoch"] >= 1
    payload = torch.load(result.final_checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["metadata"]["pipeline_variant"] == "next_gen3"
    assert payload["metadata"]["adaptive_sampling"]["state"]["epoch"] == 4
    assert payload["metadata"]["adaptive_sampling"]["policy"]["version"] == "adaptive_background_v1"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=result.final_checkpoint_path, map_location="cpu"))
    handle.model.eval()
    loaded.model.model.eval()
    inputs = torch.rand(2, 4, 32, 32) * 255
    with torch.no_grad():
        torch.testing.assert_close(handle.model(inputs), loaded.model.model(inputs), rtol=0, atol=0)
    output = tmp_path / "model.onnx"
    _export_segmentation_mask_onnx(model=loaded.model.model, input_channels=4, output_channels=1, sample_size=32,
                             threshold=0.9, onnx_path=output, probability_output=True)
    onnx.checker.check_model(onnx.load(str(output)))
    for data in datasets.values():
        data.close()


def test_multiclass_training_keeps_partial_negatives_out_of_background_mining(tmp_path):
    torch = pytest.importorskip("torch")
    from mlsystem2.models.api import create_model
    from mlsystem2.models.contracts import ModelSpec
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
    from mlsystem2.train.api import train_model
    from mlsystem2.train.contracts import TrainConfig, TrainRequest
    torch.set_num_threads(2)
    descriptors = tiles(2, 2)
    descriptors[3]["partial_negative"] = True
    sampler = AdaptiveBackgroundSampler(descriptors, seed=42)
    images = torch.rand(4, 3, 32, 32) * 255

    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            index = index[0] if isinstance(index, tuple) else index
            mask = torch.zeros(32, 32, dtype=torch.long)
            negatives = torch.zeros(2, 32, 32, dtype=torch.bool)
            if index < 2:
                mask[8:24, 8:24] = index + 1
            if index == 3:
                negatives[0, :16] = True
            return images[index], mask, {"sample_index": index,
                "valid_pixels": torch.ones(32, 32, dtype=torch.bool),
                "class_hard_negative_masks": negatives}

    train = torch.utils.data.DataLoader(Samples(), batch_size=2, sampler=sampler, collate_fn=_collate_tile_batch)
    val = torch.utils.data.DataLoader(Samples(), batch_size=2, collate_fn=_collate_tile_batch)
    model = create_model(ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=3,
        parameters={"pipeline_variant": "next_gen3", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}}))
    config = TrainConfig(pipeline_variant="next_gen3", task="multiclass", epochs=2, batch_size=2,
        device="cpu", learning_rate=1e-4, weight_decay=0.01, early_stopping_patience=5,
        loss="cross_entropy_tversky", class_weights=[0.5, 1, 1.5], class_schema=[
            {"id": i, "slug": f"class_{i}", "name": f"Класс {i}", "color": "#00aa66"} for i in (1, 2)])
    result = train_model(TrainRequest(model=model, train_loader=train, val_loader=val, test_loader=val,
        config=config, checkpoint_dir=str(tmp_path), sample_size=32))
    assert result.epochs_total == 2
    assert {m["id"] for m in result.history[-1].val_per_class_metrics} == {1, 2}
    assert len(result.diagnostics["adaptive_sampling"]["history"]) == 2
    assert set(result.diagnostics["adaptive_sampling"]["state"]["train_scores"]) <= {2}


@pytest.mark.parametrize("imagery,channels", [("kanopus", 4), ("ortho", 3)])
def test_manual_job_and_worker_yaml_keep_next_gen3_without_switching_templates(tmp_path, monkeypatch, imagery, channels):
    from sqlalchemy import select
    from mlsystem2.settings.contracts import SystemSettings
    from mlsystem2.training_ui_api._config import get_config
    from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
    from mlsystem2.training_ui_api._models import JobRow, TrainingResultRow, TrainingTemplateRow
    from mlsystem2.training_ui_api import _worker
    from mlsystem2.training_ui_api.contracts import ImageryType, TrainingJobCreate
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    config = get_config()
    dataset = SimpleNamespace(quality_metric="pixel", task="binary", input_channels=channels,
        imagery_type=ImageryType(imagery), object_types=[], images_dir=str(tmp_path / imagery),
        version="проверочная-ревизия", name="Озера/main")
    monkeypatch.setattr(_service, "_resolve_dataset_name", lambda *args: dataset)
    monkeypatch.setattr(_worker, "_dataset_config", lambda *args: {
        "images_dir": dataset.images_dir, "annotations_dir": str(tmp_path / "annotations")})
    configure_schema(None)
    factory = create_session_factory(config)
    Base.metadata.create_all(factory.kw["bind"])
    with factory() as session:
        job = _service.create_training_job(session, TrainingJobCreate(dataset_key="Озера/main",
            architecture="smp_segformer_b0", mlflow_experiment_id="1", mlflow_experiment_name="Проверка next-gen3",
            config={"train.pipeline_variant": "next_gen3", "train.pretrained": False}), config)
        assert job.pipeline_variant == "next_gen3"
        assert job.config["train.quality_metric"] == "pixel"
        stored = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        assert stored.dataset_version == "проверочная-ревизия"
        payload = _worker._build_training_config(session, session.get(JobRow, job.id), config, tmp_path / "run")
        # Worker создаёт overlay; эти общие поля CLI получает из основного settings.yml.
        payload["mlflow"] = {"enabled": False, "tracking_uri": "file:///unused", **payload["mlflow"]}
        settings = SystemSettings.model_validate(payload)
        assert settings.train.pipeline_variant == "next_gen3"
        assert settings.train.input_channels == channels
        assert settings.train.max_val_batches_per_epoch is None
        assert all(t.default_config["train.pipeline_variant"] == "legacy" for t in session.scalars(select(TrainingTemplateRow)))
