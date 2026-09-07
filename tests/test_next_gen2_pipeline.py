"""Проверки переноса исходного ноутбука без изменения legacy."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape

from mlsystem2.mlflow_adapter import _client
from mlsystem2.models import _factory
from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec, SaveCheckpointRequest
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.settings.api import load_settings
from mlsystem2.tile_preparation import _dataloader
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation._windows import build_tile_windows
from mlsystem2.tile_preparation.contracts import TileSceneSource, TileSplitRequest
from mlsystem2.train import _trainer
from mlsystem2.train.contracts import TrainConfig, TrainRequest
from mlsystem2.train_pipeline import _runner
from mlsystem2.train_pipeline.contracts import TrainPipelineRequest
from mlsystem2.training_ui_api import _service
from mlsystem2.training_ui_api._templates import (
    CONFIG_SCHEMA, NEXT_GEN2_DEFAULT_CONFIG, initial_templates, sanitize_template_config,
)


SOURCE = Path(__file__).parent / "fixtures" / "next_gen2_train.ipynb"


def _notebook_functions():
    torch = pytest.importorskip("torch")
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    tree = ast.parse("\n".join("".join(cell["source"]) for cell in notebook["cells"]))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]

    def read_geometry(path):
        features = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
        return SimpleNamespace(empty=not features, geometry=[shape(f["geometry"]) for f in features])

    namespace = {
        "np": np, "rasterio": rasterio, "rasterize": rasterize,
        "windows": rasterio.windows, "torch": torch,
        "Dataset": torch.utils.data.Dataset,
        "WeightedRandomSampler": torch.utils.data.WeightedRandomSampler,
        "tqdm": lambda values: values, "gpd": SimpleNamespace(read_file=read_geometry),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _scene(tmp_path: Path):
    image = tmp_path / "SCN01.tif"
    annotation = tmp_path / "annotation.geojson"
    pixels = np.random.default_rng(7).integers(1, 255, (4, 100, 100), dtype=np.uint8)
    pixels[:, :32, :32] = 0
    pixels[2] = 70
    with rasterio.open(
        image, "w", driver="GTiff", width=100, height=100, count=4,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 100, 1, 1), nodata=0,
    ) as target:
        target.write(pixels)
        target.descriptions = ("RED", "GRN", "BLU", "NIR")
        valid = np.full((100, 100), 255, dtype=np.uint8)
        valid[:32, :32] = 0
        target.write_mask(valid)
    annotation.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(0, 50, 45, 100))}],
    }), encoding="utf-8")
    return image, annotation


def _dataset(image, annotation, mode="train"):
    return TileDataset(
        scenes=[TileSceneSource(scene_id="SCN01", image_path=image)],
        annotation_file=annotation, tile_size=32, stride=16, mode=mode, seed=42,
        augmentation_level=0, pipeline_variant="next_gen2",
        tile_split=TileSplitRequest(strategy="notebook_random", val_fraction=0.2, seed=42),
    )


def _config(**overrides):
    return TrainConfig(
        **{
            "pipeline_variant": "next_gen2", "epochs": 3, "batch_size": 2,
            "device": "cpu", "learning_rate": 1e-4, "weight_decay": 0.01,
            "loss": "cross_entropy", "early_stopping_patience": 9,
            "threshold_mode": "fixed", "class_weights": [0.3, 1.7],
            **overrides,
        }
    )


def _spec():
    return ModelSpec(
        name="segformer_b0", input_channels=4, output_channels=1, pretrained=False,
        parameters={"pipeline_variant": "next_gen2", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}},
    )


def test_source_notebook_is_fixed_and_defaults_match_its_settings():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
        "165157276ae777ef9e7538b7daead9d27cb8dc42c8bdf8225278b40defd5a5f4"
    )
    payload = sanitize_template_config({"train.pipeline_variant": "next_gen2"})
    assert payload.items() >= NEXT_GEN2_DEFAULT_CONFIG.items()
    assert CONFIG_SCHEMA["pipeline_defaults"]["next_gen2"] == NEXT_GEN2_DEFAULT_CONFIG
    payload.update({"dataset.task": "binary", "dataset.imagery_type": "kanopus", "train.input_channels": 4})
    _service._validate_training_pipeline_variant(payload, "segformer_b0")
    assert _service._job_pipeline_variant(SimpleNamespace(config=payload)) == "next_gen2"
    legacy = next(item for item in initial_templates() if item["architecture"] == "smp_segformer_b0")
    assert legacy["default_config"]["train.pipeline_variant"] == "legacy"


@pytest.mark.parametrize("key,value", [
    ("dataset.task", "multiclass"), ("dataset.imagery_type", "ortho"),
    ("tile_preparation.context", 128), ("tile_preparation.augmentation_level", 1),
    ("train.loss", "bce_dice"), ("train.max_train_batches_per_epoch", 1),
])
def test_api_rejects_incompatible_notebook_settings(key, value):
    payload = {**NEXT_GEN2_DEFAULT_CONFIG, "dataset.task": "binary",
               "dataset.imagery_type": "kanopus", "train.input_channels": 4, key: value}
    with pytest.raises(_service.TrainingUIAPIError, match="next-gen2"):
        _service._validate_training_pipeline_variant(payload, "segformer_b0")


def test_coordinates_split_masks_and_sampler_match_original_notebook(tmp_path):
    reference = _notebook_functions()
    train_test_split = pytest.importorskip("sklearn.model_selection").train_test_split
    torch = pytest.importorskip("torch")
    image, annotation = _scene(tmp_path)
    coords = reference["build_all_patch_coords"]([(str(image), str(annotation))], 32, 16)
    train, temporary = train_test_split(coords, test_size=0.4, random_state=42)
    validation, test = train_test_split(temporary, test_size=0.5, random_state=42)
    actual = _dataset(image, annotation)
    actual_val = _dataset(image, annotation, "val")
    assert [(w.window.x, w.window.y) for w in actual._windows] == [(c[2], c[3]) for c in train]
    assert [(w.window.x, w.window.y) for w in actual_val._windows] == [(c[2], c[3]) for c in validation]
    manifest = actual.tile_split_manifest
    assert [manifest["windows"][i][1:] for i in manifest["indices"]["test"]] == [[c[2], c[3]] for c in test]
    assert set(manifest["indices"]["train"]).isdisjoint(manifest["indices"]["test"])
    assert len(coords) == 25
    assert len(build_tile_windows(100, 100, 32, 16)) == 49
    assert actual.black_filtered_window_count == 0
    original = reference["OnTheFlyDataset"](train, 32, return_positive_info=True)
    for i in range(len(actual)):
        raw, target, metadata = actual[i]
        expected_image, expected_mask, _ = original[i]
        np.testing.assert_array_equal(reference["normalize_image"](raw.copy()), expected_image.numpy())
        np.testing.assert_array_equal(target[0], expected_mask.numpy())
        assert "valid_pixels" not in metadata
    weights, sampler = reference["compute_class_weights_and_sampler"](original)
    np.testing.assert_array_equal(actual.notebook_class_weights, weights.numpy())
    np.testing.assert_array_equal(actual.sampling_weights(), sampler.weights.numpy())
    torch.manual_seed(77)
    expected_draws = list(sampler)
    torch.manual_seed(77)
    actual_draws = list(torch.utils.data.WeightedRandomSampler(actual.sampling_weights(), len(actual), True))
    assert actual_draws == expected_draws
    actual.close()
    actual_val.close()


def test_parallel_loading_preserves_batches_and_rng_across_epochs(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    image, annotation = _scene(tmp_path)
    dataset = _dataset(image, annotation)
    settings = SimpleNamespace(tile_preparation=SimpleNamespace(
        tile_size=32, stride=16, context=0, seed=42, augmentation_level=0,
        positive_factor=0.5, hard_negative_factor=0.0, background_factor=0.5,
        class_balance=False, num_workers=0, prefetch_epochs=1000,
    ))
    monkeypatch.setattr(_dataloader, "get_settings", lambda: settings)
    monkeypatch.setattr(_dataloader, "TileDataset", lambda **kwargs: dataset)
    request = SimpleNamespace(
        scenes=dataset._scenes, annotation_file=annotation, hard_negative_annotation_file=None,
        class_annotations=[], classes=[], mode="train", tile_split=dataset._tile_split,
        include_object_instances=False, pipeline_variant="next_gen2", collect_band_histogram=False,
        batch_size=4,
    )
    observed = []
    for workers in (0, 2):
        dataset.close()
        settings.tile_preparation.num_workers = workers
        loader = _dataloader.create_tile_dataloader(request)
        assert loader.num_workers == workers
        assert not loader.persistent_workers
        if workers:
            assert loader.prefetch_factor == 2
        torch.manual_seed(1729)
        epochs = []
        for _ in range(2):
            batches = []
            for images, masks, meta in loader:
                batches.append((images.clone(), masks.clone(), torch.rand(3), meta))
            epochs.append((batches, torch.get_rng_state().clone()))
        observed.append(epochs)
    for serial, parallel in zip(*observed):
        assert torch.equal(serial[1], parallel[1])
        for expected, actual in zip(serial[0], parallel[0], strict=True):
            for index in range(3):
                assert torch.equal(expected[index], actual[index])
            assert expected[3] == actual[3]
    dataset.close()


def test_one_epoch_matches_notebook_loss_gradients_and_adamw(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = create_model(_spec())
    original = deepcopy(model.model.model)
    config = _config()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
    reference_optimizer = torch.optim.AdamW(original.parameters(), lr=config.learning_rate)
    reference = _notebook_functions()
    batches = [
        (torch.randint(0, 255, (count, 4, 32, 32)).float(), torch.randint(0, 2, (count, 1, 32, 32)).float())
        for count in (2, 1)
    ]
    total = 0.0
    torch.manual_seed(999)
    original.train()
    for images, masks in batches:
        normalized = torch.from_numpy(np.stack([reference["normalize_image"](item.numpy().copy()) for item in images]))
        reference_optimizer.zero_grad()
        logits = torch.nn.functional.interpolate(original(normalized).logits, size=(32, 32), mode="bilinear", align_corners=False)
        loss = torch.nn.functional.cross_entropy(logits, masks[:, 0].long(), weight=torch.tensor(config.class_weights))
        loss.backward()
        reference_optimizer.step()
        total += loss.item() * len(images)
    torch.manual_seed(999)
    actual = _trainer._train_epoch(torch, model.model, batches, optimizer, torch.device("cpu"), config, 1)
    assert actual["loss"] == pytest.approx(total / 3, abs=2e-6)
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(dict(model.model.model.named_parameters())[name], parameter, atol=3e-6, rtol=3e-5)


def test_two_class_checkpoint_roundtrip_is_offline_and_keeps_preprocessing(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    model = create_model(_spec())
    model.model.eval()
    images = torch.randint(0, 255, (2, 4, 32, 32)).float()
    images[:, 2] = 12
    expected = model.model(images)
    path = tmp_path / "best.pt"
    save_checkpoint(SaveCheckpointRequest(model=model, checkpoint_uri=str(path)))
    def no_network(*args, **kwargs):
        pytest.fail("Загрузка сохранённого чекпойнта обратилась к сети")
    monkeypatch.setattr(transformers.SegformerForSemanticSegmentation, "from_pretrained", no_network)
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))
    loaded.model.model.eval()
    torch.testing.assert_close(loaded.model.model(images), expected)
    assert loaded.model.model.model.config.num_labels == 2
    assert loaded.model.spec.output_channels == 1
    assert loaded.model.spec.parameters["preprocessing"]["mode"] == "window_minmax"


def test_mlflow_returns_f1_at_minimum_validation_loss(monkeypatch):
    histories = {
        "val/quality_f1": [SimpleNamespace(value=value, step=i) for i, value in enumerate([0.9, 0.4, 0.95], 1)],
        "val/loss": [SimpleNamespace(value=value, step=i) for i, value in enumerate([0.7, 0.3, 0.5], 1)],
        "val/best_threshold": [SimpleNamespace(value=0.5, step=i) for i in range(1, 4)],
    }
    client = SimpleNamespace(
        get_run=lambda _: SimpleNamespace(data=SimpleNamespace(tags={"pipeline_variant": "next_gen2", "quality_metric": "pixel"}), info=SimpleNamespace(artifact_uri="file:///artifacts")),
        get_metric_history=lambda _, name: histories[name],
    )
    monkeypatch.setattr(_client, "_mlflow", lambda: SimpleNamespace(set_tracking_uri=lambda _: None, tracking=SimpleNamespace(MlflowClient=lambda: client)))
    result = _client.get_best_training_checkpoint("local", "run")
    assert result.epoch == 2
    assert result.f1_score == 0.4
    assert result.threshold == 0.5


def test_complete_pipeline_uses_notebook_loaders_and_saves_native_artifacts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    image, annotation = _scene(tmp_path)
    scenes = tmp_path / "scenes.txt"
    scenes.write_text(image.name + "\n", encoding="utf-8")
    settings_path = tmp_path / "settings.yaml"
    train = {key.split(".", 1)[1]: value for key, value in NEXT_GEN2_DEFAULT_CONFIG.items() if key.startswith("train.")}
    train.update({"model_name": "segformer_b0", "input_channels": 4, "output_channels": 1,
                  "epochs": 2, "batch_size": 8, "device": "cpu"})
    settings = SystemSettings.model_validate({
        "runtime": {"project_root": str(tmp_path), "scratch_root": str(tmp_path / "scratch"),
                    "logs_root": str(tmp_path / "logs"), "cleanup_scratch_after_mlflow_log": False},
        "dataset": {"images_dir": str(tmp_path), "scenes_file": str(scenes),
                    "annotation_file": str(annotation), "val_fraction": 0.2},
        "tile_preparation": {"tile_size": 32, "stride": 16, "context": 0, "num_workers": 0},
        "train": train,
        "mlflow": {"enabled": False, "tracking_uri": "file:///unused", "experiment_name": "Проверка next-gen2"},
    })
    settings_path.write_text(yaml.safe_dump(settings.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
    load_settings(settings_path)
    pretrained_calls = []
    def local_pretrained(*args, **kwargs):
        pretrained_calls.append((args, kwargs))
        return transformers.SegformerForSemanticSegmentation(transformers.SegformerConfig(num_labels=2))
    monkeypatch.setattr(transformers.SegformerForSemanticSegmentation, "from_pretrained", local_pretrained)
    logged = []
    requests = []
    def create_loader(request):
        requests.append(request)
        return _dataloader.create_tile_dataloader(request)
    deps = replace(_runner._default_dependencies(), create_tile_dataloader=create_loader,
                   log_training_artifacts=lambda run, result: logged.append(result))
    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)
    assert result.status.value == "succeeded", result.report.errors
    assert [r.pipeline_variant for r in requests] == ["next_gen2", "next_gen2"]
    assert all(r.tile_split.strategy == "notebook_random" for r in requests)
    assert pretrained_calls[0][1]["num_labels"] == 2
    assert pretrained_calls[0][1]["use_safetensors"] is True
    trained = logged[0]
    assert trained.diagnostics["pipeline_variant"] == "next_gen2"
    assert len(trained.diagnostics["split_manifest"]["train"]["indices"]["test"]) == 5
    payload = torch.load(trained.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["pipeline_variant"] == "next_gen2"
    assert payload["metadata"]["val_loss"] == min(e.val_loss for e in trained.history)
    assert payload["metadata"]["confidence_threshold"] == 0.5
    assert Path(trained.final_checkpoint_path).is_file()


def test_notebook_input_adapter_retains_random_bias_and_both_heads():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.SegformerForSemanticSegmentation(transformers.SegformerConfig(num_labels=2))
    projection = _factory._first_patch_projection(model)
    projection.bias.data.fill_(123)
    rgb = projection.weight.detach().clone()
    classifier = model.decode_head.classifier
    _factory._adapt_pretrained_segformer(model, 4, 2, notebook=True)
    adapted = _factory._first_patch_projection(model)
    torch.testing.assert_close(adapted.weight[:, :3], rgb)
    torch.testing.assert_close(adapted.weight[:, 3], rgb[:, 0])
    assert not torch.any(adapted.bias == 123)
    assert model.decode_head.classifier is classifier


def test_scheduler_and_early_stopping_follow_loss_even_when_f1_grows(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = create_model(_spec())
    losses = [0.7, 0.2, 0.3, 0.4]
    scores = [0.6, 0.4, 0.7, 0.9]
    observed = []
    scheduler_type = torch.optim.lr_scheduler.ReduceLROnPlateau
    class ObservedScheduler(scheduler_type):
        def step(self, metrics, epoch=None):
            observed.append(float(metrics))
            return super().step(metrics, epoch)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", ObservedScheduler)
    monkeypatch.setattr(_trainer, "_train_epoch", lambda *args, **kwargs: {"loss": 0.4})
    def validation(*args, **kwargs):
        epoch = args[5]
        return {
            "loss": losses[epoch - 1], "best_threshold": 0.5, "best_pixel_threshold": 0.5,
            "best_threshold_pixel_f1": scores[epoch - 1], "best_threshold_pixel_precision": 0.5,
            "best_threshold_pixel_recall": 0.5, "best_threshold_precision": 0.5,
            "best_threshold_recall": 0.5, "quality_f1": scores[epoch - 1],
            "quality_precision": 0.5, "quality_recall": 0.5,
        }
    monkeypatch.setattr(_trainer, "_validate_epoch", validation)
    result = _trainer.train_model(TrainRequest(
        model=model, train_loader=[], val_loader=[], config=_config(epochs=10, early_stopping_patience=2),
        checkpoint_dir=str(tmp_path), sample_size=32,
    ))
    assert result.epochs_total == 4
    assert observed == losses
    payload = torch.load(result.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["epoch"] == 2
    assert result.diagnostics["checkpoint_selection"]["quality_f1"] == 0.4


def test_next_gen2_onnx_keeps_window_normalization_and_binary_output(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    onnx = pytest.importorskip("onnx")
    from mlsystem2.training_ui_api import _model_export
    torch.set_num_threads(2)
    model = create_model(_spec())
    path = tmp_path / "model.onnx"
    _model_export._export_binary_mask_onnx(
        model=model.model, input_channels=4, sample_size=32, threshold=0.5, onnx_path=path,
    )
    exported = onnx.load(str(path))
    onnx.checker.check_model(exported)
    operations = {node.op_type for node in exported.graph.node}
    assert {"ReduceMin", "ReduceMax", "Sigmoid"} <= operations
    assert exported.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 1
