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
from shapely.ops import unary_union

from mlsystem2.mlflow_adapter import _client
from mlsystem2.models import _factory
from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec, SaveCheckpointRequest
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.settings.api import load_settings
from mlsystem2.tile_preparation import _dataloader
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation._windows import build_tile_windows
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest, TileSceneSource, TileSplitRequest
from mlsystem2.train import _trainer
from mlsystem2.train.contracts import TrainConfig, TrainRequest
from mlsystem2.train_pipeline import _runner
from mlsystem2.train_pipeline.contracts import TrainPipelineRequest
from mlsystem2.training_ui_api import _pseudo_runner, _service, _test_samples
from mlsystem2.training_ui_api._templates import (
    CONFIG_SCHEMA, NEXT_GEN2_DEFAULT_CONFIG, initial_templates, sanitize_template_config,
)


SOURCE = Path(__file__).parent / "fixtures" / "next_gen2_train.ipynb"
EVAL_SOURCE = SOURCE.with_name("next_gen2_eval.ipynb")


def _eval_notebook_functions():
    torch = pytest.importorskip("torch")
    # .gitattributes сохраняет байты эталона также при checkout на Windows.
    assert hashlib.sha256(EVAL_SOURCE.read_bytes()).hexdigest() == (
        "0e9e2abac0a2390737419da01c7a274ed2f1eea8ccf204e6d26a272b07b0507b"
    )
    notebook = json.loads(EVAL_SOURCE.read_text(encoding="utf-8"))
    tree = ast.parse("\n".join("".join(cell["source"]) for cell in notebook["cells"]
                               if cell["cell_type"] == "code"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    threshold = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "theshold" for target in node.targets))
    namespace = {"np": np, "rasterio": rasterio, "torch": torch,
                 "Window": rasterio.windows.Window, "F": torch.nn.functional,
                 "DEVICE": "cpu", "theshold": threshold,
                 "tqdm": lambda values, **kwargs: values}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(EVAL_SOURCE), "exec"), namespace)
    return namespace


def _eval_model():
    torch = pytest.importorskip("torch")

    class LocalLogits(torch.nn.Module):
        def forward(self, images):
            local = torch.nn.functional.avg_pool2d(images[:, :1], 4)
            axis = torch.linspace(-1, 1, local.shape[-1]).reshape(1, 1, 1, -1)
            foreground = 4 - 6 * local + 4 * axis
            return SimpleNamespace(logits=torch.cat((torch.zeros_like(foreground), foreground), dim=1))

    return LocalLogits().eval()


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_inference_probabilities_and_mask_match_eval_notebook(tmp_path, batch_size):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    reference = _eval_notebook_functions()
    image, _ = _scene(tmp_path)
    # Маска TIFF не должна скрывать исходные пиксели от нормализации ноутбука.
    with rasterio.open(image, "r+") as dataset:
        valid = np.full((100, 100), 255, dtype=np.uint8)
        valid[32:64, 32:64] = 0
        dataset.write_mask(valid)
    model = _eval_model()
    expected_mask, expected_probabilities = reference["sliding_window_inference"](
        model, image, 32, 16, batch_size,
    )
    wrapped = _factory._wrap_next_gen(_spec(), model).model
    metrics = {}
    with rasterio.open(image) as dataset:
        actual_mask, actual_probabilities = _pseudo_runner._infer_notebook_scene_mask(
            dataset=dataset, input_indexes=(1, 2, 3, 4), input_channels=4,
            torch=torch, model=wrapped, tile_size=32, stride=16,
            batch_size=batch_size, threshold=reference["theshold"], device="cpu", metrics=metrics,
        )
    np.testing.assert_array_equal(actual_probabilities, expected_probabilities)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    assert metrics["tile_count"] == 25
    assert np.any(actual_mask[:32, :32])  # Нулевой тайл тоже проходит модель.
    assert not np.any(actual_mask[96:]) and not np.any(actual_mask[:, 96:])
    assert reference["theshold"] == 0.9
    legacy = _pseudo_runner._infer_test_tile_mask(
        torch=torch, model=wrapped, image_path=image, tile_size=32, stride=16,
        threshold=0.9, device="cpu", postprocess_profile=_pseudo_runner._POSTPROCESS_NONE,
    )
    assert np.any(actual_mask != legacy)  # Усреднение до порога существенно для результата.


def test_notebook_inference_handles_image_smaller_than_window(tmp_path):
    image, _ = _scene(tmp_path)
    reference = _eval_notebook_functions()
    expected_mask, expected_probabilities = reference["sliding_window_inference"](
        _eval_model(), image, 128, 64, 8,
    )
    with rasterio.open(image) as dataset:
        actual_mask, actual_probabilities = _pseudo_runner._infer_notebook_scene_mask(
            dataset=dataset, input_indexes=(1, 2, 3, 4), input_channels=4,
            torch=pytest.importorskip("torch"), model=None, tile_size=128, stride=64,
            batch_size=8, threshold=0.9, device="cpu",
        )
    np.testing.assert_array_equal(actual_probabilities, expected_probabilities)
    np.testing.assert_array_equal(actual_mask, expected_mask)


def test_pseudo_and_test_f1_use_eval_threshold_without_changing_checkpoint_metrics(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    image, _ = _scene(tmp_path)
    model = _eval_model()
    reference = _eval_notebook_functions()
    expected_mask, _ = reference["sliding_window_inference"](model, image, 32, 16, 8)
    checkpoint_metadata = {"pipeline_variant": "next_gen2", "confidence_threshold": 0.5,
                           "val_best_threshold": 0.5, "sample_size": 32, "inference_context": 0}
    loaded = SimpleNamespace(model=_factory._wrap_next_gen(_spec(), model),
                             artifact=SimpleNamespace(metadata=checkpoint_metadata))
    monkeypatch.setattr(_pseudo_runner, "load_checkpoint", lambda request: loaded)
    monkeypatch.setattr(_pseudo_runner, "_select_postprocess_profile", lambda count: _pseudo_runner._POSTPROCESS_DETAIL_V2)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    scenes = tmp_path / "scenes.txt"
    scenes.write_text(image.name, encoding="utf-8")
    config = {"run_root": str(tmp_path / "pseudo"), "checkpoint_uri": str(checkpoint),
              "images_root": str(tmp_path), "scenes_file": str(scenes),
              "output_geojson": str(tmp_path / "prediction.geojson"), "threshold": 0.5,
              "tile_size": 32, "stride": 16, "batch_size": 8, "device": "cpu",
              "class_key": "test", "class_name": "Проверка", "input_channels": 4}
    report = _pseudo_runner.run_pseudo_markup(config)
    assert report["status"] == "ok", report
    assert report["source"]["threshold"] == 0.9
    assert report["source"]["inference_merge"] == "gaussian_probabilities"
    assert report["source"]["batch_size"] == 32
    assert report["postprocess_profile"] == "none"
    features = json.loads(Path(config["output_geojson"]).read_text(encoding="utf-8"))["features"]
    with rasterio.open(image) as dataset:
        geometries = [(rasterio.warp.transform_geom("EPSG:4326", dataset.crs, item["geometry"]), 1)
                      for item in features]
        actual_mask = rasterize(geometries, out_shape=(100, 100), transform=dataset.transform, dtype="uint8")
        truth = tmp_path / "truth.tif"
        with rasterio.open(truth, "w", driver="GTiff", width=100, height=100, count=1,
                           dtype="uint8", crs=dataset.crs, transform=dataset.transform) as output:
            output.write(expected_mask, 1)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    report = _pseudo_runner.run_test_sample_f1({
        **config, "run_root": str(tmp_path / "test-f1"), "postprocess_profile": "detail_v2",
        "tiles": [{"index": 0, "image_path": str(image), "mask_path": str(truth)}],
    })
    assert report["status"] == "ok", report
    assert report["threshold"] == 0.9
    assert report["true_positive"] == int(expected_mask.sum())
    assert report["false_positive"] == report["false_negative"] == 0
    assert checkpoint_metadata["confidence_threshold"] == 0.5
    assert config["threshold"] == 0.5


def test_inference_revision_invalidates_only_next_gen2_test_metrics():
    source_job = SimpleNamespace(config={"train.pipeline_variant": "legacy"})
    session = SimpleNamespace(scalar=lambda query: None, get=lambda kind, key: source_job)
    result = SimpleNamespace(job_id="test-job")
    previous = _test_samples._effective_inference_template(session, "segformer_b0", "test", "none")[2]
    legacy = _test_samples._effective_inference_template(
        session, "segformer_b0", "test", "none", training_result=result,
    )[2]
    assert previous == legacy
    source_job.config["train.pipeline_variant"] = "next_gen2"
    updated = _test_samples._effective_inference_template(
        session, "segformer_b0", "test", "none", training_result=result,
    )[2]
    assert updated != previous


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


def _scene(tmp_path: Path, name="SCN01", x_offset=0):
    image = tmp_path / f"{name}.tif"
    annotation = tmp_path / f"{name}.geojson"
    pixels = np.random.default_rng(7).integers(1, 255, (4, 100, 100), dtype=np.uint8)
    pixels[:, :32, :32] = 0
    pixels[2] = 70
    with rasterio.open(
        image, "w", driver="GTiff", width=100, height=100, count=4,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(x_offset, 100, 1, 1), nodata=0,
    ) as target:
        target.write(pixels)
        target.descriptions = ("RED", "GRN", "BLU", "NIR")
        valid = np.full((100, 100), 255, dtype=np.uint8)
        valid[:32, :32] = 0
        target.write_mask(valid)
    annotation.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(x_offset, 50, x_offset + 45, 100))}],
    }), encoding="utf-8")
    return image, annotation


def _dataset(image, annotation, mode="train"):
    return TileDataset(
        scenes=[TileSceneSource(scene_id="SCN01", image_path=image)],
        annotation_file=annotation, tile_size=32, stride=16, mode=mode, seed=42,
        augmentation_level=0, pipeline_variant="next_gen2",
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


def test_source_notebook_is_fixed_and_ui_profile_is_compatible():
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


def test_coordinates_masks_and_sampler_match_original_notebook(tmp_path):
    reference = _notebook_functions()
    torch = pytest.importorskip("torch")
    image, annotation = _scene(tmp_path)
    coords = reference["build_all_patch_coords"]([(str(image), str(annotation))], 32, 16)
    actual = _dataset(image, annotation)
    assert [(w.window.x, w.window.y) for w in actual._windows] == [(c[2], c[3]) for c in coords]
    assert len(coords) == 25
    assert len(build_tile_windows(100, 100, 32, 16)) == 49
    assert actual.black_filtered_window_count == 0
    original = reference["OnTheFlyDataset"](coords, 32, return_positive_info=True)
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


def test_tile_split_excludes_validation_areas_across_scenes_and_ignores_their_labels(tmp_path):
    torch = pytest.importorskip("torch")
    identifiers = ["SCN01", "SCN02", "SCN03"]
    scenes = []
    for name, offset in zip(identifiers, (0, 40, 300), strict=True):
        image, annotation = _scene(tmp_path, name, offset)
        footprint = tmp_path / f"{name}_footprint.geojson"
        footprint.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": mapping(box(offset, 0, offset + 20, 100))},
        ]}), encoding="utf-8")
        scenes.append(TileSceneSource(scene_id=name, image_path=image, annotation_file=annotation, footprint_file=footprint))
    split = TileSplitRequest(strategy="window_random", val_fraction=0.2, spatial_purge=True, seed=42)
    def build(mode, sources=scenes):
        return TileDataset(scenes=sources, tile_size=32, stride=16, mode=mode, seed=42,
                           augmentation_level=0, pipeline_variant="next_gen2", tile_split=split)
    train, val = build("train"), build("val")
    try:
        manifest = train.tile_split_manifest
        assert manifest["indices"] == val.tile_split_manifest["indices"]
        subsets = [set(manifest["indices"][key]) for key in ("train", "val", "purged")]
        assert all(subsets[left].isdisjoint(subsets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        assert set.union(*subsets) == set(range(75))
        assert set(manifest["train_scene_ids"]) & set(manifest["validation_scene_ids"])
        validation_areas = []
        for item in val._windows:
            with rasterio.open(scenes[item.scene_index].image_path) as source:
                validation_areas.append(box(*source.window_bounds(rasterio.windows.Window(item.window.x, item.window.y, 32, 32))))
        validation_area = unary_union(validation_areas)
        for item in train._windows:
            with rasterio.open(scenes[item.scene_index].image_path) as source:
                rectangle = rasterio.windows.Window(item.window.x, item.window.y, 32, 32)
                assert box(*source.window_bounds(rectangle)).intersection(validation_area).area == 0
        cross_scene_purge = 0
        for index in manifest["indices"]["purged"]:
            scene_id, x, y = manifest["windows"][index]
            scene = next(scene for scene in scenes if scene.scene_id == scene_id)
            with rasterio.open(scene.image_path) as source:
                area = box(*source.window_bounds(rasterio.windows.Window(x, y, 32, 32)))
            assert area.intersection(validation_area).area > 0
            cross_scene_purge += any(area.intersection(other).area > 0 and item.scene_id != scene_id
                                     for other, item in zip(validation_areas, val._windows, strict=True))
        assert cross_scene_purge > 0
        assert manifest["geographic_overlap_after_purge"] == 0
        assert len(val) == 15
        assert sum(val[index][1].sum() for index in range(len(val))) > 0
        draws = torch.utils.data.WeightedRandomSampler(train.sampling_weights(), 1000, replacement=True)
        validation_keys = {(item.scene_id, item.window.x, item.window.y) for item in val._windows}
        assert all((train._windows[index].scene_id, train._windows[index].window.x, train._windows[index].window.y)
                   not in validation_keys for index in draws)
        weights, sampling = train.notebook_class_weights, train.sampling_weights()
        for scene in scenes:
            path = Path(scene.annotation_file)
            payload = json.loads(path.read_text(encoding="utf-8"))
            kept = []
            for feature in payload["features"]:
                geometry = shape(feature["geometry"]).difference(validation_area)
                if not geometry.is_empty:
                    kept.append({**feature, "geometry": mapping(geometry)})
            path.write_text(json.dumps({**payload, "features": kept}), encoding="utf-8")
        modified_val = build("val")
        try:
            assert sum(modified_val[index][1].sum() for index in range(len(modified_val))) == 0
        finally:
            modified_val.close()
        modified = build("train", list(reversed(scenes)))
        try:
            np.testing.assert_array_equal(modified.notebook_class_weights, weights)
            np.testing.assert_array_equal(sorted(modified.sampling_weights()), sorted(sampling))
            assert {tuple((w.scene_id, w.window.x, w.window.y)) for w in modified._windows} == {
                (w.scene_id, w.window.x, w.window.y) for w in train._windows
            }
        finally:
            modified.close()
    finally:
        train.close()
        val.close()


def test_tile_split_accepts_one_scene_and_keeps_touching_nonoverlapping_tiles(tmp_path):
    image, annotation = _scene(tmp_path)
    split = TileSplitRequest(strategy="window_random", val_fraction=0.2, spatial_purge=True)
    datasets = [TileDataset(scenes=[TileSceneSource(scene_id="SCN01", image_path=image)],
                           annotation_file=annotation, tile_size=32, stride=32, mode=mode,
                           pipeline_variant="next_gen2", tile_split=split, seed=42, augmentation_level=0)
                for mode in ("train", "val")]
    try:
        train, val = datasets
        assert len(train) == 7 and len(val) == 2
        assert train.tile_split_manifest["purged_window_count"] == 0
        assert train.tile_split_manifest["train_scene_ids"] == val.tile_split_manifest["validation_scene_ids"] == ["SCN01"]
    finally:
        for dataset in datasets:
            dataset.close()


@pytest.mark.parametrize("strategy,purge", [("window_random", False), ("scene_fold", True)])
def test_next_gen2_api_rejects_split_without_spatial_isolation(strategy, purge):
    with pytest.raises(ValueError, match="next-gen2"):
        TileDataloaderRequest(
            scenes=[TileSceneSource(scene_id="сцена", image_path="image.tif")],
            annotation_file="annotation.geojson", batch_size=32, mode="train", pipeline_variant="next_gen2",
            tile_split=TileSplitRequest(strategy=strategy, val_fraction=0.2, spatial_purge=purge),
        )


def test_validation_does_not_change_model_parameters_buffers_or_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = create_model(_spec()).model
    model.train()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    batch = (torch.rand(2, 4, 32, 32) * 255, torch.zeros(2, 1, 32, 32))
    _trainer._validate_epoch(torch, model, [batch], torch.device("cpu"), _config(), 1)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters())


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


@pytest.mark.parametrize("selection,epoch,f1", [
    (None, 2, 0.4), ("val_loss", 2, 0.4), ("quality_f1", 3, 0.95),
])
def test_mlflow_preserves_old_loss_selection_and_reads_new_f1_selection(monkeypatch, selection, epoch, f1):
    histories = {
        "val/quality_f1": [SimpleNamespace(value=value, step=i) for i, value in enumerate([0.9, 0.4, 0.95, 0.95], 1)],
        "val/loss": [SimpleNamespace(value=value, step=i) for i, value in enumerate([0.7, 0.3, 0.5, 0.5], 1)],
        "val/best_threshold": [SimpleNamespace(value=0.5, step=i) for i in range(1, 5)],
    }
    tags = {"pipeline_variant": "next_gen2", "quality_metric": "pixel"}
    if selection is not None:
        tags["checkpoint_selection_metric"] = selection
    client = SimpleNamespace(
        get_run=lambda _: SimpleNamespace(data=SimpleNamespace(tags=tags), info=SimpleNamespace(artifact_uri="file:///artifacts")),
        get_metric_history=lambda _, name: histories[name],
    )
    monkeypatch.setattr(_client, "_mlflow", lambda: SimpleNamespace(set_tracking_uri=lambda _: None, tracking=SimpleNamespace(MlflowClient=lambda: client)))
    result = _client.get_best_training_checkpoint("local", "run")
    assert result.epoch == epoch
    assert result.f1_score == f1
    assert result.threshold == 0.5


def test_complete_pipeline_uses_notebook_loaders_and_saves_native_artifacts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    scenes = [_scene(tmp_path, f"SCN{index:02d}", index * 200) for index in range(5)]
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for image, annotation in scenes:
        annotation.rename(annotations / f"{tmp_path.name}_{image.stem}.geojson")
    settings_path = tmp_path / "settings.yaml"
    train = {key.split(".", 1)[1]: value for key, value in NEXT_GEN2_DEFAULT_CONFIG.items() if key.startswith("train.")}
    train.update({"model_name": "segformer_b0", "input_channels": 4, "output_channels": 1,
                  "epochs": 2, "batch_size": 8, "device": "cpu"})
    settings = SystemSettings.model_validate({
        "runtime": {"project_root": str(tmp_path), "scratch_root": str(tmp_path / "scratch"),
                    "logs_root": str(tmp_path / "logs"), "cleanup_scratch_after_mlflow_log": False},
        "dataset": {"images_dir": str(tmp_path), "annotations_dir": str(annotations), "val_fraction": 0.2},
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
    assert all(r.tile_split.strategy == "window_random" and r.tile_split.spatial_purge for r in requests)
    assert _runner._mlflow_start_request(settings, TrainPipelineRequest()).tags["checkpoint_selection_metric"] == "quality_f1"
    assert pretrained_calls[0][1]["num_labels"] == 2
    assert pretrained_calls[0][1]["use_safetensors"] is True
    trained = logged[0]
    assert trained.diagnostics["pipeline_variant"] == "next_gen2"
    split = trained.diagnostics["split_manifest"]["train"]
    assert len(split["indices"]["val"]) == 25
    assert set(split["train_scene_ids"]) & set(split["validation_scene_ids"])
    assert split["geographic_overlap_after_purge"] == 0
    payload = torch.load(trained.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["pipeline_variant"] == "next_gen2"
    assert payload["metadata"]["val_quality_f1"] == max(e.val_quality_f1 for e in trained.history)
    assert payload["metadata"]["checkpoint_selection_metric"] == "quality_f1"
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


def test_early_stopping_follows_f1_while_scheduler_keeps_notebook_loss(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = create_model(_spec())
    losses = [0.7, 0.2, 0.3, 0.4, 0.5, 0.6]
    scores = [0.6, 0.4, 0.7, 0.9, 0.9, 0.8]
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
    assert result.epochs_total == 6
    assert observed == losses
    payload = torch.load(result.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["epoch"] == 4
    assert result.diagnostics["checkpoint_selection"]["quality_f1"] == 0.9
    assert result.diagnostics["checkpoint_selection"]["metric"] == "quality_f1"
    final = torch.load(result.final_checkpoint_path, map_location="cpu", weights_only=True)
    assert final["metadata"]["epoch"] == 6


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
