"""Сквозные инварианты обучения областей и границ."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, box, mapping

from mlsystem2.inference.api import create_object_scene, object_window_origins, separate_objects
from mlsystem2.inference.contracts import ObjectSceneRequest, ObjectWindowPrediction
from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import ModelSpec, LoadCheckpointRequest, SaveCheckpointRequest
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
from mlsystem2.tile_preparation._object_targets import boundary_targets
from mlsystem2.tile_preparation.contracts import TileSceneSource, TileSplitRequest
from mlsystem2.train._object_f1 import object_loss, validate_objects
from mlsystem2.train.contracts import TrainConfig


def _scenes(tmp_path, *, rgba=True):
    scenes = []
    for i in range(3):
        image, annotation = tmp_path / f"scene{i}.tif", tmp_path / f"scene{i}.geojson"
        x = i * 200
        data = np.full((4 if rgba else 3, 80, 90), 80, dtype=np.uint8)
        if rgba:
            data[3] = 255
            data[3, :4] = 0
        with rasterio.open(image, "w", driver="GTiff", width=90, height=80, count=data.shape[0], dtype="uint8",
                           crs="EPSG:3857", transform=from_origin(x, 80, 1, 1), nodata=None if rgba else 0) as ds:
            ds.write(data)
            if rgba:
                ds.colorinterp = tuple(rasterio.enums.ColorInterp[v] for v in ("red", "green", "blue", "alpha"))
        shapes = [box(x+10, 10, x+40, 65), MultiPolygon([box(x+40, 10, x+65, 65), box(x+70, 15, x+80, 30)])]
        annotation.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(g)} for g in shapes]}), encoding="utf-8")
        scenes.append(TileSceneSource(scene_id=f"scene{i}", image_path=image, annotation_file=annotation))
    return scenes


def _dataset(scenes, mode="train", augmentation=0):
    return TileDataset(scenes=scenes, tile_size=64, stride=32, mode=mode, seed=42, augmentation_level=augmentation,
        include_object_instances=True, pipeline_variant="object_f1", input_channels=3,
        tile_split=TileSplitRequest(strategy="scene_groups", val_fraction=0.2, test_fraction=0.2))


def _config():
    return TrainConfig(pipeline_variant="object_f1", task="binary", quality_metric="objects", class_weights=[1, 1],
                       epochs=2, batch_size=2, device="cpu", learning_rate=1e-4, weight_decay=.01,
                       loss="cross_entropy_tversky", early_stopping_patience=10)


def test_separation_touching_small_fallback_and_holes():
    area = np.zeros((40, 50), np.float32)
    area[5:35, 5:45] = .95
    area[12:16, 12:16] = 0
    boundary = np.zeros_like(area)
    boundary[:, 24:26] = .95
    labels = separate_objects(area, boundary, np.ones_like(area, bool))
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[20, 20] != labels[20, 30]
    assert labels[13, 13] == 0
    labels = separate_objects(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool))
    assert np.all(labels == 1)


def test_window_merge_keeps_objects_across_seams_and_covers_small_scene(tmp_path):
    size = 32
    area = np.zeros((47, 59), np.float32)
    area[3:45, 3:57] = .95
    boundary = np.zeros_like(area)
    boundary[:, 29:31] = .95
    accumulator = create_object_scene(ObjectSceneRequest(width=59, height=47, tile_size=size, work_dir=str(tmp_path)))
    try:
        for y in object_window_origins(47, size):
            for x in object_window_origins(59, size):
                accumulator.add_window(ObjectWindowPrediction(x=x, y=y, probabilities=np.stack([area[y:y+size, x:x+size], boundary[y:y+size, x:x+size]]), valid_pixels=np.ones((size, size), bool)))
        result = accumulator.finish()
        assert result.object_count == 2
        assert result.valid_pixels.all()
        np.testing.assert_array_equal(result.instances, separate_objects(area, boundary, np.ones_like(area, bool)))
    finally:
        accumulator.close()
    assert object_window_origins(12, size) == [0]
    assert not list(tmp_path.iterdir())


def test_targets_keep_internal_seam_but_ignore_crop_and_invalid():
    labels = np.ones((20, 20), np.int32)
    labels[:, 10:] = 2
    valid = np.ones_like(labels, bool)
    valid[10:12, 3:5] = False
    boundary, usable = boundary_targets(labels, valid, np.zeros_like(valid))
    assert boundary[5, 9] == boundary[5, 10] == 1
    assert boundary[5, 3] == 0
    assert not usable[:2].any() and not usable[9:13, 2:6].any()


def test_rgb_alpha_split_ids_and_augmentation_reproducible(tmp_path):
    scenes = _scenes(tmp_path)
    splits = [_dataset(scenes, mode) for mode in ("train", "val", "test")]
    try:
        memberships = [set(item.scene_id for item in dataset._windows) for dataset in splits]
        assert not memberships[0] & memberships[1] and not memberships[0] & memberships[2]
        ids = set()
        for dataset in splits:
            for image, mask, meta in dataset:
                assert image.shape == (3, 64, 64)
                assert np.array_equal(mask[0] > 0, meta["object_instances"] > 0)
                ids.update(np.unique(meta["object_instances"]))
        assert ids == {0, 1, 2, 3}
        one, two = _dataset(scenes, augmentation=3), _dataset(scenes, augmentation=3)
        try:
            for index in range(len(one)):
                a, b = one[index], two[index]
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[2]["object_instances"], b[2]["object_instances"])
                np.testing.assert_array_equal(a[1][0] > 0, a[2]["object_instances"] > 0)
        finally:
            one.close()
            two.close()
    finally:
        for dataset in splits:
            dataset.close()


def test_model_both_heads_gradients_checkpoint_and_scene_validation(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    dataset = _dataset(_scenes(tmp_path), "val")
    try:
        batch = _collate_tile_batch([dataset[0], dataset[1]])
        images, masks, meta = batch
        spec = ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=1, parameters={
            "pipeline_variant": "object_f1", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}})
        handle = create_model(spec)
        logits = handle.model(images)
        assert logits.shape == (2, 3, 64, 64)
        logits.retain_grad()
        loss, _, _ = object_loss(torch, logits, masks, meta["valid_pixels"][:, None], meta, _config())
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert all(logits.grad[:, i].abs().sum() > 0 for i in range(3))
        path = str(tmp_path / "model.pt")
        save_checkpoint(SaveCheckpointRequest(model=handle, checkpoint_uri=path))
        restored = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=path)).model.model
        restored.eval()
        handle.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(handle.model(images), restored(images))
        batches = [_collate_tile_batch([dataset[i]]) for i in range(len(dataset))]
        result = validate_objects(torch, restored, batches, "cpu", _config(), 1, None)
        assert np.isfinite(result["loss"]) and len(result["per_scene_metrics"]) == 1
        assert 0 <= result["quality_f1"] <= 1
    finally:
        dataset.close()


def test_onnx_two_probabilities_and_instance_pipeline(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    import yaml
    from mlsystem2.training_ui_api._model_export import _export_segmentation_mask_onnx, _pipeline_yaml, _triton_config

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x[:, :1], -x[:, :1], x[:, 1:2]), dim=1)

    path = tmp_path / "object.onnx"
    _export_segmentation_mask_onnx(model=Model(), input_channels=3, output_channels=1, sample_size=32,
        threshold=.5, onnx_path=path, probability_output=True, object_output=True)
    image = np.random.default_rng(42).normal(size=(1, 3, 32, 32)).astype(np.float32)
    actual = reference.ReferenceEvaluator(onnx.load(path)).run(["probabilities"], {"input": image})[0]
    logits = Model()(torch.from_numpy(image))
    expected = torch.cat((logits[:, :2].softmax(1)[:, 1:2], logits[:, 2:3].sigmoid()), 1).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    pipeline = yaml.safe_load(_pipeline_yaml("objects", 512, 3, probability_output=True, object_output=True,
        postprocess_config={"postprocess.min_area_m2": 10.0}))
    bricks = pipeline["config"]["bricks"]
    assert bricks[0]["_class"] == "ObjectF1Segmentation"
    assert bricks[1]["value_property_name"] == "instance_id"
    assert all(b["_class"] not in {"MultiThresholding", "MaskMorphology"} for b in bricks)
    assert "dims: [ -1, 2, -1, -1 ]" in _triton_config("objects", 3, foreground_channels=2, probability_output=True)


def test_training_two_epochs_uses_object_score_and_evaluates_best(tmp_path):
    torch = pytest.importorskip("torch")
    from mlsystem2.train.api import train_model
    from mlsystem2.train.contracts import TrainRequest
    from mlsystem2.train_pipeline._runner import _tile_request
    from mlsystem2.dataset_preparing.contracts import PreparedDataset, PreparedScene

    torch.set_num_threads(2)
    scenes = _scenes(tmp_path)
    prepared = PreparedDataset(format="per_image_binary", scenes=[PreparedScene(
        scene_id=s.scene_id, image_path=str(s.image_path), annotation_file=str(s.annotation_file)) for s in scenes])
    request = _tile_request(prepared, 2, "train", TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2),
                            include_object_instances=True, pipeline_variant="object_f1", input_channels=3)
    assert request.input_channels == 3
    datasets = [_dataset(scenes, mode) for mode in ("train", "val", "test")]
    try:
        loaders = [[_collate_tile_batch([dataset[i], dataset[min(i+1, len(dataset)-1)]])
                    for i in range(0, len(dataset), 2)] for dataset in datasets]
        handle = create_model(ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=1,
            parameters={"pipeline_variant": "object_f1", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}}))
        result = train_model(TrainRequest(model=handle, train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2],
                            config=_config(), checkpoint_dir=str(tmp_path / "checkpoints"), sample_size=64))
        assert result.epochs_total == 2
        best = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=result.best_checkpoint_path))
        assert best.artifact.metadata["checkpoint_selection_metric"] == "val/object_f1"
        assert "object_f1" in result.diagnostics["test_metrics"]
        assert all(item.train_boundary_loss is not None and item.val_region_loss is not None for item in result.history)
    finally:
        for dataset in datasets:
            dataset.close()


def test_scene_groups_use_valid_footprints_and_reject_leakage(tmp_path):
    from mlsystem2.tile_preparation._object_targets import scene_group_split
    from mlsystem2.tile_preparation.contracts import TilePreparationError

    scenes = _scenes(tmp_path)
    request = TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2)
    # Два снимка одной территории обязаны попасть в одну часть.
    duplicate = scenes[0].model_copy(update={"scene_id": "повтор"})
    split, manifest = scene_group_split(scenes + [duplicate], request)
    assert any({"scene0", "повтор"} <= set(ids) for ids in split.values())
    assert len(manifest["groups"]) == 3
    with pytest.raises(TilePreparationError, match="три независимые"):
        scene_group_split([scenes[0], duplicate, scenes[1]], request)


def test_alpha_cannot_be_used_as_nir(tmp_path):
    from mlsystem2.tile_preparation.contracts import TilePreparationError
    scenes = _scenes(tmp_path)
    with pytest.raises(TilePreparationError, match="alpha"):
        TileDataset(scenes=scenes, tile_size=64, stride=32, mode="val", seed=42, augmentation_level=0,
                    include_object_instances=True, pipeline_variant="object_f1", input_channels=4,
                    tile_split=TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2))
    # Настоящий четвёртый спектральный канал остаётся входом модели.
    for scene in scenes:
        with rasterio.open(scene.image_path, "r+") as source:
            source.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                                  rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.undefined)
    dataset = TileDataset(scenes=scenes, tile_size=64, stride=32, mode="val", seed=42, augmentation_level=0,
                         include_object_instances=True, pipeline_variant="object_f1", input_channels=4,
                         tile_split=TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2))
    try:
        assert dataset[0][0].shape == (4, 64, 64)
    finally:
        dataset.close()


def test_conflicting_annotations_have_stable_ids_and_ignore_boundary(tmp_path):
    scenes = _scenes(tmp_path)
    for scene in scenes:
        p = scene.annotation_file
        data = json.loads(p.read_text(encoding="utf-8"))
        data["features"].append(data["features"][0])
        p.write_text(json.dumps(data), encoding="utf-8")
    dataset = _dataset(scenes, "val")
    try:
        _, _, meta = dataset[0]
        assert meta["overlap_pixels"] > 0
        assert all(item["overlapping_polygon_pairs"] == 1 for item in dataset.scene_tile_diagnostics)
        assert 4 in meta["object_instances"] and 1 not in meta["object_instances"]
        assert not meta["boundary_valid"][meta["object_instances"] == 4].any()
        np.testing.assert_array_equal(meta["object_instances"], dataset[0][2]["object_instances"])
    finally:
        dataset.close()


def test_native_inference_preserves_instance_ids_and_small_scene(tmp_path):
    from mlsystem2.training_ui_api._object_inference import predict_instances
    image = tmp_path / "small.tif"
    with rasterio.open(image, "w", driver="GTiff", width=19, height=17, count=4, dtype="uint8",
                       crs="EPSG:3857", transform=from_origin(0, 17, 1, 1)) as out:
        data = np.full((4,17,19),255,np.uint8)
        data[3,0] = 0
        out.write(data)
        out.colorinterp = tuple(rasterio.enums.ColorInterp[v] for v in ("red","green","blue","alpha"))
    prediction = np.zeros((2,32,32), np.float32)
    prediction[0] = .9
    prediction[1,:,9:11] = .9
    with rasterio.open(image) as source:
        accumulator, result = predict_instances(source, lambda images: np.repeat(prediction[None],len(images),0),
                                                input_indexes=(1,2,3),tile_size=32)
        try:
            expected = separate_objects(prediction[0,:17,:19],prediction[1,:17,:19],source.dataset_mask()>0)
            np.testing.assert_array_equal(result.instances,expected)
            assert result.object_count == 2 and not result.instances[0].any()
        finally:
            accumulator.close()


def test_export_runtime_contains_same_cpu_implementation(tmp_path):
    from pathlib import Path
    import mlsystem2.inference._objects as shared
    from mlsystem2.training_ui_api._model_export import _write_object_runtime
    _write_object_runtime(tmp_path)
    assert (tmp_path / "runtime/mlsystem2/inference/_objects.py").read_bytes() == Path(shared.__file__).read_bytes()
    assert "register_notebook_bricks" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_masked_pixels_have_no_gradient_in_either_output():
    torch = pytest.importorskip("torch")
    logits = torch.randn((2,3,16,16),requires_grad=True)
    masks = torch.ones((2,1,16,16))
    valid = torch.ones_like(masks,dtype=torch.bool)
    valid[:,:,0:5]=False
    loss,_,_ = object_loss(torch,logits,masks,valid,{"boundary_target":masks[:,0],"boundary_valid":valid[:,0]},_config())
    loss.backward()
    assert not logits.grad[:,:,0:5].any()
    assert logits.grad[:,:,6:].abs().sum()>0
