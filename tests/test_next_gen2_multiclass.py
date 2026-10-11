"""Мультиклассовый next-gen2: данные, обучение, сохранение и оконный инференс."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from mlsystem2.models.api import create_model, load_checkpoint
from mlsystem2.models.contracts import ModelSpec, LoadCheckpointRequest
from mlsystem2.settings.api import load_settings
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import TileSceneSource, TileClassDefinition, TileSplitRequest, TileDataloaderRequest
from mlsystem2.train import _trainer
from mlsystem2.train.contracts import TrainConfig, TrainRequest
from mlsystem2.training_ui_api import _service, _pseudo_runner, _model_export
from mlsystem2.training_ui_api._geoalert_notebook import _multiclass_probability_masks
from mlsystem2.training_ui_api._templates import NEXT_GEN2_DEFAULT_CONFIG


SCHEMA = [{"id": i, "slug": f"type_{i}", "name": f"Тип {i}", "color": "#00ff66"} for i in range(1, 4)]


def _config(**overrides):
    return TrainConfig(**{
        "task": "multiclass", "pipeline_variant": "next_gen2", "class_schema": SCHEMA,
        "epochs": 1, "batch_size": 2, "device": "cpu", "learning_rate": 1e-4, "weight_decay": 0.01,
        "loss": "cross_entropy_tversky", "class_weights": [0.4, 0.6, 1.0, 2.0],
        "tversky_alpha": 0.75, "tversky_beta": 0.25, "early_stopping_patience": 3, "threshold_mode": "fixed",
        **overrides,
    })


@pytest.mark.parametrize("imagery,channels", [("ortho", 3), ("kanopus", 4)])
@pytest.mark.parametrize("architecture", [f"smp_segformer_b{i}" for i in range(4)])
def test_profile_accepts_multiclass_and_keeps_object_f1_binary(imagery, channels, architecture):
    from mlsystem2.training_ui_api._templates import sanitize_template_config

    payload = sanitize_template_config(NEXT_GEN2_DEFAULT_CONFIG, architecture=architecture)
    payload.update({"dataset.task": "multiclass", "dataset.imagery_type": imagery, "train.input_channels": channels})
    _service._validate_training_pipeline_variant(payload, architecture)
    with pytest.raises(_service.TrainingUIAPIError, match="только бинарную"):
        _service._validate_training_pipeline_variant({**payload, "train.pipeline_variant": "object_f1"}, architecture)
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root / "configs/settings.server.yaml", root / "configs/run.next-gen2.server.yaml").model_dump()
    settings["dataset"]["annotations_dir"] = "/dataset/annotations"
    settings["train"].update(task="multiclass", model_name=architecture, input_channels=channels, output_channels=4,
                              batch_size=payload["train.batch_size"])
    assert SystemSettings.model_validate(settings).train.output_channels == 4
    assert len(_config().class_weights) == 4
    with pytest.raises(ValueError, match="веса всех классов"):
        _config(class_weights=[1, 1])
    with pytest.raises(ValueError, match="бинарную"):
        _config(pipeline_variant="object_f1", quality_metric="objects")


def test_training_checkpoint_and_export_preserve_every_class(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("onnx")
    torch.set_num_threads(2)
    spec = ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=4, pretrained=False,
                     parameters={"pipeline_variant": "next_gen2", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}})
    model = create_model(spec)
    image = torch.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).float() % 251
    mask = torch.zeros((2, 32, 32), dtype=torch.long)
    for i in range(1, 4):
        mask[:, i * 6:(i + 1) * 6, 4:28] = i
    batches = [(image, mask)]
    result = _trainer.train_model(TrainRequest(model=model, train_loader=batches, val_loader=batches, test_loader=batches,
                                              config=_config(), checkpoint_dir=str(tmp_path / "weights"), sample_size=32))
    assert len(result.history) == 1
    assert {item["slug"] for item in result.history[0].val_per_class_metrics} == {item["slug"] for item in SCHEMA}
    assert result.history[0].val_best_threshold == 0.5
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=result.best_checkpoint_path, map_location="cpu"))
    assert loaded.model.spec.output_channels == 4
    assert loaded.artifact.metadata["task"] == "multiclass"
    assert loaded.artifact.metadata["class_schema"] == [{**item, "priority": 0} for item in SCHEMA]
    assert loaded.artifact.metadata["checkpoint_selection_metric"] == "val_loss"
    assert result.diagnostics["test_metrics"]["threshold"] == 0.5
    assert {item["slug"] for item in result.diagnostics["test_metrics"]["per_class_metrics"]} == {item["slug"] for item in SCHEMA}
    loaded.model.model.eval()
    with torch.no_grad():
        logits = loaded.model.model(image)
    assert logits.shape == (2, 4, 32, 32)
    path = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(model=loaded.model.model, input_channels=3, output_channels=4,
                                                sample_size=32, threshold=0.9, onnx_path=path, probability_output=True)
    import onnx
    output = onnx.load(path).graph.output[0]
    assert output.name == "probabilities" and output.type.tensor_type.shape.dim[1].dim_value == 4


def test_class_specific_negative_does_not_teach_other_classes_background():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    logits = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]], requires_grad=True)
    partial = torch.tensor([[[[False]], [[True]], [[False]]]])
    loss = _trainer._loss(torch, logits, torch.zeros((1, 1, 1), dtype=torch.long), _config(),
                          class_hard_negative_pixels=partial)
    expected = -torch.log(1 - torch.softmax(logits, dim=1)[:, 2]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2, 0, 0] > 0
    assert logits.grad[0, 3, 0, 0] < 0
    zero = _trainer._loss(torch, logits, torch.zeros((1, 1, 1), dtype=torch.long), _config(),
                          class_hard_negative_pixels=partial, valid_pixels=torch.zeros((1, 1, 1), dtype=torch.bool))
    assert zero.item() == 0


def test_validation_uses_the_same_strict_threshold_as_window_inference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    class Borderline(torch.nn.Module):
        def forward(self, image):
            return torch.tensor([-100.0, -100.0, 0.0, 0.0]).reshape(1, 4, 1, 1).expand(image.shape[0], 4, 2, 2)

    image = torch.ones((1, 3, 2, 2))
    truth = torch.full((1, 2, 2), 2, dtype=torch.long)
    score = _trainer._validate_multiclass_epoch(torch, Borderline(), [(image, truth)], "cpu", _config(), 1)
    assert score["best_threshold"] == 0.5
    assert score["foreground_pixel_f1"] == 0


@pytest.mark.parametrize("channels", [3, 4])
def test_multiclass_sampler_weights_and_augmentation_share_geometry(tmp_path, channels):
    import albumentations as A
    import cv2

    image, annotation = tmp_path / "scene.tif", tmp_path / "scene.geojson"
    transform = from_origin(0, 64, 1, 1)
    with rasterio.open(image, "w", driver="GTiff", width=64, height=64, count=channels, dtype="uint8", crs="EPSG:3857", transform=transform) as target:
        target.write(np.ones((channels, 64, 64), np.uint8))
        target.colorinterp = (rasterio.enums.ColorInterp.undefined,) * channels
    annotation.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {"_mlsystem2_class": "type_2"}, "geometry": mapping(box(0, 32, 16, 64))},
                     {"type": "Feature", "properties": {"_mlsystem2_class": "type_3"}, "geometry": mapping(box(16, 32, 32, 64))},
                     {"type": "Feature", "properties": {"_mlsystem2_role": "hard_negative", "_mlsystem2_class": "type_1"}, "geometry": mapping(box(32, 32, 48, 64))}]}))
    classes = [TileClassDefinition(class_id=item["id"], **{k: item[k] for k in ("slug", "name", "color")}) for item in SCHEMA]
    scene = TileSceneSource(scene_id="scene", image_path=image, annotation_file=annotation)
    split = TileSplitRequest(val_fraction=0.2, test_fraction=0.2, strategy="window_random", seed=42)
    assert TileDataloaderRequest(scenes=[scene], classes=classes, batch_size=2, mode="train", pipeline_variant="next_gen2", tile_split=split)
    dataset = TileDataset(scenes=[scene], classes=classes, tile_size=32, stride=32, mode="train", seed=42,
                          pipeline_variant="next_gen2", augmentation_level=0, input_channels=channels)
    try:
        assert len(dataset.notebook_class_weights) == 4
        assert np.isfinite(dataset.notebook_class_weights).all()
        np.testing.assert_allclose(dataset.sampling_weights(), np.array([7, 8, 1, 1]) / 17)
        _, before, meta = dataset[1]
        dataset._notebook_transform = A.Compose([A.HorizontalFlip(p=1)], strict=True, save_applied_params=True)
        _, after, moved = dataset[1]
        assert after.shape == (32, 32)
        np.testing.assert_array_equal(after, before[:, ::-1])
        np.testing.assert_array_equal(moved["class_hard_negative_masks"], meta["class_hard_negative_masks"][:, :, ::-1])
        # Проверяем реальную аффинную геометрию: простой flip не обнаруживает
        # неподдерживаемые OpenCV boolean-маски отрицательных примеров.
        for sample_index in (0, 1):
            dataset._notebook_transform = None
            _, before, meta = dataset[sample_index]
            dataset._notebook_transform = A.Compose([
                A.Affine(scale=1.08, rotate=17, translate_px={"x": 3, "y": -2},
                         border_mode=cv2.BORDER_REFLECT_101, p=1),
            ], strict=True, save_applied_params=True)
            _, after, moved = dataset[sample_index]
            matrix = dataset._notebook_transform.transforms[0].params["matrix"][:2]
            expected = cv2.warpAffine(
                before.astype(np.float32), matrix, (32, 32),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
            ).astype(np.int64)
            negatives = np.stack([
                cv2.warpAffine(value.astype(np.uint8), matrix, (32, 32),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                for value in meta["class_hard_negative_masks"]
            ]).astype(bool)
            assert after.dtype == np.int64
            assert moved["class_hard_negative_masks"].dtype == np.bool_
            np.testing.assert_array_equal(after, expected)
            np.testing.assert_array_equal(moved["class_hard_negative_masks"], negatives)
    finally:
        dataset.close()


def test_native_and_geoalert_choose_class_after_gaussian_merge(tmp_path):
    torch = pytest.importorskip("torch")
    from mlsystem2.models import _factory

    class Logits(torch.nn.Module):
        def forward(self, images):
            return torch.stack((images[:, 0] * 0, images[:, 0] * 0 + 1, images[:, 0] * 0 + 3,
                                images[:, 0] * 0 + 2), dim=1)

    spec = ModelSpec(name="smp_segformer_b0", input_channels=3, output_channels=4, pretrained=False,
                     parameters={"pipeline_variant": "next_gen2", "preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}})
    model = _factory._wrap_next_gen(spec, Logits()).model
    image = tmp_path / "scene.tif"
    profile = dict(driver="GTiff", width=48, height=48, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 48, 1, 1))
    with rasterio.open(image, "w", **profile) as target:
        target.write(np.ones((3, 48, 48), np.uint8))
    with rasterio.open(image) as source:
        labels, confidence = _pseudo_runner._infer_notebook_scene_mask(dataset=source, input_indexes=(1, 2, 3),
            input_channels=3, torch=torch, model=model, tile_size=32, stride=16, batch_size=2, threshold=0.3, device="cpu")
    assert np.all(labels == 2) and np.all(confidence > 0.6)
    probabilities = np.exp([0, 1, 3, 2]) / np.exp([0, 1, 3, 2]).sum()
    inputs, outputs = [f"p{i}" for i in range(4)], [f"mask{i}" for i in range(1, 4)]
    profile.update(count=1, dtype="float32")
    for name, value in zip(inputs, probabilities, strict=True):
        with rasterio.open(tmp_path / f"{name}.tif", "w", **profile) as target:
            target.write(np.full((48, 48), value, np.float32), 1)
    _multiclass_probability_masks(tmp_path, inputs, outputs, 0.3)
    for class_id, name in enumerate(outputs, start=1):
        with rasterio.open(tmp_path / f"{name}.tif") as target:
            np.testing.assert_array_equal(target.read(1), labels == class_id)
    _multiclass_probability_masks(tmp_path, inputs, outputs, 0.9)
    with rasterio.open(tmp_path / "mask2.tif") as target:
        assert not target.read(1).any()
