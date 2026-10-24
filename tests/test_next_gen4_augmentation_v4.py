"""Вырезки внутри train-тайла и отдельная честная оценка плотных окон."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from mlsystem2.tile_preparation._augmentations import build_notebook_augmentations
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest, TileSceneSource, TileSplitRequest
from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config
from mlsystem2.train._dense_windows import DenseWindowMetrics


VARIANT = "next_gen4_augmentation_v4"
POLICY = "validity_guard_v4_context_crop_v1"


def test_empty_geometric_result_reverts_all_masks_once_only_for_new_profile():
    A = pytest.importorskip("albumentations")
    original = {"image": np.full((32, 32, 4), 30, np.uint8), "mask": np.ones((32, 32), np.uint8),
                "valid": np.ones((32, 32), np.uint8)}
    class EmptyGeometry:
        calls = 0
        def __call__(self, **values):
            self.calls += 1
            return {**{key: np.zeros_like(value) for key, value in values.items()}, "applied_transforms": [("Affine", {})]}
    for crop, reverted in ((None, False), (32, True)):
        transform = build_notebook_augmentations(42, masked=True, preserve_validity=True, context_crop_size=crop)
        transform.geometry = EmptyGeometry()
        transform.photometry = A.Compose([], save_applied_params=True)
        result = transform(**original)
        assert transform.geometry.calls == 1 and result["augmentation_guard"]["geometry_reverted"] is reverted
        for key in original:
            np.testing.assert_array_equal(result[key], original[key] if reverted else np.zeros_like(original[key]))


@pytest.mark.parametrize("channels", [3, 4])
def test_crop_coordinates_masks_validity_and_reproducible_spawn(channels):
    A = pytest.importorskip("albumentations")
    import cv2

    transform = build_notebook_augmentations(42, masked=True, preserve_validity=True, context_crop_size=64)
    crop = transform.geometry.transforms[0]
    assert isinstance(crop, A.RandomResizedCrop)
    assert crop.scale == (0.25, 1.0) and crop.ratio == (1.0, 1.0) and crop.p == 0.5
    crop.p = 1
    transform.geometry = A.Compose([crop], additional_targets={"valid": "mask"}, save_applied_params=True)
    transform.photometry = A.Compose([], save_applied_params=True)
    yy, xx = np.indices((64, 64))
    image = np.stack([20 + yy + xx] * channels, axis=-1).astype(np.uint8)
    valid = np.ones((64, 64), np.uint8)
    valid[:8] = 0
    image[valid == 0] = 0
    mask = (xx // 8).astype(np.int64) - 1
    negative = ((xx > 32) & (yy > 16)).astype(np.uint8)
    original = dict(image=image, mask=mask, valid=valid, masks=[negative])
    restored = pickle.loads(pickle.dumps(transform))
    actual = transform(**original)
    repeated = restored(**original)
    for key in original:
        np.testing.assert_array_equal(actual[key], repeated[key])
    assert not any(actual["augmentation_guard"].values())
    x1, y1, x2, y2 = actual["applied_transforms"][0][1]["crop_coords"]
    assert 0 <= x1 < x2 <= 64 and 0 <= y1 < y2 <= 64 and x2 - x1 == y2 - y1
    for key in ("mask", "valid"):
        np.testing.assert_array_equal(actual[key], cv2.resize(original[key][y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_NEAREST))
    np.testing.assert_array_equal(actual["masks"][0], cv2.resize(negative[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_NEAREST))
    assert np.array_equal(actual["valid"] > 0, actual["image"].any(axis=-1))
    # Прежняя v3 не получает новый случайный оператор.
    old = build_notebook_augmentations(42, masked=True, preserve_validity=True)
    assert not any(isinstance(item, A.RandomResizedCrop) for item in old.geometry.transforms)


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{n}" for n in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_profile_defaults_differ_only_by_variant(architecture, size):
    original = sanitize_template_config({"train.pipeline_variant": "next_gen4_augmentation_v3", "tile_preparation.tile_size": size}, architecture=architecture)
    new = sanitize_template_config({"train.pipeline_variant": VARIANT, "tile_preparation.tile_size": size}, architecture=architecture)
    assert new == {**original, "train.pipeline_variant": VARIANT}
    template = next(item for item in initial_templates() if item["architecture"] == architecture)
    assert VARIANT in template["config_schema"]["fields"][0]["options"]
    assert template["config_schema"]["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "valid-mask" in template["config_schema"]["pipeline_descriptions"][VARIANT]


def test_dense_metrics_count_pixels_not_average_f1_and_exclude_nodata():
    torch = pytest.importorskip("torch")
    masks = torch.ones((4, 1, 10, 10))
    masks[1, 0, 0] = 0  # Ровно 90% объекта.
    valid = torch.ones_like(masks, dtype=torch.bool)
    valid[2, 0, :2] = False  # Не полноценное окно, исключить.
    masks[3] = 0  # Пустое окно, исключить.
    probabilities = torch.full((4, 10, 10), 0.7)
    probabilities[0] = 0.95
    metrics = DenseWindowMetrics()
    metrics.observe(torch, probabilities, masks, valid)
    result = metrics.snapshot()
    group = result["groups"]["0.9"]
    assert group["window_count"] == 2
    assert group["thresholds"]["0.5"] == dict(tp=190, fp=10, fn=0, precision=0.95, recall=1.0, f1=380 / 390)
    assert group["thresholds"]["0.9"]["fn"] == 90
    assert result["groups"]["0.99"]["window_count"] == 1
    assert DenseWindowMetrics().snapshot()["groups"]["0.99"]["thresholds"]["0.9"]["f1"] is None


def test_dense_metrics_multiclass_ignore_ambiguous_negatives():
    torch = pytest.importorskip("torch")
    metrics = DenseWindowMetrics()
    masks = torch.full((2, 10, 10), 2)
    masks[0, :5] = 1
    negatives = torch.zeros((2, 2, 10, 10), dtype=torch.bool)
    negatives[1, 0, :2] = True
    metrics.observe(torch, torch.full((2, 10, 10), 0.95), masks, None, negatives)
    result = metrics.snapshot()["groups"]["0.99"]
    assert result["window_count"] == 1 and result["thresholds"]["0.9"]["tp"] == 100


def test_dense_threshold_is_strict_like_external_inference():
    torch = pytest.importorskip("torch")
    metrics = DenseWindowMetrics()
    metrics.observe(torch, torch.full((1, 10, 10), 0.9), torch.ones((1, 10, 10)), None)
    result = metrics.snapshot()
    assert result["comparison"] == "strict_more"
    assert result["groups"]["0.99"]["thresholds"]["0.9"]["tp"] == 0
    assert result["groups"]["0.99"]["thresholds"]["0.5"]["tp"] == 100


def test_continuation_cannot_silently_replace_saved_crop_policy(monkeypatch):
    from mlsystem2.models.contracts import ModelSpec
    from mlsystem2.train_pipeline import _runner
    from mlsystem2.train_pipeline.contracts import TrainPipelineError

    spec = ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1, parameters={
        "pipeline_variant": VARIANT, "augmentation_policy": POLICY,
        "preprocessing": {"mode": "global_minmax"}, "context_crop": {"output_size": 512},
    })
    loaded_model = SimpleNamespace(spec=spec.model_copy(deep=True))
    loaded_model.spec.parameters["context_crop"]["output_size"] = 1024
    monkeypatch.setattr(_runner, "_model_spec", lambda *args: spec)
    deps = SimpleNamespace(load_checkpoint=lambda request: SimpleNamespace(model=loaded_model))
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT, initial_checkpoint_uri="best.pt", device="cpu"))
    with pytest.raises(TrainPipelineError, match="Политика аугментаций"):
        _runner._load_or_create_model(settings, deps)
    loaded_model.spec = spec.model_copy(deep=True)
    assert _runner._load_or_create_model(settings, deps) is loaded_model


def test_audit_loader_keeps_exact_train_windows_and_disables_only_its_augmentations(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mlsystem2.tile_preparation import _dataloader

    path, annotation = tmp_path / "снимок.tif", tmp_path / "разметка.geojson"
    with rasterio.open(path, "w", driver="GTiff", width=256, height=256, count=4,
                       dtype="uint8", transform=from_origin(0, 256, 1, 1), crs="EPSG:3857", nodata=0) as output:
        output.write(np.full((4, 256, 256), 50, np.uint8))
        output.colorinterp = (rasterio.enums.ColorInterp.gray,) + (rasterio.enums.ColorInterp.undefined,) * 3
    annotation.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 256, 256))}
    ]}), encoding="utf-8")
    settings = SimpleNamespace(tile_size=64, context=0, stride=64, seed=42, augmentation_level=3,
        positive_factor=0.5, hard_negative_factor=0, background_factor=0.5, class_balance=None, num_workers=0)
    monkeypatch.setattr(_dataloader, "get_settings", lambda: SimpleNamespace(tile_preparation=settings))
    loader = _dataloader.create_tile_dataloader(TileDataloaderRequest(
        scenes=[TileSceneSource(scene_id="снимок", image_path=path, annotation_file=annotation)],
        batch_size=2, mode="train", pipeline_variant=VARIANT, input_channels=4,
        tile_split=TileSplitRequest(val_fraction=0.2, test_fraction=0.2),
    ))
    original = loader.dataset
    subset = loader.dense_window_loader.dataset
    snapshot = subset.dataset
    assert original._notebook_transform is not None and snapshot._notebook_transform is None
    assert original._zone_transform is not None and snapshot._zone_transform is None
    assert snapshot._windows == original._windows
    assert len(subset) == len(original)
    assert original.tile_split_manifest["dense_windows"]["foreground_fraction"]["0.99"] == len(original)
    assert snapshot._mode == "val" and original._mode == "train"
    images, masks, meta = next(iter(loader.dense_window_loader))
    assert images.shape == (2, 4, 64, 64) and (masks == 1).all()
    assert meta["augmented_tile_count"] == 0
    original.close()
    snapshot.close()


@pytest.mark.parametrize("with_test", [False, True])
def test_best_checkpoint_audit_is_separate_from_test_and_training(tmp_path, monkeypatch, with_test):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from mlsystem2.models import _factory
    from mlsystem2.models.api import create_model
    from mlsystem2.models.contracts import ModelSpec
    from mlsystem2.tile_preparation._adaptive_sampling import AdaptiveBackgroundSampler
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
    from mlsystem2.train.api import train_model
    from mlsystem2.train.contracts import TrainRequest, TrainConfig

    torch.set_num_threads(2)
    monkeypatch.setattr(_factory, "_create_smp_segformer", lambda spec, **kwargs: torch.nn.Conv2d(4, 2, 1))
    model = create_model(ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1, parameters={
        "pipeline_variant": VARIANT, "augmentation_policy": POLICY,
        "context_crop": {"area_scale": [0.25, 1.0], "ratio": [1.0, 1.0], "probability": 0.5,
                         "output_size": 512, "source": "within_retained_train_window"},
        "preprocessing": {"mode": "global_minmax", "epsilon": 1e-6, "nodata": 0.0,
                          "mask_output_nodata": True, "source": "retained_train_windows",
                          "low": [0.] * 4, "high": [255.] * 4, "valid_pixel_count": 4096, "window_count": 4},
    }))
    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, index):
            index = index[0] if isinstance(index, tuple) else index
            return torch.full((4, 32, 32), 40.), torch.full((1, 32, 32), float(index < 2)), {
                "sample_index": index, "valid_pixels": torch.ones((32, 32), dtype=torch.bool),
            }
    samples = Samples()
    sampler = AdaptiveBackgroundSampler([{"positive": i < 2, "valid_pixels": 1024,
        "scene_id": "снимок", "group": str(i), "hard_negative": False} for i in range(4)], seed=42)
    train = torch.utils.data.DataLoader(samples, batch_size=2, sampler=sampler, collate_fn=_collate_tile_batch)
    val = torch.utils.data.DataLoader(samples, batch_size=2, collate_fn=_collate_tile_batch)
    audit = torch.utils.data.DataLoader(torch.utils.data.Subset(samples, [0, 1]), batch_size=2, collate_fn=_collate_tile_batch)
    result = train_model(TrainRequest(model=model, train_loader=train, val_loader=val,
        test_loader=val if with_test else None, dense_train_loader=audit, sample_size=32,
        config=TrainConfig(pipeline_variant=VARIANT, epochs=2, batch_size=2, learning_rate=1e-4,
                           weight_decay=0.01, early_stopping_patience=5, device="cpu",
                           class_weights=[1., 1.], loss="cross_entropy_tversky"),
        checkpoint_dir=str(tmp_path)))
    assert result.diagnostics["dense_train_windows"]["evaluation_performed"]
    assert result.diagnostics["dense_train_windows"]["groups"]["0.99"]["window_count"] == 2
    assert ("test_metrics" in result.diagnostics) is with_test
    assert ("test_dense_windows" in result.diagnostics) is with_test
    assert len(result.diagnostics["adaptive_sampling"]["history"]) == 2
