"""Регрессия обучения на RGB, RGB+alpha и RGB+NIR через штатные контракты."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetClassAnnotation, DatasetClassDefinition
from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import TilePreparationError, TileSplitRequest
from mlsystem2.train import _trainer
from mlsystem2.train.contracts import TrainConfig
from mlsystem2.train_pipeline import _runner


def _fixture(tmp_path, *, channels=3, alpha=True, hidden=243, nodata=None):
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for index, rgba in enumerate((alpha, False)):
        count = channels + int(rgba)
        data = np.stack([np.full((64, 64), 35 + band * 30, np.uint8) for band in range(count)])
        if rgba:
            data[-1] = 255
            data[-1, :16] = 0
            data[:-1, :16] = hidden
        path = images / f"scene{index}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=count,
                           dtype="uint8", crs="EPSG:3857", transform=from_origin(index * 100, 64, 1, 1),
                           nodata=nodata) as dst:
            dst.write(data)
            dst.colorinterp = (
                (ColorInterp.gray, ColorInterp.undefined, ColorInterp.undefined, ColorInterp.undefined)
                if channels == 4 and not rgba else
                (ColorInterp.red, ColorInterp.green, ColorInterp.blue, *([ColorInterp.alpha] if rgba else []))
            )
            for band, name in enumerate(["RED", "GRN", "BLU", "ALPHA" if rgba else "NIR"][:count], 1):
                dst.set_band_description(band, name)
        payload = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                   "features": [{"type": "Feature", "id": "positive", "properties": {"_mlsystem2_role": "positive", "_mlsystem2_class": "class_a"},
                                 "geometry": mapping(box(index * 100, 32, index * 100 + 32, 64))}]}
        (annotations / f"images_scene{index}.geojson").write_text(json.dumps(payload), encoding="utf-8")
    return images, annotations


def _prepare(images, annotations, variant, channels):
    settings = SimpleNamespace(
        dataset=SimpleNamespace(images_dir=str(images), scenes_file=None, annotation_file=None,
                                hard_negative_annotation_file=None, annotations_dir=str(annotations),
                                classes=[], val_fraction=0.2),
        train=SimpleNamespace(pipeline_variant=variant, input_channels=channels))
    return prepare_dataset(_runner._dataset_request(settings))


def _tiles(prepared, variant, channels, *, mode="val", augmentation=0):
    split = TileSplitRequest(strategy="scene_groups" if variant == "object_f1" else "scene_fold" if variant == "next_gen" else "window_random",
                             val_fraction=0.2, test_fraction=0.2 if variant in {"next_gen2", "object_f1"} else 0)
    request = _runner._tile_request(prepared, 2, mode, split, pipeline_variant=variant,
                                   input_channels=channels, include_object_instances=variant == "object_f1")
    assert request.input_channels == channels
    return TileDataset(scenes=request.scenes, tile_size=64, stride=32, mode=mode, seed=42,
                       augmentation_level=augmentation, pipeline_variant=variant, input_channels=request.input_channels,
                       class_annotations=request.class_annotations, classes=request.classes,
                       include_object_instances=request.include_object_instances)


@pytest.mark.parametrize("variant", ["legacy", "next_gen2", "object_f1"])
@pytest.mark.parametrize("nodata", [None, 0])
def test_mixed_rgb_and_rgba_training_channels_and_alpha_validity(tmp_path, variant, nodata):
    images, annotations = _fixture(tmp_path, nodata=nodata)
    prepared = _prepare(images, annotations, variant, 3)
    assert prepared.report.status == "ok", prepared.report.errors
    assert prepared.report.band_count == 3 and prepared.report.dtypes == ["uint8"] * 3
    tiles = _tiles(prepared.dataset, variant, 3)
    try:
        rgba_index = next(i for i, w in enumerate(tiles._windows) if w.scene_index == 0 and w.window.x == w.window.y == 0)
        rgb_index = next(i for i, w in enumerate(tiles._windows) if w.scene_index == 1 and w.window.x == w.window.y == 0)
        image, mask, meta = tiles[rgba_index]
        assert image.shape == (3, 64, 64)
        assert not meta["valid_pixels"][:16].any() and meta["valid_pixels"][16:].all()
        assert not image[:, :16].any() and not mask[..., :16, :].any()
        batch = _collate_tile_batch([tiles[rgba_index], tiles[rgb_index]])
        assert batch[0].shape == (2, 3, 64, 64)
        assert batch[2]["valid_pixels"].shape == (2, 64, 64)
        assert batch[2]["valid_pixels"][1].all()
    finally:
        tiles.close()


@pytest.mark.parametrize("variant", ["legacy", "next_gen", "next_gen2", "object_f1"])
def test_kanopus_keeps_all_four_bands_and_values(tmp_path, variant):
    images, annotations = _fixture(tmp_path, channels=4, alpha=False, nodata=0)
    prepared = _prepare(images, annotations, variant, 4)
    assert prepared.report.status == "ok", prepared.report.errors
    tiles = _tiles(prepared.dataset, variant, 4)
    try:
        image, _, _ = tiles[0]
        assert image.shape == (4, 64, 64)
        np.testing.assert_array_equal(image[:, 20, 20], [35, 65, 95, 125])
        assert tiles.channel_count == 4
    finally:
        tiles.close()


@pytest.mark.parametrize("variant", ["legacy", "next_gen2", "object_f1"])
def test_rgba_hidden_pixels_do_not_change_augmentation_or_sampling(tmp_path, variant):
    directories = [tmp_path / "one", tmp_path / "two"]
    loaders = []
    for folder, hidden in zip(directories, (21, 249)):
        folder.mkdir()
        images, annotations = _fixture(folder, hidden=hidden)
        prepared = _prepare(images, annotations, variant, 3)
        loaders.append(_tiles(prepared.dataset, variant, 3, mode="train", augmentation=3))
    try:
        for index in range(len(loaders[0])):
            first, second = loaders[0][index], loaders[1][index]
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            np.testing.assert_array_equal(first[2]["valid_pixels"], second[2]["valid_pixels"])
            assert not first[0][:, ~first[2]["valid_pixels"]].any()
        if variant != "legacy":
            np.testing.assert_array_equal(loaders[0]._notebook_positive_ratios, loaders[1]._notebook_positive_ratios)
            assert loaders[0]._notebook_positive_ratios[0] == pytest.approx(1 / 6)
    finally:
        for loader in loaders:
            loader.close()


@pytest.mark.parametrize("variant", ["legacy", "next_gen", "next_gen2", "object_f1"])
def test_alpha_is_never_used_as_kanopus_nir(tmp_path, variant):
    images, annotations = _fixture(tmp_path)
    prepared = _prepare(images, annotations, variant, 4)
    assert prepared.report.status == "error"
    assert any("alpha" in message for message in prepared.report.errors)
    rgb = _prepare(images, annotations, "legacy", 3)
    with pytest.raises(TilePreparationError, match="NIR"):
        _tiles(rgb.dataset, variant, 4)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dataset_format", ["legacy_multiclass", "per_image_multiclass"])
def test_multiclass_preserves_raster_channel_contract(tmp_path, channels, dataset_format):
    images, annotations = _fixture(tmp_path, channels=channels, alpha=channels == 3)
    prepared = _prepare(images, annotations, "legacy", channels)
    assert prepared.report.status == "ok", prepared.report.errors
    classes = [DatasetClassDefinition(id=1, slug="class_a", name="Первый класс", color="#ff0000"),
               DatasetClassDefinition(id=2, slug="class_b", name="Второй класс", color="#00ff00")]
    dataset = prepared.dataset.model_copy(update={"format": dataset_format,
        "scenes": [s.model_copy(update={"annotation_file": None}) for s in prepared.dataset.scenes]
            if dataset_format == "legacy_multiclass" else prepared.dataset.scenes,
        "classes": classes if dataset_format == "per_image_multiclass" else [],
        "class_annotations": [DatasetClassAnnotation(class_id=c.id, slug=c.slug, name=c.name,
            annotation_file=str(annotations / "images_scene0.geojson")) for c in classes]
            if dataset_format == "legacy_multiclass" else []})
    tiles = _tiles(dataset, "legacy", channels)
    try:
        image, mask, meta = tiles[0]
        assert image.shape == (channels, 64, 64) and mask.shape == (64, 64)
        assert tiles.uses_multiclass_masks
        if channels == 3:
            assert not meta["valid_pixels"][:16].any()
            assert not image[:, :16].any() and not mask[:16].any()
    finally:
        tiles.close()


@pytest.mark.parametrize("nodata", [None, 0])
def test_fully_transparent_windows_are_filtered_even_with_hidden_rgb(tmp_path, nodata):
    from mlsystem2.tile_preparation._valid_footprint import filter_valid_windows
    from mlsystem2.tile_preparation._windows import TileWindow

    images, _ = _fixture(tmp_path, nodata=nodata)
    with rasterio.open(images / "scene0.tif", "r+") as source:
        alpha = source.read(4)
        alpha[:32] = 0
        source.write(alpha, 4)
    with rasterio.open(images / "scene0.tif") as source:
        windows = [TileWindow(0, 0, 32, 32), TileWindow(0, 32, 32, 32)]
        selected, diagnostics = filter_valid_windows(source, windows, nodata=0)
        assert selected == [windows[1]]
        assert diagnostics.black_filtered_window_count == 1


@pytest.mark.parametrize("variant", ["legacy", "next_gen2"])
def test_alpha_pixels_do_not_contribute_to_loss_or_gradients(tmp_path, variant):
    torch = pytest.importorskip("torch")
    images, annotations = _fixture(tmp_path)
    prepared = _prepare(images, annotations, variant, 3)
    tiles = _tiles(prepared.dataset, variant, 3)
    try:
        _, masks, meta = _collate_tile_batch([tiles[0]])
        valid = meta["valid_pixels"][:, None]
        logits = torch.zeros((1, 2 if variant == "next_gen2" else 1, 64, 64), requires_grad=True)
        config = TrainConfig(pipeline_variant=variant, epochs=1, batch_size=1, device="cpu",
                             learning_rate=1e-4, weight_decay=0.01, early_stopping_patience=2,
                             loss="cross_entropy_tversky" if variant == "next_gen2" else "bce_dice",
                             class_weights=[1, 1] if variant == "next_gen2" else [])
        original = _trainer._loss(torch, logits, masks, config, valid_pixels=valid)
        modified = logits.detach().clone()
        modified[:, :, :16] = 100
        torch.testing.assert_close(original, _trainer._loss(torch, modified, masks, config, valid_pixels=valid))
        original.backward()
        assert not logits.grad[:, :, :16].any()
        assert logits.grad[:, :, 16:].abs().sum() > 0
    finally:
        tiles.close()


@pytest.mark.parametrize("variant", ["legacy", "next_gen2"])
def test_validation_ignores_false_predictions_on_transparent_pixels(tmp_path, variant):
    torch = pytest.importorskip("torch")
    images, annotations = _fixture(tmp_path)
    prepared = _prepare(images, annotations, variant, 3)
    tiles = _tiles(prepared.dataset, variant, 3)
    try:
        batch = _collate_tile_batch([tiles[0]])
        output_channels = 2 if variant == "next_gen2" else 1
        logits = torch.randn((1, output_channels, 64, 64), generator=torch.Generator().manual_seed(42))

        class PredictionModel(torch.nn.Module):
            def forward(self, image, **kwargs):
                return logits

        config = TrainConfig(pipeline_variant=variant, epochs=1, batch_size=1, device="cpu",
                             learning_rate=1e-4, weight_decay=0.01, early_stopping_patience=2,
                             loss="cross_entropy_tversky" if variant == "next_gen2" else "bce_dice",
                             class_weights=[1, 1] if variant == "next_gen2" else [])
        first = _trainer._validate_epoch(torch, PredictionModel(), [batch], "cpu", config, 1)
        logits[:, :, :16] = 100
        second = _trainer._validate_epoch(torch, PredictionModel(), [batch], "cpu", config, 1)
        assert first == second
    finally:
        tiles.close()


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("variant", ["legacy", "next_gen2"])
def test_segformer_b2_optimizer_step_from_raster_batch(tmp_path, channels, variant):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from mlsystem2.models.api import create_model
    from mlsystem2.models.contracts import ModelSpec

    torch.set_num_threads(2)
    images, annotations = _fixture(tmp_path, channels=channels, alpha=channels == 3)
    prepared = _prepare(images, annotations, variant, channels)
    tiles = _tiles(prepared.dataset, variant, channels)
    try:
        batch = _collate_tile_batch([tiles[0]])
        model = create_model(ModelSpec(name="smp_segformer_b2", input_channels=channels, output_channels=1,
            parameters={"pipeline_variant": variant,
                        **({"preprocessing": {"mode": "window_minmax", "epsilon": 1e-6}}
                           if variant == "next_gen2" else {})}))
        config = TrainConfig(pipeline_variant=variant, epochs=1, batch_size=1, device="cpu",
                             learning_rate=1e-4, weight_decay=0.01, early_stopping_patience=2,
                             loss="cross_entropy_tversky" if variant == "next_gen2" else "bce_dice",
                             class_weights=[1, 1] if variant == "next_gen2" else [])
        before = [p.detach().clone() for p in model.model.parameters()]
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
        metrics = _trainer._train_epoch(torch, model.model, [batch], optimizer, torch.device("cpu"), config, 1)
        assert np.isfinite(metrics["loss"])
        assert any(not torch.equal(p, old) for p, old in zip(model.model.parameters(), before, strict=True))
    finally:
        tiles.close()
