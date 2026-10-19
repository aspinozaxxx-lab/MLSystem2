"""Неперекрывающаяся сетка, выбор размера и исключение nodata в инференсе."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import TileSceneSource, TileSplitRequest
from mlsystem2.training_ui_api import _geoalert_notebook, _model_export, _pseudo_runner
from mlsystem2.training_ui_api._templates import NEXT_GEN4_DEFAULT_CONFIG, initial_templates, sanitize_template_config


def raster(path, *, width=130, height=105, invalid=True):
    values = np.full((4, height, width), 80, dtype="uint8")
    if invalid:
        values[:, :10, :10] = 0
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=4,
                       dtype="uint8", photometric="MINISBLACK", transform=from_origin(0, height, 1, 1), crs="EPSG:3857", nodata=0) as target:
        target.write(values)
        target.colorinterp = (rasterio.enums.ColorInterp.gray,) + (rasterio.enums.ColorInterp.undefined,) * 3
        mask = np.full((height, width), 255, dtype="uint8")
        if invalid:
            mask[40:50, 40:50] = 0  # Под маской остаются ненулевые каналы.
        target.write_mask(mask)


def dataset(tmp_path, mode, *, zones=False):
    image = tmp_path / "image.tif"
    annotation = tmp_path / "annotation.geojson"
    if not image.exists():
        raster(image)
        annotation.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 130, 105))}]}), encoding="utf-8")
    scenes = [TileSceneSource(scene_id="снимок", image_path=str(image), annotation_file=str(annotation))]
    if zones:
        scenes = [TileSceneSource(scene_id=f"снимок::zone:{i}", parent_scene_id="снимок", zone_id=str(i),
            image_path=str(image), annotation_file=str(annotation), region_geometry=mapping(geometry),
            region_window=window) for i, geometry, window in (
                (1, box(3, 5, 100, 98), (3, 7, 97, 93)), (2, box(22, 0, 130, 90), (22, 15, 108, 90))) ]
    return TileDataset(scenes=scenes, annotation_file=None, hard_negative_annotation_file=None,
        tile_size=32, stride=32, context=0, mode=mode, seed=42, augmentation_level=0,
        positive_factor=0.5, hard_negative_factor=0, background_factor=0.5,
        tile_split=TileSplitRequest(val_fraction=0.2, test_fraction=0.2, seed=42, strategy="window_random"),
        input_channels=4, pipeline_variant="next_gen4", statistics_workers=0)


@pytest.mark.parametrize("zones", [False, True])
def test_grid_disjoint_same_source_zones_edges_and_determinism(tmp_path, zones):
    parts = {mode: dataset(tmp_path, mode, zones=zones) for mode in ("train", "val", "test")}
    keys = {mode: {(tile.window.x, tile.window.y) for tile in value._windows} for mode, value in parts.items()}
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"])
    assert all(x % 32 == 0 and y % 32 == 0 for values in keys.values() for x, y in values)
    assert dataset(tmp_path, "val", zones=zones).tile_split_manifest == parts["val"].tile_split_manifest
    if not zones:
        assert set.union(*keys.values()) == {(x, y) for y in range(0, 105, 32) for x in range(0, 130, 32)}
    for value in parts.values():
        assert value.tile_split_manifest["same_source_overlap_pairs"] == 0
        for i, tile in enumerate(value._windows):
            _, _, meta = value[i]
            valid = np.asarray(meta["valid_pixels"])
            if tile.window.x == 128:
                assert not valid[..., 2:].any()
            if tile.window.y == 96:
                assert not valid[..., 9:, :].any()
        value.close()


@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_template_api_and_settings_allow_tile_choice(size):
    from mlsystem2.training_ui_api._service import _validate_training_pipeline_variant
    config = sanitize_template_config({"train.pipeline_variant": "next_gen4", "tile_preparation.tile_size": size}, architecture="smp_segformer_b1")
    assert config["tile_preparation.stride"] == size
    assert config["train.batch_size"] == {512: 8, 768: 4, 1024: 2, 1536: 1}[size]
    assert NEXT_GEN4_DEFAULT_CONFIG["tile_preparation.stride"] == 512
    _validate_training_pipeline_variant({**config, "dataset.task": "binary", "dataset.imagery_type": "kanopus", "train.input_channels": 4}, "smp_segformer_b1")
    template = next(item for item in initial_templates() if item["architecture"] == "smp_segformer_b1")
    assert "next_gen4" in template["config_schema"]["fields"][0]["options"]


@pytest.mark.parametrize("pad", [False, True])
def test_native_gaussian_removes_raw_nodata_internal_mask_and_keeps_edges(tmp_path, monkeypatch, pad):
    image = tmp_path / "image.tif"
    raster(image)
    def predict(torch, model, images, **kwargs):
        return None, np.ones((len(images), images.shape[-2], images.shape[-1]), dtype="float32")
    monkeypatch.setattr(_pseudo_runner, "_predict_tiles", predict)
    with rasterio.open(image) as source:
        mask, confidence = _pseudo_runner._infer_notebook_scene_mask(dataset=source, input_indexes=(1, 2, 3, 4), input_channels=4,
            torch=None, model=None, tile_size=32, stride=16, batch_size=3, threshold=0.9, device="cpu", pad_edges=pad)
    assert not mask[:10, :10].any() and not mask[40:50, 40:50].any()
    assert not confidence[40:50, 40:50].any()
    assert mask[20, 20] == 1
    assert mask[-1, -1] == int(pad)


def test_next_gen4_masks_input_before_normalization(tmp_path, monkeypatch):
    image = tmp_path / "image.tif"
    raster(image)
    seen = []
    def predict(torch, model, images, **kwargs):
        seen.extend(images.copy())
        return None, np.ones((len(images), 128, 128), dtype="float32")
    monkeypatch.setattr(_pseudo_runner, "_predict_tiles", predict)
    with rasterio.open(image) as source:
        _pseudo_runner._infer_notebook_scene_mask(dataset=source, input_indexes=(1, 2, 3, 4), input_channels=4,
            torch=None, model=None, tile_size=128, stride=64, batch_size=3, threshold=0.9, device="cpu", pad_edges=True)
    assert not seen[0][:, 40:50, 40:50].any()
    assert np.all(seen[0][:, 20, 20] == 80)


def test_portable_gaussian_matches_native_and_preserves_internal_masks(tmp_path, monkeypatch):
    image = tmp_path / "input.tif"
    raster(image)
    _geoalert_notebook._split_unmasked(image, tmp_path, ["r", "g", "b", "n"])
    seen = []
    def predict(image):
        seen.append(image.copy())
        return np.ones((1, 128, 128), dtype="float32")
    _geoalert_notebook._sliding_window_probabilities(tmp_path, ["r", "g", "b", "n"], ["p"],
        predict, 128, 64, 0.25, pad_edges=True)
    assert not seen[0][:, 40:50, 40:50].any()
    with rasterio.open(tmp_path / "p.tif") as result:
        values = result.read(1)
        assert not values[:10, :10].any() and not values[40:50, 40:50].any()
        assert values[-1, -1] == pytest.approx(1)
    import yaml
    text = _model_export._pipeline_yaml("реки".encode().hex(), 768, 4, probability_output=True, pad_edges=True)
    bricks = yaml.safe_load(text)["config"]["bricks"]
    assert next(b for b in bricks if b["_class"] == "SlidingWindowSegmentation")["pad_edges"] is True
    assert bricks[-1]["_class"] == "ClipToRasterValidData"


def test_final_vector_clip_cannot_fill_nodata_holes(tmp_path):
    image = tmp_path / "image.tif"
    raster(image)
    with rasterio.open(image) as source:
        valid = _pseudo_runner.valid_data_mask(source)
        features = _pseudo_runner._features_from_mask(np.ones(valid.shape, dtype="uint8"), source.transform,
            source.crs, source.res, "снимок", {}, valid_mask=valid)
        geometries = [(rasterio.warp.transform_geom("EPSG:4326", source.crs, f["geometry"]), 1) for f in features]
        result = rasterio.features.rasterize(geometries, out_shape=valid.shape, transform=source.transform)
    np.testing.assert_array_equal(result, valid)


def test_output_nodata_mask_survives_onnx(tmp_path):
    torch = pytest.importorskip("torch")
    from mlsystem2.models._factory import _InputPreprocessingWrapper
    class Foreground(torch.nn.Module):
        def forward(self, image):
            return torch.cat((image[:, :1] * 0, image[:, :1] * 0 + 10), dim=1)
    model = _InputPreprocessingWrapper(Foreground(), {"mode": "window_minmax", "mask_output_nodata": True})
    image = torch.ones(1, 4, 32, 32)
    image[:, :, :8] = 0
    logits = model(image)
    assert logits[:, :, :8].max() < -900 and logits[:, :, 8:].min() > 9
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    target = tmp_path / "masked.onnx"
    _model_export._export_segmentation_mask_onnx(model=model, input_channels=4, output_channels=1,
        sample_size=32, threshold=0.9, onnx_path=target, probability_output=True)
    onnx.checker.check_model(str(target))
    result = reference.ReferenceEvaluator(str(target)).run(None, {"input": image.numpy()})[0]
    assert not result[..., :8, :].any()
