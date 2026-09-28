"""Зоны изолируют обучение от неразмеченной территории исходного снимка."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape

from mlsystem2.dataset_preparing.api import annotation_regions, prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import TileSceneSource


def feature(key, geometry, role="annotation_zone", slug=None):
    properties = {"_mlsystem2_role": role, "_mlsystem2_origin_key": key}
    if slug is not None:
        properties["_mlsystem2_class"] = slug
    return {"type": "Feature", "id": key, "geometry": mapping(geometry), "properties": properties}


def payload(features):
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}, "features": features}


def prepare_fixture(tmp_path, zones=None, channels=3, alpha=False):
    images = tmp_path / "images"
    images.mkdir(parents=True)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    raster = images / "scene.tif"
    data = np.full((channels, 32, 32), 80, dtype=np.uint8)
    data[:, 13:15, 9:11] = 0
    if alpha:
        data[-1] = 255
        data[-1, 13:15, 9:11] = 0
    with rasterio.open(raster, "w", driver="GTiff", width=32, height=32, count=channels,
                       dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 32, 1, 1), nodata=0) as target:
        target.write(data)
        if alpha:
            target.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                                  rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha)
        elif channels == 4:
            target.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                                  rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.undefined)
    markup = annotations / "images_scene.geojson"
    markup.write_text(json.dumps(payload([feature("object", box(5, 14, 13, 25), "positive"), *(zones or [])])), encoding="utf-8")
    result = prepare_dataset(DatasetPreparationRequest(images_dir=str(images), annotations_dir=str(annotations),
                                                       val_fraction=0.2, expected_band_count=3 if alpha else channels,
                                                       allow_rgb_alpha=alpha))
    assert result.report.status == "ok", result.report.errors
    return result, raster, markup


def loader(scenes, pipeline="legacy", tile=16, augmentation=0, mode="val", **kwargs):
    return TileDataset(scenes=[TileSceneSource(**scene.model_dump()) for scene in scenes],
                       tile_size=tile, stride=tile // 2, mode=mode, seed=42,
                       augmentation_level=augmentation, pipeline_variant=pipeline,
                       input_channels=3 if pipeline == "object_f1" else None, **kwargs)


def test_regions_keep_overlaps_and_intersect_classes():
    regions = annotation_regions(payload([feature("a", box(0, 0, 10, 10), slug="a"),
                                          feature("b", box(5, 0, 15, 10), slug="b")]), ["a", "b"])
    assert [item.zone_id for item in regions] == ["a", "b"]
    assert all(shape(item.geometry).equals(box(5, 0, 10, 10)) for item in regions)
    assert annotation_regions(payload([]), ["a", "b"]) is None
    assert annotation_regions(payload([feature("a", box(0, 0, 2, 2), slug="a"),
                                       feature("b", box(3, 3, 4, 4), slug="b")]), ["a", "b"]) == []
    assert len(annotation_regions(payload([feature("common", box(0, 0, 5, 5))]), ["a", "b"])) == 1


def test_preparation_preserves_source_and_independent_zones(tmp_path):
    result, raster, _ = prepare_fixture(tmp_path, [feature("one", box(4, 12, 14, 26)), feature("two", box(8, 10, 18, 24))])
    assert len(result.dataset.scenes) == 2
    assert {scene.image_path for scene in result.dataset.scenes} == {raster.resolve().as_posix()}
    assert len({scene.parent_scene_id for scene in result.dataset.scenes}) == 1
    assert result.dataset.scenes[0].region_window == (4, 6, 10, 14)
    assert result.report.positive_objects == 1
    assert result.report.scenes[0].annotation_zone_count == 2


@pytest.mark.parametrize("pipeline,context,channels", [("legacy", 0, 3), ("legacy", 4, 4), ("next_gen2", 0, 3), ("next_gen2", 0, 4), ("object_f1", 0, 3)])
def test_masked_tiles_do_not_depend_on_outside_image_or_objects(tmp_path, pipeline, context, channels):
    zone = Polygon([(4, 12), (14, 12), (14, 26), (4, 26)], holes=[[(7, 17), (10, 17), (10, 20), (7, 20)]])
    result, raster, markup = prepare_fixture(tmp_path, [feature("zone", zone)], channels=channels)
    first = loader(result.dataset.scenes, pipeline, mode="train", context=context)
    before = [first[i] for i in range(len(first))]
    old_ratios = first._notebook_positive_ratios
    first.close()
    with rasterio.open(raster, "r+") as target:
        image = target.read()
        outside = geometry_mask([mapping(zone)], out_shape=(32, 32), transform=target.transform)
        image[:, outside] = 247
        target.write(image)
    annotations = json.loads(markup.read_text())
    annotations["features"].append(feature("outside", box(20, 20, 30, 30), "positive"))
    markup.write_text(json.dumps(annotations), encoding="utf-8")
    second = loader(result.dataset.scenes, pipeline, mode="train", context=context)
    assert len(second) == len(before)
    for (image, mask, meta), index in zip(before, range(len(second))):
        actual_image, actual_mask, actual_meta = second[index]
        np.testing.assert_array_equal(image, actual_image)
        np.testing.assert_array_equal(mask, actual_mask)
        np.testing.assert_array_equal(meta["valid_pixels"], actual_meta["valid_pixels"])
        assert not image[:, ~meta["valid_pixels"]].any()
        assert not mask[..., ~meta["valid_pixels"]].any()
    if old_ratios is not None:
        np.testing.assert_array_equal(old_ratios, second._notebook_positive_ratios)
    second.close()


@pytest.mark.parametrize("pipeline", ["legacy", "next_gen2", "object_f1"])
def test_augmentation_keeps_invalid_area_empty_and_is_repeatable(tmp_path, pipeline):
    result, _, _ = prepare_fixture(tmp_path, [feature("zone", box(4, 12, 14, 26))])
    first = loader(result.dataset.scenes, pipeline, augmentation=2, mode="train")
    second = loader(result.dataset.scenes, pipeline, augmentation=2, mode="train")
    for i in range(len(first)):
        image, mask, meta = first[i]
        other_image, other_mask, other_meta = second[i]
        np.testing.assert_array_equal(image, other_image)
        np.testing.assert_array_equal(mask, other_mask)
        np.testing.assert_array_equal(meta["valid_pixels"], other_meta["valid_pixels"])
        assert not image[:, ~meta["valid_pixels"]].any()
        if pipeline == "object_f1":
            assert not meta["boundary_valid"][~meta["valid_pixels"]].any()
    first.close()
    second.close()


@pytest.mark.parametrize("tile", [512, 768, 1024, 1536])
def test_small_zone_is_padded_for_all_sizes(tmp_path, tile):
    result, _, _ = prepare_fixture(tmp_path, [feature("zone", box(4, 12, 14, 26))])
    dataset = loader(result.dataset.scenes, "next_gen2", tile=tile)
    image, mask, meta = dataset[0]
    assert image.shape == (3, tile, tile)
    assert mask.shape == (1, tile, tile)
    assert int(meta["valid_pixels"].sum()) == 136
    assert not image[:, 14:, :].any()
    dataset.close()


def test_multipolygon_and_rgba_use_alpha_validity(tmp_path):
    geometry = MultiPolygon([box(4, 12, 14, 26), box(20, 20, 23, 23)])
    result, _, _ = prepare_fixture(tmp_path, [feature("zone", geometry)], channels=4, alpha=True)
    dataset = loader(result.dataset.scenes, "object_f1", tile=32)
    image, _, meta = dataset[0]
    assert image.shape[0] == 3
    assert meta["valid_pixels"].sum() == 145
    assert meta["window"]["x"] == meta["window"]["y"] == 0
    assert meta["scene_shape"] == {"width": 19, "height": 14}
    dataset.close()


def test_alpha_is_honoured_when_nodata_is_also_set(tmp_path):
    result, raster, _ = prepare_fixture(tmp_path, [feature("zone", box(4, 12, 14, 26))], channels=4, alpha=True)
    with rasterio.open(raster, "r+") as target:
        values = target.read()
        values[:3, 8:10, 6:8] = 80
        values[3, 8:10, 6:8] = 0
        target.write(values)
    dataset = loader(result.dataset.scenes, "object_f1", tile=32)
    image, _, meta = dataset[0]
    assert meta["valid_pixels"].sum() == 132
    assert not image[:, ~meta["valid_pixels"]].any()
    dataset.close()


@pytest.mark.parametrize("pipeline,loss,channels", [("legacy", "bce_dice", 1), ("legacy", "focal_tversky", 1),
                                                     ("next_gen2", "cross_entropy_tversky", 2), ("legacy", "cross_entropy_dice", 3)])
def test_invalid_predictions_have_no_loss_or_gradient(pipeline, loss, channels):
    torch = pytest.importorskip("torch")
    from mlsystem2.train._trainer import _loss

    config = SimpleNamespace(pipeline_variant=pipeline, task="multiclass" if channels == 3 else "binary",
                             loss=loss, class_weights=[1.0, 1.0], pos_weight=1.0, background_weight=1.0,
                             hard_negative_weight=2.0, tversky_alpha=0.4, tversky_beta=0.6, focal_alpha=0.6)
    logits = torch.randn((2, channels, 8, 8), requires_grad=True)
    masks = torch.zeros((2, 8, 8), dtype=torch.long) if channels == 3 else torch.zeros((2, 1, 8, 8))
    masks[..., 2:5, 2:5] = 1
    valid = torch.zeros_like(masks, dtype=torch.bool)
    valid[..., 1:6, 1:6] = True
    value = _loss(torch, logits, masks, config, valid_pixels=valid)
    value.backward()
    expanded = valid.unsqueeze(1) if valid.ndim == 3 else valid
    assert not logits.grad.masked_select(~expanded.expand_as(logits)).any()
    changed = logits.detach().clone().masked_fill(~expanded.expand_as(logits), 123)
    torch.testing.assert_close(value, _loss(torch, changed, masks, config, valid_pixels=valid))
    empty = _loss(torch, logits, masks, config, valid_pixels=torch.zeros_like(valid))
    assert torch.isfinite(empty) and empty.item() == 0


def test_empty_clipped_zone_is_not_silently_removed():
    from mlsystem2.training_ui_api._dataset_editor import _clip_geojson_to_footprint
    from mlsystem2.training_ui_api.contracts import TrainingUIAPIError

    with pytest.raises(TrainingUIAPIError, match="Размеченная зона"):
        _clip_geojson_to_footprint(payload([feature("zone", box(20, 20, 30, 30))]), box(0, 0, 10, 10))


def test_reference_tiles_stay_inside_a_zone(tmp_path):
    from mlsystem2.training_ui_api._markup_export import _build_candidates, _load_annotations

    zone = box(4, 12, 14, 26)
    _, raster, markup = prepare_fixture(tmp_path, [feature("zone", zone)])
    annotations = _load_annotations(markup, positive_only=True)
    candidates = _build_candidates(sources=[(raster, annotations)], images_root=raster.parent,
                                   tile_width=4, tile_height=4, max_grid_origins=30)
    assert candidates
    assert all(zone.covers(item.raster_footprint) for item in candidates)
    assert not _build_candidates(sources=[(raster, annotations)], images_root=raster.parent,
                                tile_width=16, tile_height=16, max_grid_origins=30)


def test_mixed_batch_and_raster_handles(tmp_path):
    pytest.importorskip("torch")
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch

    result, _, _ = prepare_fixture(tmp_path, [feature("zone", box(4, 12, 14, 26))])
    zone = result.dataset.scenes[0]
    whole = zone.model_copy(update={"scene_id": "whole", "zone_id": None, "region_geometry": None, "region_window": None})
    dataset = loader([zone, whole], "next_gen2")
    zone_index = next(i for i, item in enumerate(dataset._windows) if item.scene_index == 0)
    whole_index = next(i for i, item in enumerate(dataset._windows) if item.scene_index == 1)
    images, masks, meta = _collate_tile_batch([dataset[zone_index], dataset[whole_index]])
    assert images.shape[0] == masks.shape[0] == 2
    assert not meta["valid_pixels"][0].all()
    assert meta["valid_pixels"][1].all()
    assert len(dataset._datasets) == 1
    dataset.close()


def test_scene_groups_keep_all_zones_of_parent_together(tmp_path):
    from mlsystem2.tile_preparation._object_targets import scene_group_split
    from mlsystem2.tile_preparation.contracts import TileSplitRequest

    result, _, _ = prepare_fixture(tmp_path, [feature("one", box(4, 12, 14, 26)), feature("two", box(18, 12, 25, 26))])
    scenes = []
    for index in range(3):
        footprint = tmp_path / f"footprint-{index}.geojson"
        footprint.write_text(json.dumps(payload([feature("footprint", box(index * 100, 0, index * 100 + 30, 30))])), encoding="utf-8")
        for scene in result.dataset.scenes:
            scenes.append(scene.model_copy(update={"scene_id": f"parent{index}:{scene.zone_id}",
                                                  "parent_scene_id": f"parent{index}", "footprint_file": str(footprint)}))
    parts, _ = scene_group_split(scenes, TileSplitRequest(val_fraction=.2, test_fraction=.2, strategy="scene_groups"))
    assert all(len(items) == 2 for items in parts.values())
    assert all(len({item.split(":")[0] for item in items}) == 1 for items in parts.values())


@pytest.mark.parametrize("pipeline,channels", [("legacy", 1), ("next_gen2", 2), ("legacy", 3), ("object_f1", 3)])
def test_validation_ignores_predictions_outside_zone(pipeline, channels):
    torch = pytest.importorskip("torch")
    from mlsystem2.train._trainer import _validate_epoch
    from mlsystem2.train.contracts import TrainConfig

    multiclass = channels == 3 and pipeline == "legacy"
    config = TrainConfig(pipeline_variant=pipeline, task="multiclass" if multiclass else "binary",
                         quality_metric="objects" if pipeline == "object_f1" else "pixel",
                         loss="cross_entropy_dice" if multiclass else "bce_dice" if pipeline == "legacy" else "cross_entropy_tversky",
                         class_weights=[1., 1.], class_slugs=["a", "b"] if multiclass else [],
                         epochs=1, batch_size=1, device="cpu", learning_rate=.0001, weight_decay=.01,
                         early_stopping_patience=2)
    valid = torch.zeros((1, 16, 16), dtype=torch.bool)
    valid[:, 3:13, 3:13] = True
    masks = torch.zeros((1, 16, 16), dtype=torch.long) if multiclass else torch.zeros((1, 1, 16, 16))
    masks[..., 5:10, 5:10] = 1
    meta = {"valid_pixels": valid}
    if pipeline == "object_f1":
        meta.update(boundary_target=torch.zeros((1, 16, 16)), boundary_valid=valid,
                    object_instances=masks[:, 0].int(), scene_ids=["scene:zone"],
                    scene_shapes=[{"width": 16, "height": 16}], windows=[{"x": 0, "y": 0}])
    logits = torch.randn((1, channels, 16, 16), generator=torch.Generator().manual_seed(42))
    changed = logits.masked_fill(~valid.unsqueeze(1), 40.)
    class PredictionModel(torch.nn.Module):
        def forward(self, image, **kwargs):
            return image
    first = _validate_epoch(torch, PredictionModel(), [(logits, masks, meta)], "cpu", config, 1)
    second = _validate_epoch(torch, PredictionModel(), [(changed, masks, meta)], "cpu", config, 1)
    assert first == second


@pytest.mark.parametrize("pipeline", ["next_gen2", "object_f1"])
def test_zone_augmentations_are_reproducible_in_workers(tmp_path, pipeline):
    torch = pytest.importorskip("torch")
    from mlsystem2.tile_preparation._dataloader import _collate_tile_batch, _seed_tile_worker

    result, _, _ = prepare_fixture(tmp_path, [feature("zone", box(4, 12, 14, 26))])
    def collect():
        dataset = loader(result.dataset.scenes, pipeline, augmentation=3, mode="train")
        batches = list(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2,
                       worker_init_fn=_seed_tile_worker, collate_fn=_collate_tile_batch,
                       multiprocessing_context="spawn", timeout=60,
                       generator=torch.Generator().manual_seed(42)))
        dataset.close()
        return batches
    for (image, mask, meta), (other_image, other_mask, other_meta) in zip(collect(), collect(), strict=True):
        torch.testing.assert_close(image, other_image)
        torch.testing.assert_close(mask, other_mask)
        torch.testing.assert_close(meta["valid_pixels"], other_meta["valid_pixels"])
