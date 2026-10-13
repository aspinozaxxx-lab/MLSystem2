"""Цели второго объектного профиля сохраняют площадь и разделительные швы."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from mlsystem2.tile_preparation._annotations import AnnotationIndex
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation._dataloader import _collate_tile_batch
from mlsystem2.tile_preparation._object_targets import boundary_targets, boundary_targets_v2
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest, TileSceneSource, TileSplitRequest


def _scenes(tmp_path):
    result = []
    for index in range(3):
        image = tmp_path / f"сцена{index}.tif"
        annotation = tmp_path / f"сцена{index}.geojson"
        x = index * 200
        data = np.full((4, 64, 64), 80, np.uint8)
        data[3] = 255
        data[3, :4] = 0
        with rasterio.open(image, "w", driver="GTiff", width=64, height=64, count=4,
                           dtype="uint8", crs="EPSG:3857", transform=from_origin(x, 64, 1, 1)) as source:
            source.write(data)
            source.colorinterp = tuple(rasterio.enums.ColorInterp[name] for name in ("red", "green", "blue", "alpha"))
        features = [{"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
                    for geometry in (box(x + 6, 6, x + 33, 58), box(x + 32, 6, x + 58, 58))]
        annotation.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        result.append(TileSceneSource(scene_id=f"сцена{index}", image_path=image, annotation_file=annotation))
    return result


def _dataset(scenes, mode="train", augmentation=0, variant="object_f1_v2"):
    return TileDataset(scenes=scenes, tile_size=64, stride=32, mode=mode, seed=42,
        augmentation_level=augmentation, include_object_instances=True, pipeline_variant=variant,
        input_channels=3, tile_split=TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2))


def test_wide_targets_preserve_full_ids_and_narrow_interiors():
    instances = np.zeros((96, 128), np.int32)
    instances[8:88, 8:58] = 11
    instances[8:88, 58:108] = 20
    instances[8:88, 110:116] = 42
    original = instances.copy()
    boundary, usable = boundary_targets_v2(instances, np.ones_like(instances, bool))
    np.testing.assert_array_equal(instances, original)
    assert np.all(boundary[45, 50:66] == 1)
    assert boundary[45, 49] == boundary[45, 66] == 0
    for key in (11, 20, 42):
        assert np.any((instances == key) & (boundary == 0) & usable)
        assert np.any((instances == key) & (boundary == 1) & usable)
    assert not np.any(boundary[instances == 0])


def test_crop_and_invalid_transitions_do_not_become_artificial_boundary():
    instances = np.ones((40, 40), np.int32)
    valid = np.ones_like(instances, bool)
    valid[12:18, 12:18] = False
    instances[~valid] = 0
    boundary, usable = boundary_targets_v2(instances, valid)
    assert not boundary.any()
    assert not usable[0].any() and not usable[-1].any()
    assert not usable[:, 0].any() and not usable[:, -1].any()
    assert not usable[11:19, 11:19].any()
    assert usable[9, 9]


def test_one_pixel_overlap_keeps_new_seam_and_legacy_stays_ignored():
    instances = np.ones((48, 64), np.int32)
    instances[:, 32:] = 2
    valid = np.ones_like(instances, bool)
    ambiguous = np.zeros_like(valid)
    ambiguous[:, 32] = True
    old_boundary, old_usable = boundary_targets(instances, valid, ambiguous)
    boundary, usable = boundary_targets_v2(instances, valid)
    assert old_boundary[20, 31] == 1 and not old_usable[20, 31]
    assert boundary[20, 31] == boundary[20, 32] == 1
    assert usable[20, 31] and usable[20, 32]


def test_overlap_audit_distinguishes_duplicates_and_serious_conflicts():
    index = AnnotationIndex([box(0, 0, 20, 20), box(19, 0, 40, 20), box(0, 0, 20, 20)])
    audit = index.instance_diagnostics((0, 0, 40, 20), detailed=True)
    assert audit["overlapping_polygon_pairs"] == 3
    assert audit["duplicate_polygon_pairs"] == 1
    assert audit["substantial_overlap_pairs"] == 1
    assert audit["annotation_warnings"]
    assert len(index.query_instances((0, 0, 40, 20))) == 3
    assert index.instance_diagnostics((0, 0, 40, 20)) == {
        "polygon_parts": 3, "overlapping_polygon_pairs": 3}


def test_v2_loader_rgb_alpha_split_batch_targets_and_train_statistics(tmp_path):
    scenes = _scenes(tmp_path)
    datasets = [_dataset(scenes, mode) for mode in ("train", "val", "test")]
    try:
        memberships = [{item.scene_id for item in dataset._windows} for dataset in datasets]
        assert all(not memberships[a] & memberships[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        for dataset in datasets:
            image, mask, meta = dataset[0]
            assert image.shape == (3, 64, 64)
            np.testing.assert_array_equal(mask[0] > 0, meta["object_instances"] > 0)
            assert not meta["valid_pixels"][:4].any()
            assert not meta["boundary_valid"][:5].any()
            assert meta["boundary_valid"][20, 32]
            assert meta["boundary_target"][20, 31:33].all()
            assert meta["overlap_pixels"] > 0
        train = datasets[0]
        image, mask, meta = train[0]
        target = (meta["object_instances"] > 0).astype(np.int32)
        target[meta["boundary_target"] > 0] = 2
        counts = np.bincount(target[meta["boundary_valid"]], minlength=3)
        np.testing.assert_array_equal(train.object_target_counts, counts)
        expected = 1 / counts
        expected /= expected.mean()
        expected[2] *= 3
        np.testing.assert_allclose(train.notebook_class_weights, expected, rtol=1e-6)
        assert datasets[1].object_target_counts == datasets[2].object_target_counts == []
        assert train.tile_split_manifest["boundary_width_pixels"] == 8
        assert train.scene_tile_diagnostics[0]["overlapping_polygon_pairs"] == 1
        pytest.importorskip("torch")
        _, _, batch_meta = _collate_tile_batch([train[0], train[0]])
        assert batch_meta["boundary_target"].shape == (2, 64, 64)
        assert len(batch_meta["scene_ids"]) == 2
    finally:
        for dataset in datasets:
            dataset.close()


def test_v2_augmentation_reproducible_with_synchronous_ids_and_masks(tmp_path):
    scenes = _scenes(tmp_path)
    first, second = _dataset(scenes, augmentation=3), _dataset(scenes, augmentation=3)
    try:
        for _ in range(3):
            left, right = first[0], second[0]
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[2]["object_instances"], right[2]["object_instances"])
            np.testing.assert_array_equal(left[2]["boundary_target"], right[2]["boundary_target"])
            np.testing.assert_array_equal(left[1][0] > 0, left[2]["object_instances"] > 0)
    finally:
        first.close()
        second.close()


def test_v2_contract_requires_object_ids_and_independent_scenes(tmp_path):
    scenes = _scenes(tmp_path)
    request = dict(scenes=scenes, batch_size=1, mode="test", pipeline_variant="object_f1_v2",
                   input_channels=3, include_object_instances=True,
                   tile_split=TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2))
    assert TileDataloaderRequest(**request).pipeline_variant == "object_f1_v2"
    with pytest.raises(ValueError, match="полные эпохи"):
        TileDataloaderRequest(**{**request, "include_object_instances": False})


def test_v2_class_weights_ignore_heldout_annotations_and_missing_classes(tmp_path):
    scenes = _scenes(tmp_path)
    original = _dataset(scenes)
    try:
        weights = original.notebook_class_weights
        counts = original.object_target_counts
        train_scenes = {item.scene_id for item in original._windows}
    finally:
        original.close()
    empty = json.dumps({"type": "FeatureCollection", "features": []})
    for scene in scenes:
        if scene.scene_id not in train_scenes:
            scene.annotation_file.write_text(empty, encoding="utf-8")
    changed = _dataset(scenes)
    try:
        assert changed.object_target_counts == counts
        assert changed.notebook_class_weights == weights
    finally:
        changed.close()
    for scene in scenes:
        scene.annotation_file.write_text(empty, encoding="utf-8")
    background_only = _dataset(scenes)
    try:
        assert background_only.object_target_counts[0] > 0
        assert background_only.object_target_counts[1:] == [0, 0]
        assert background_only.notebook_class_weights == [1.0, 1.0, 3.0]
    finally:
        background_only.close()


@pytest.mark.parametrize("zoned", [False, True])
def test_parallel_statistics_match_previous_full_rasterization(tmp_path, zoned, caplog):
    from rasterio.windows import Window
    from mlsystem2.tile_preparation._object_targets import rasterize_objects

    caplog.set_level("INFO", logger="mlsystem2.tile_preparation")
    scenes = _scenes(tmp_path)
    for index, scene in enumerate(scenes):
        payload = json.loads(scene.annotation_file.read_text(encoding="utf-8"))
        payload["features"].append({"type": "Feature", "properties": {"_mlsystem2_role": "hard_negative"},
                                    "geometry": mapping(box(index * 200 + 20, 8, index * 200 + 35, 24))})
        scene.annotation_file.write_text(json.dumps(payload), encoding="utf-8")
        if zoned:
            scenes[index] = scene.model_copy(update={"region_geometry": mapping(box(index * 200 + 3, 3, index * 200 + 61, 61)),
                "region_window": (0, 0, 64, 64), "parent_scene_id": scene.scene_id, "zone_id": "зона"})
    options = dict(scenes=scenes, tile_size=32, stride=16, mode="train", seed=42,
        augmentation_level=3, include_object_instances=True, pipeline_variant="object_f1_v2",
        input_channels=3, tile_split=TileSplitRequest(strategy="scene_groups", val_fraction=.2, test_fraction=.2))
    serial, parallel = TileDataset(**options), TileDataset(**options, statistics_workers=2)
    try:
        counts = np.zeros(3, dtype=np.float64)
        ratios, negatives = [], []
        for item in serial._windows:
            source = serial._open_dataset(item.scene_index)
            window = Window(item.window.x, item.window.y, 32, 32)
            invalid = serial._region_invalid(item.scene_index, source, window) if zoned else serial._read_invalid_data_pixels(source, window)
            mask = serial._read_supervision_mask(item.scene_index, source, window, invalid)[0]
            instances, _ = rasterize_objects(serial._positive_index(item.scene_index), source, window, 32, invalid | (mask < 0))
            boundary, usable = boundary_targets_v2(instances, ~invalid)
            target = (instances > 0).astype(np.int32)
            target[boundary > 0] = 2
            counts += np.bincount(target[usable], minlength=3)
            ratios.append(float(np.count_nonzero((mask > 0) & ~invalid) / max(1, np.count_nonzero(~invalid))))
            negatives.append(bool(np.any(mask == -1)))
        for dataset in (serial, parallel):
            np.testing.assert_array_equal(dataset.object_target_counts, counts)
            np.testing.assert_array_equal(dataset._notebook_positive_ratios, ratios)
            np.testing.assert_array_equal(dataset._notebook_hard_negatives, negatives)
        assert serial.notebook_class_weights == parallel.notebook_class_weights
        assert serial.tile_split_manifest == parallel.tile_split_manifest
        assert serial.sampling_weights() == parallel.sampling_weights()
        left, right = serial[0], parallel[0]
        for a, b in ((left[0], right[0]), (left[1], right[1]), (left[2]["boundary_target"], right[2]["boundary_target"])):
            np.testing.assert_array_equal(a, b)
        assert "процессов=2" in caplog.text
        assert "статистика train готова" in caplog.text
    finally:
        serial.close()
        parallel.close()


def test_statistics_skip_unused_overlap_raster_without_changing_ids(tmp_path, monkeypatch):
    from rasterio.windows import Window
    from mlsystem2.tile_preparation import _object_targets

    dataset = _dataset(_scenes(tmp_path))
    try:
        item = dataset._windows[0]
        source = dataset._open_dataset(item.scene_index)
        window = Window(item.window.x, item.window.y, 64, 64)
        index = dataset._positive_index(item.scene_index)
        invalid = dataset._read_invalid_data_pixels(source, window)
        reference, overlaps = _object_targets.rasterize_objects(index, source, window, 64, invalid)
        calls = []
        original = _object_targets.rasterize

        def tracked_rasterize(*args, **kwargs):
            calls.append(kwargs.get("merge_alg"))
            return original(*args, **kwargs)

        monkeypatch.setattr(_object_targets, "rasterize", tracked_rasterize)
        actual, unused = _object_targets.rasterize_objects(index, source, window, 64, invalid, collect_overlaps=False)
        np.testing.assert_array_equal(actual, reference)
        assert overlaps.any()
        assert unused is None and len(calls) == 1
    finally:
        dataset.close()


def _record_statistics_shutdown(dataset, marker):
    marker.write_text("закрыт" if not dataset._datasets else "открыт", encoding="utf-8")


def _statistics_shutdown_probe(paths, marker):
    from collections import OrderedDict
    from multiprocessing.util import Finalize
    from types import SimpleNamespace
    from mlsystem2.tile_preparation._statistics import _initialize_worker

    dataset = TileDataset.__new__(TileDataset)
    dataset._datasets = OrderedDict()
    dataset._annotation_indexes = {}
    dataset._scenes = [SimpleNamespace(image_path=path) for path in paths]
    _initialize_worker(dataset)
    for index in range(len(paths)):
        dataset._open_dataset(index).read(1)
    # Проверка завершается после рабочего finalizer, но до разрушения интерпретатора.
    # Без явного закрытия глобального датасета все TIFF в этот момент ещё открыты.
    Finalize(None, _record_statistics_shutdown, args=(dataset, marker), exitpriority=0)


def test_statistics_worker_closes_gdal_before_python_teardown(tmp_path):
    import multiprocessing

    marker = tmp_path / "закрытие.txt"
    paths = [scene.image_path for scene in _scenes(tmp_path)]
    process = multiprocessing.get_context("spawn").Process(
        target=_statistics_shutdown_probe, args=(paths, marker))
    process.start()
    try:
        process.join(timeout=30)
        assert not process.is_alive(), "Процесс статистики не завершился"
        assert process.exitcode == 0
        assert marker.read_text(encoding="utf-8") == "закрыт"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        process.close()
