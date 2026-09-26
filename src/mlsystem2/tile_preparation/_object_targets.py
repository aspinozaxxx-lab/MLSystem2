"""Разметка границ объектов и независимые группы снимков."""

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import MergeAlg
from rasterio.features import rasterize, shapes
from scipy import ndimage
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform, unary_union

from .contracts import TilePreparationError


def rasterize_objects(index, dataset, window, size, invalid):
    parts = index.query_instances(dataset.window_bounds(window))
    if not parts:
        return np.zeros((size, size), dtype=np.int32), np.zeros((size, size), dtype=bool)
    transform = dataset.window_transform(window)
    instances = rasterize([(geometry, key) for key, geometry in parts],
                          out_shape=(size, size), transform=transform, dtype="int32")
    counts = rasterize([(geometry, 1) for _, geometry in parts],
                       out_shape=(size, size), transform=transform, dtype="int32", merge_alg=MergeAlg.add)
    instances[invalid] = 0
    return instances, counts > 1


def boundary_targets(instances, valid, ambiguous):
    """Внутренняя двухпиксельная граница; края вырезки и конфликты игнорируются."""
    low = ndimage.minimum_filter(instances, size=5, mode="nearest")
    high = ndimage.maximum_filter(instances, size=5, mode="nearest")
    boundary = (instances > 0) & ((low != instances) | (high != instances))
    usable = ndimage.binary_erosion(valid & ~ambiguous, structure=np.ones((5, 5)), border_value=0)
    return boundary.astype(np.float32), usable


def scene_group_split(scenes, request):
    footprints = []
    for scene in scenes:
        with rasterio.open(scene.image_path) as source:
            if scene.footprint_file is not None and Path(scene.footprint_file).is_file():
                data = json.loads(Path(scene.footprint_file).read_text(encoding="utf-8"))
                geometry = unary_union([shape(item["geometry"]) for item in data.get("features", []) if item.get("geometry")])
            else:
                # Без companion-файла читаем валидную территорию блоками, включая alpha.
                pieces = []
                for _, window in source.block_windows(1):
                    valid = source.dataset_mask(window=window) > 0
                    pieces.extend(shape(item) for item, _ in shapes(valid.astype("uint8"), mask=valid,
                        transform=source.window_transform(window)))
                geometry = unary_union(pieces)
            if source.crs is not None:
                geometry = transform(Transformer.from_crs(source.crs, "EPSG:4326", always_xy=True).transform, geometry)
            footprints.append(geometry)
    parent = list(range(len(scenes)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, geometry in enumerate(footprints):
        for j in range(i):
            if geometry.intersects(footprints[j]) and geometry.intersection(footprints[j]).area > 0:
                parent[root(i)] = root(j)
    groups = {}
    for i, scene in enumerate(scenes):
        groups.setdefault(root(i), []).append(scene.scene_id)
    ordered = sorted((sorted(group) for group in groups.values()),
                     key=lambda group: hashlib.sha256(f"{request.seed}|{'|'.join(group)}".encode()).digest())
    if len(ordered) < 3:
        raise TilePreparationError("object f1 требует минимум три независимые группы снимков для train/validation/test")
    selected = {"train": [], "val": [], "test": []}
    remaining = list(ordered)
    for mode, fraction, reserve in [("val", request.val_fraction, 2), ("test", request.test_fraction, 1)]:
        target = max(1, int(len(scenes) * fraction + 0.5))
        while len(remaining) > reserve and (not selected[mode] or len(selected[mode]) < target):
            selected[mode].extend(remaining.pop(0))
    selected["train"] = [scene for group in remaining for scene in group]
    return selected, {"strategy": "scene_groups", "seed": request.seed, "groups": ordered,
                      "train_scene_ids": selected["train"], "validation_scene_ids": selected["val"],
                      "test_scene_ids": selected["test"], "val_fraction": request.val_fraction,
                      "test_fraction": request.test_fraction}
