"""Снимок масок именно того запуска, который рассчитал тестовый F1."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes
from rasterio.warp import transform_bounds, transform_geom

from ._raster_http import raster_revision
from .contracts import TrainingUIAPIError


def save_test_f1_scene(
    root: Path, config: dict[str, Any], tile: dict[str, Any],
    truth: np.ndarray, prediction: np.ndarray, report: dict[str, Any],
    *, object_snapshots: dict[int | None, tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] | None = None,
) -> dict[str, Any]:
    """Сохранить эталон и фактический прогноз до удаления временных файлов."""
    root.mkdir(parents=True, exist_ok=True)
    scene_id = f"tile-{int(tile['index'])}"
    image_path = Path(tile["image_path"])
    with rasterio.open(image_path) as image:
        if image.crs is None:
            raise RuntimeError("У тестового снимка отсутствует система координат.")
        transform = list(image.transform)[:6]
        crs = image.crs.to_string()
        bounds = transform_bounds(image.crs, "EPSG:3857", *image.bounds, densify_pts=21)
    target = tile.get("target_class_id")
    # В managed-выборке бинарный эталон относится к конкретному выходному каналу.
    reference = (truth > 0).astype(np.uint8) * int(target) if target is not None else truth
    masks = {"reference": reference, "predicted": prediction}
    for class_id, (true_instances, predicted_instances, pairs) in (object_snapshots or {}).items():
        prefix = f"objects_{class_id if class_id is not None else 0}"
        masks[f"{prefix}_reference"] = true_instances.astype(np.int32, copy=False)
        masks[f"{prefix}_predicted"] = predicted_instances.astype(np.int32, copy=False)
        masks[f"{prefix}_matches"] = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    np.savez_compressed(root / f"{scene_id}.npz", **masks)
    return {
        "id": scene_id, "image_path": str(image_path), "image_revision": raster_revision(image_path),
        "transform": transform, "crs": crs, "bounds": list(bounds),
        "sample_name": tile.get("test_sample_name") or config.get("class_name") or "Тестовая разметка",
        "sample_id": tile.get("test_sample_id") or config.get("test_sample_id"),
        "sample_revision": int(tile.get("test_sample_revision") or config.get("test_sample_revision") or 1),
        "target_class_id": int(target) if target is not None else None,
        "target_class_slug": tile.get("target_class_slug"),
        "source_tile_index": int(tile.get("source_tile_index") or tile["index"]),
        "class_schema": list(config.get("object_types") or []), "report": report,
        "object_layers_available": bool(object_snapshots),
    }


def save_test_f1_manifest(root: Path, result_id: str, scenes: list[dict[str, Any]]) -> None:
    (root / "manifest.json").write_text(
        json.dumps({"version": 1, "training_result_id": result_id, "scenes": scenes}, ensure_ascii=False),
        encoding="utf-8",
    )


@lru_cache(maxsize=4)
def test_f1_layers(
    mask_path: str, mtime_ns: int, size: int, transform_values: tuple[float, ...],
    crs: str, class_id: int | None, metric: str = "pixel",
) -> dict[str, Any]:
    """Точные TP/FP/FN в исходной сетке, без объединения разных тестовых TIFF."""
    with np.load(mask_path, allow_pickle=False) as masks:
        reference, predicted = masks["reference"], masks["predicted"]
        if metric == "objects":
            prefix = f"objects_{class_id if class_id is not None else 0}"
            if f"{prefix}_matches" not in masks:
                raise TrainingUIAPIError("В прежнем расчёте не сохранены экземпляры объектов. Подготовьте визуализацию заново.")
            true_instances = masks[f"{prefix}_reference"]
            predicted_instances = masks[f"{prefix}_predicted"]
            pairs = masks[f"{prefix}_matches"]
    truth = reference == class_id if class_id is not None else reference > 0
    prediction = predicted == class_id if class_id is not None else predicted > 0
    masks = {
        "tp": truth & prediction, "fp": ~truth & prediction, "fn": truth & ~prediction,
        "reference": truth, "predicted": prediction,
    }
    if metric == "objects":
        matched_truth = np.isin(true_instances, pairs[:, 0])
        matched_prediction = np.isin(predicted_instances, pairs[:, 1])
        masks.update(tp=matched_truth | matched_prediction,
                     fp=(predicted_instances > 0) & ~matched_prediction,
                     fn=(true_instances > 0) & ~matched_truth)
    transform = Affine(*transform_values)
    features = []
    for layer, mask in masks.items():
        for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
            features.append({
                "type": "Feature", "properties": {"test_f1_layer": layer},
                "geometry": transform_geom(crs, "EPSG:3857", geometry),
            })
    return {"type": "FeatureCollection", "features": features}
