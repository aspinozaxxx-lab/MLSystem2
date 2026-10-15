"""Снимок масок именно того запуска, который рассчитал тестовый F1."""

from __future__ import annotations

import json
import hashlib
import gzip
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes
from rasterio.warp import transform_bounds, transform_geom

from ._raster_http import raster_revision
from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request


def raster_content_sha256(path: Path) -> str:
    stat = path.stat()
    return _raster_content_sha256(str(path.resolve()), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _raster_content_sha256(path: str, modified: int, changed: int, size: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_test_f1_scene(
    root: Path, config: dict[str, Any], tile: dict[str, Any],
    truth: np.ndarray, prediction: np.ndarray, report: dict[str, Any],
    *, object_snapshots: dict[int | None, tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] | None = None,
) -> dict[str, Any]:
    """Сохранить эталон и фактический прогноз до удаления временных файлов."""
    root.mkdir(parents=True, exist_ok=True)
    scene_id = f"tile-{int(tile['index'])}"
    image_path = Path(tile["image_path"])
    viewer_image_path = Path(tile.get("viewer_image_path") or image_path)
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
        "id": scene_id, "image_path": str(viewer_image_path), "image_revision": raster_revision(viewer_image_path),
        "image_sha256": raster_content_sha256(image_path),
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


@lru_cache(maxsize=2)
def _object_snapshot(mask_path: str, mtime_ns: int, size: int, class_id: int | None,
                     source: tuple[str, int, int, str] | None, schema: tuple[tuple[int, str], ...]):
    """Восстановить экземпляры из сохранённого прогноза без загрузки сети."""
    from scipy.ndimage import label

    with np.load(mask_path, allow_pickle=False) as masks:
        prefix = f"objects_{class_id if class_id is not None else 0}"
        if f"{prefix}_matches" in masks:
            return masks[f"{prefix}_reference"], masks[f"{prefix}_predicted"], masks[f"{prefix}_matches"], None
        reference, predicted = masks["reference"], masks["predicted"]
    truth = reference == class_id if class_id is not None else reference > 0
    prediction = predicted == class_id if class_id is not None else predicted > 0
    true_instances = None
    if source is not None:
        from ._pseudo_runner import _test_tile_instance_mask, _test_tile_class_instance_masks, _combine_class_instance_masks

        geojson_path, _, _, image_path = source
        try:
            if schema:
                instances = _test_tile_class_instance_masks(Path(geojson_path), Path(image_path), truth.shape,
                                                            [{"id": key, "slug": slug} for key, slug in schema])
                candidate = instances[class_id] if class_id is not None else _combine_class_instance_masks(instances)
            else:
                candidate = _test_tile_instance_mask(Path(geojson_path), Path(image_path), truth.shape)
            # Никогда не смешивать новый эталон со старым прогнозом.
            if np.array_equal(candidate > 0, truth):
                true_instances = candidate
        except (OSError, ValueError, RuntimeError):
            pass
    reconstructed = true_instances is None
    if reconstructed:
        true_instances = label(truth, structure=np.ones((3, 3), dtype=np.uint8))[0]
    predicted_instances = label(prediction, structure=np.ones((3, 3), dtype=np.uint8))[0]
    score = compute_object_f1(ObjectF1Request(y_true_instances=true_instances, y_pred_instances=predicted_instances))
    return true_instances, predicted_instances, np.asarray(score.matched_pairs, np.int32).reshape(-1, 2), reconstructed


@lru_cache(maxsize=128)
def test_f1_object_score(mask_path: str, mtime_ns: int, size: int, class_id: int | None,
                         source: tuple[str, int, int, str] | None, schema: tuple[tuple[int, str], ...]):
    reference, predicted, pairs, reconstructed = _object_snapshot(mask_path, mtime_ns, size, class_id, source, schema)
    tp = len(pairs)
    return {"true_positive": tp, "false_positive": np.count_nonzero(np.unique(predicted)) - tp,
            "false_negative": np.count_nonzero(np.unique(reference)) - tp}, reconstructed


def test_f1_layers(
    mask_path: str, mtime_ns: int, size: int, transform_values: tuple[float, ...],
    crs: str, class_id: int | None, metric: str = "pixel",
    source: tuple[str, int, int, str] | None = None, schema: tuple[tuple[int, str], ...] = (),
) -> dict[str, Any]:
    """Точные TP/FP/FN в исходной сетке, без объединения разных тестовых TIFF."""
    with np.load(mask_path, allow_pickle=False) as masks:
        reference, predicted = masks["reference"], masks["predicted"]
    if metric == "objects":
        true_instances, predicted_instances, pairs, _ = _object_snapshot(mask_path, mtime_ns, size, class_id, source, schema)
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
    # Без общей среды каждый полигон повторно создаёт GDAL/AWS-сессию.
    with rasterio.Env():
        for layer, mask in masks.items():
            for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                features.append({
                    "type": "Feature", "properties": {"test_f1_layer": layer},
                    "geometry": transform_geom(crs, "EPSG:3857", geometry),
                })
    return {"type": "FeatureCollection", "features": features}


_LAYER_CONTENT_CACHE: OrderedDict[tuple, bytes] = OrderedDict()
_LAYER_CONTENT_LOCK = Lock()
_LAYER_CONTENT_LIMIT = 64 * 1024 * 1024
_LAYER_CONTENT_BYTES = 0


def test_f1_layer_content(
    mask_path: str, mtime_ns: int, size: int, transform_values: tuple[float, ...],
    crs: str, class_id: int | None, metric: str = "pixel",
    source: tuple[str, int, int, str] | None = None, schema: tuple[tuple[int, str], ...] = (),
) -> bytes:
    """Кэшировать сжатый JSON, а не миллионы Python-объектов координат."""
    global _LAYER_CONTENT_BYTES
    key = mask_path, mtime_ns, size, transform_values, crs, class_id, metric, source, schema
    with _LAYER_CONTENT_LOCK:
        if key in _LAYER_CONTENT_CACHE:
            _LAYER_CONTENT_CACHE.move_to_end(key)
            return _LAYER_CONTENT_CACHE[key]
    content = gzip.compress(json.dumps(test_f1_layers(*key), ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")).encode("utf-8"), compresslevel=1, mtime=0)
    with _LAYER_CONTENT_LOCK:
        # Одновременный запрос мог уже опубликовать тот же неизменяемый ответ.
        if key in _LAYER_CONTENT_CACHE:
            _LAYER_CONTENT_CACHE.move_to_end(key)
            return _LAYER_CONTENT_CACHE[key]
        if len(content) <= _LAYER_CONTENT_LIMIT:
            _LAYER_CONTENT_CACHE[key] = content
            _LAYER_CONTENT_BYTES += len(content)
            while _LAYER_CONTENT_BYTES > _LAYER_CONTENT_LIMIT or len(_LAYER_CONTENT_CACHE) > 128:
                _, removed = _LAYER_CONTENT_CACHE.popitem(last=False)
                _LAYER_CONTENT_BYTES -= len(removed)
    return content
