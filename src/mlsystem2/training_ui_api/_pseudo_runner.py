"""Запуск псевдоразметки для training UI API."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio import features as rasterio_features
from rasterio.warp import transform_geom
from rasterio.windows import Window
import yaml

from mlsystem2.mlflow_adapter.api import download_run_artifact
from mlsystem2.models.api import load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-training-ui-pseudo-runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run_pseudo_markup(payload)
    report_path = Path(payload["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"ok", "partial"} else 1


def run_pseudo_markup(config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run_root = Path(config["run_root"])
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    output_geojson = Path(config["output_geojson"])

    scenes = _read_scenes(Path(config["scenes_file"]))
    image_index = _image_index(Path(config["images_root"]))
    checkpoint_path = _resolve_checkpoint(config, run_root / "checkpoint")
    threshold = float(config.get("threshold") or 0.5)
    tile_size = int(config.get("tile_size") or 768)
    stride = int(config.get("stride") or tile_size)
    batch_size = int(config.get("batch_size") or 1)
    device = str(config.get("device") or "cpu")

    try:
        torch = _torch()
        loaded = load_checkpoint(
            LoadCheckpointRequest(checkpoint_uri=str(checkpoint_path), map_location=device)
        )
        model = loaded.model.model
        model.to(torch.device(device))
        model.eval()
    except Exception as exc:  # noqa: BLE001
        _write_feature_collection(output_geojson, [])
        return _summary(
            config,
            scenes=scenes,
            status="error",
            output_geojson=output_geojson,
            started=started,
            scene_reports=[],
            failures=[{"stage": "load_checkpoint", "error": repr(exc)}],
            missing=[],
            feature_count=0,
        )

    all_features: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[str] = []

    for number, scene in enumerate(scenes, 1):
        image_path = _find_image(scene, image_index)
        if image_path is None:
            missing.append(scene)
            scene_reports.append({"scene_id": scene, "status": "missing_image", "feature_count": 0})
            continue
        scene_started = time.time()
        try:
            scene_features = _infer_scene(
                torch=torch,
                model=model,
                image_path=image_path,
                scene=scene,
                config=config,
                tile_size=tile_size,
                stride=stride,
                batch_size=batch_size,
                threshold=threshold,
                device=device,
            )
            all_features.extend(scene_features)
            _write_feature_collection(
                run_root / "per_scene" / scene / "pseudo_markup.geojson",
                scene_features,
            )
            scene_reports.append(
                {
                    "scene_id": scene,
                    "number": number,
                    "status": "ok",
                    "image": str(image_path),
                    "feature_count": len(scene_features),
                    "elapsed_sec": round(time.time() - scene_started, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"scene_id": scene, "image": str(image_path), "error": repr(exc)})
            scene_reports.append(
                {
                    "scene_id": scene,
                    "number": number,
                    "status": "failed",
                    "image": str(image_path),
                    "feature_count": 0,
                    "error": repr(exc),
                }
            )

    _write_feature_collection(output_geojson, all_features)
    status = "ok" if not failures and not missing else "partial"
    if failures and len(failures) == len(scenes):
        status = "error"
    return _summary(
        config,
        scenes=scenes,
        status=status,
        output_geojson=output_geojson,
        started=started,
        scene_reports=scene_reports,
        failures=failures,
        missing=missing,
        feature_count=len(all_features),
    )


def _infer_scene(
    *,
    torch,
    model,
    image_path: Path,
    scene: str,
    config: dict[str, Any],
    tile_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    device: str,
) -> list[dict[str, Any]]:
    del batch_size
    with rasterio.open(image_path) as dataset:
        nodata = _resolve_nodata(dataset)
        mask = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        for window in _windows(dataset.width, dataset.height, tile_size, stride):
            image = dataset.read(
                window=window,
                boundless=True,
                fill_value=nodata,
                out_shape=(dataset.count, tile_size, tile_size),
                masked=False,
            )
            if image.shape[0] > 4:
                image = image[:4]
            if image.shape[0] < 4:
                image = np.pad(image, ((0, 4 - image.shape[0]), (0, 0), (0, 0)))
            if np.all(_nodata_pixels(image, nodata)):
                continue
            tile_mask = _predict_tile(
                torch,
                model,
                image.astype(np.float32, copy=False),
                threshold=threshold,
                device=device,
            )
            crop_h = min(tile_size, dataset.height - int(window.row_off))
            crop_w = min(tile_size, dataset.width - int(window.col_off))
            y0 = int(window.row_off)
            x0 = int(window.col_off)
            mask[y0 : y0 + crop_h, x0 : x0 + crop_w] = np.maximum(
                mask[y0 : y0 + crop_h, x0 : x0 + crop_w],
                tile_mask[:crop_h, :crop_w],
            )
        return _features_from_mask(mask, dataset.transform, dataset.crs, dataset.res, scene, config)


def _predict_tile(torch, model, image: np.ndarray, *, threshold: float, device: str) -> np.ndarray:
    tensor = torch.as_tensor(image[None, :, :, :], dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        output = model(tensor)
        logits = output.logits if hasattr(output, "logits") else output
        if logits.shape[-2:] != tensor.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probs = torch.sigmoid(logits[:, :1, :, :])
        predicted = probs[0, 0].detach().cpu().numpy() >= threshold
    return predicted.astype(np.uint8)


def _features_from_mask(
    mask: np.ndarray,
    transform,
    crs,
    resolution: tuple[float, float],
    scene: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for geometry, value in rasterio_features.shapes(mask, mask=mask > 0, transform=transform):
        if int(value) != 1:
            continue
        source_crs = str(crs) if crs is not None else None
        if source_crs:
            geometry = transform_geom(source_crs, "EPSG:4326", geometry)
        output.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "_x_res": abs(float(resolution[0])),
                    "_y_res": abs(float(resolution[1])),
                    "_crs": source_crs,
                    "scene_id": scene,
                    "class_key": config.get("class_key"),
                    "class_name": config.get("class_name"),
                    "source_model": config.get("source_model"),
                    "source_run_id": config.get("mlflow_run_id"),
                    "source_checkpoint": config.get("checkpoint_uri"),
                    "source_threshold": config.get("threshold"),
                    "source_f1_score": config.get("checkpoint_f1_score"),
                    "source_epoch": config.get("checkpoint_epoch"),
                },
            }
        )
    return output


def _resolve_checkpoint(config: dict[str, Any], dst_dir: Path) -> Path:
    local = config.get("local_checkpoint_path")
    if local and Path(str(local)).is_file():
        return Path(str(local))
    run_id = config.get("mlflow_run_id")
    artifact_path = config.get("checkpoint_artifact_path") or "checkpoints/best.pt"
    if run_id:
        downloaded = download_run_artifact(
            tracking_uri=str(config["mlflow_tracking_uri"]),
            run_id=str(run_id),
            artifact_path=str(artifact_path),
            dst_dir=dst_dir,
        )
        return Path(downloaded.local_path)
    checkpoint_uri = str(config.get("checkpoint_uri") or "")
    if checkpoint_uri and Path(checkpoint_uri).is_file():
        return Path(checkpoint_uri)
    raise RuntimeError("Не задан локальный checkpoint или MLflow run id для скачивания best.pt.")


def _windows(width: int, height: int, tile_size: int, stride: int):
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            yield Window(x, y, tile_size, tile_size)


def _read_scenes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_index(images_root: Path) -> dict[str, Path]:
    files = [*images_root.rglob("*.tif"), *images_root.rglob("*.tiff")]
    return {path.stem: path for path in files}


def _find_image(scene: str, index: dict[str, Path]) -> Path | None:
    base = scene[:-4] if scene.endswith("_cog") else scene
    for candidate in (scene, base, f"{base}_cog"):
        path = index.get(candidate)
        if path is not None:
            return path
    return None


def _resolve_nodata(dataset) -> object:
    if dataset.nodata is not None:
        return dataset.nodata
    for nodata in dataset.nodatavals:
        if nodata is not None:
            return nodata
    return 0


def _nodata_pixels(image: np.ndarray, nodata: object) -> np.ndarray:
    if _is_nan(nodata):
        return np.all(np.isnan(image), axis=0)
    return np.all(image == nodata, axis=0)


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _write_feature_collection(path: Path, features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )


def _summary(
    config: dict[str, Any],
    *,
    scenes: list[str],
    status: str,
    output_geojson: Path,
    started: float,
    scene_reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    missing: list[str],
    feature_count: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "class_key": config.get("class_key"),
        "class_name": config.get("class_name"),
        "scene_count": len(scenes),
        "processed": sum(1 for item in scene_reports if item.get("status") == "ok"),
        "failed": len(failures),
        "missing_images": len(missing),
        "feature_count": feature_count,
        "output_geojson": str(output_geojson),
        "elapsed_sec": round(time.time() - started, 3),
        "source": {
            "run_id": config.get("mlflow_run_id"),
            "checkpoint": config.get("checkpoint_uri"),
            "threshold": config.get("threshold"),
            "f1_score": config.get("checkpoint_f1_score"),
            "epoch": config.get("checkpoint_epoch"),
        },
        "scenes": scene_reports,
        "failures": failures,
        "missing": missing,
    }


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Для псевдоразметки нужен установленный PyTorch.") from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
