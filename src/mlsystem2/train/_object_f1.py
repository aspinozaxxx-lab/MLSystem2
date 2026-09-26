"""Обучение границ и оценка объектов после сборки полных снимков."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mlsystem2.inference.api import create_object_scene
from mlsystem2.inference.contracts import ObjectSceneRequest, ObjectWindowPrediction
from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request

from .contracts import TrainError


def object_loss(torch, logits, masks, valid, meta, config):
    from segmentation_models_pytorch.losses import TverskyLoss

    if logits.ndim != 4 or logits.shape[1] != 3:
        raise TrainError("object f1 ожидает logits фона, объекта и границы")
    target = masks[:, 0].long()
    usable = valid[:, 0]
    if usable.any():
        ignored_target = target.masked_fill(~usable, -100)
        ce = torch.nn.functional.cross_entropy(logits[:, :2], ignored_target, ignore_index=-100,
             weight=torch.as_tensor(config.class_weights, dtype=logits.dtype, device=logits.device))
        tversky = TverskyLoss(mode="multiclass", alpha=0.75, beta=0.25, ignore_index=-100)(logits[:, :2], ignored_target)
        region = 0.25 * ce + 0.75 * tversky
    else:
        region = logits[:, :2].sum() * 0
    target_boundary = meta["boundary_target"].to(device=logits.device, dtype=logits.dtype, non_blocking=True)
    boundary_valid = meta["boundary_valid"].to(device=logits.device, dtype=torch.bool, non_blocking=True)
    prediction = logits[:, 2]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target_boundary, reduction="none")
    bce = (bce * boundary_valid).sum() / boundary_valid.sum().clamp_min(1)
    probability = prediction.sigmoid() * boundary_valid
    truth = target_boundary * boundary_valid
    dice = 1 - (2 * (probability * truth).sum() + 1e-6) / (probability.sum() + truth.sum() + 1e-6)
    boundary = 0.5 * bce + 0.5 * dice
    return region + 0.5 * boundary, region, boundary


def validate_objects(torch, model, loader, device, config, epoch, pause_controller):
    from ._trainer import _ensure_finite_tensor, _forward_logits, _prepare_supervision_masks, _prepare_valid_pixels, _threshold_metrics

    model.eval()
    scenes = {}
    total = np.zeros(3, dtype=np.float64)
    examples = 0
    overlap_pixels = 0
    pixel_counts = {"tp": 0, "fp": 0, "fn": 0}
    object_counts = {"tp": 0, "fp": 0, "fn": 0}
    per_scene = []
    with TemporaryDirectory(prefix="object-f1-validation-") as temporary:
        try:
            with torch.no_grad():
                for batch_index, (images, masks, meta) in enumerate(loader, start=1):
                    if pause_controller is not None:
                        pause_controller.pause_if_requested()
                    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                    masks, _ = _prepare_supervision_masks(torch, masks, config, device)
                    valid = _prepare_valid_pixels(torch, meta, config, device, masks)
                    _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
                    logits = _forward_logits(torch, model, images, masks)
                    loss, region, boundary = object_loss(torch, logits, masks, valid, meta, config)
                    _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
                    size = int(images.shape[-1])
                    batch_size = int(images.shape[0])
                    total += np.array([loss.item(), region.item(), boundary.item()]) * batch_size
                    examples += batch_size
                    overlap_pixels += int(meta.get("overlap_pixels", 0))
                    probabilities = torch.cat((logits[:, :2].softmax(1)[:, 1:2], logits[:, 2:3].sigmoid()), 1).cpu().numpy()
                    valid_batch = valid[:, 0].cpu().numpy()
                    truth_batch = meta["object_instances"].cpu().numpy()
                    for i, scene_id in enumerate(meta["scene_ids"]):
                        shape = meta["scene_shapes"][i]
                        width, height = int(shape["width"]), int(shape["height"])
                        if scene_id not in scenes:
                            accumulator = create_object_scene(ObjectSceneRequest(width=width, height=height, tile_size=size, work_dir=temporary))
                            truth = np.memmap(Path(temporary) / f"truth-{len(scenes)}", mode="w+", dtype="int32", shape=(height, width))
                            scenes[scene_id] = (accumulator, truth)
                        accumulator, truth = scenes[scene_id]
                        window = meta["windows"][i]
                        x, y = int(window["x"]), int(window["y"])
                        accumulator.add_window(ObjectWindowPrediction(x=x, y=y, probabilities=probabilities[i], valid_pixels=valid_batch[i]))
                        h, w = min(size, height-y), min(size, width-x)
                        truth[y:y+h, x:x+w] = np.where(valid_batch[i, :h, :w], truth_batch[i, :h, :w], 0)
                    del images, masks, valid, logits, loss, region, boundary
            if not examples:
                raise TrainError("object f1: validation не содержит окон")
            for scene_id, (accumulator, truth) in scenes.items():
                if pause_controller is not None:
                    pause_controller.pause_if_requested()
                result = accumulator.finish()
                metrics = compute_object_f1(ObjectF1Request(y_true_instances=truth, y_pred_instances=result.instances))
                counts = {"tp": metrics.true_positive, "fp": metrics.false_positive, "fn": metrics.false_negative}
                for key in counts:
                    object_counts[key] += counts[key]
                pixels = {"tp": 0, "fp": 0, "fn": 0}
                for y in range(0, truth.shape[0], 256):
                    gt = truth[y:y+256] > 0
                    pred = result.instances[y:y+256] > 0
                    valid = result.valid_pixels[y:y+256]
                    pixels["tp"] += int((gt & pred & valid).sum())
                    pixels["fp"] += int((~gt & pred & valid).sum())
                    pixels["fn"] += int((gt & ~pred & valid).sum())
                for key in pixels:
                    pixel_counts[key] += pixels[key]
                pp, pr, pf = _threshold_metrics(pixels)
                per_scene.append({"scene_id": scene_id, "object_f1": metrics.f1, "object_precision": metrics.precision,
                                  "object_recall": metrics.recall, "pixel_f1": pf, "pixel_precision": pp, "pixel_recall": pr,
                                  "predicted_objects": result.object_count, **counts})
            pp, pr, pf = _threshold_metrics(pixel_counts)
            op, ore, of = _threshold_metrics(object_counts)
            return {"loss": total[0] / examples, "region_loss": total[1] / examples, "boundary_loss": total[2] / examples,
                    "quality_f1": of, "quality_precision": op, "quality_recall": ore, "best_threshold": 0.5,
                    "best_pixel_threshold": 0.5, "best_threshold_pixel_f1": pf, "best_threshold_pixel_precision": pp,
                    "best_threshold_pixel_recall": pr, "best_threshold_precision": op, "best_threshold_recall": ore,
                    "best_threshold_object_f1": of, "best_threshold_object_precision": op, "best_threshold_object_recall": ore,
                    "per_scene_metrics": per_scene, "metric_warnings": [f"Пикселей пересечения разметки в окнах: {overlap_pixels}"] if overlap_pixels else []}
        finally:
            for accumulator, truth in scenes.values():
                truth._mmap.close()
                accumulator.close()
