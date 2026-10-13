"""Три обучаемых класса и диагностика разделения полных объектов."""

from collections import Counter
import logging
from time import perf_counter

import numpy as np

from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request


def object_v2_loss(torch, logits, masks, valid, meta, config):
    from segmentation_models_pytorch.losses import TverskyLoss

    boundary = meta["boundary_target"].to(device=logits.device, dtype=torch.bool)
    usable = valid[:, 0] & meta["boundary_valid"].to(device=logits.device, dtype=torch.bool)
    if not usable.any():
        zero = logits.sum() * 0
        return zero, zero, zero
    foreground = masks[:, 0] > 0
    target = torch.where(foreground & boundary, 2, foreground.long()).masked_fill(~usable, -100)
    ce = torch.nn.functional.cross_entropy(
        logits, target, ignore_index=-100,
        weight=torch.as_tensor(config.class_weights, dtype=logits.dtype, device=logits.device),
    )
    tversky = TverskyLoss(mode="multiclass", alpha=0.75, beta=0.25, ignore_index=-100)(logits, target)
    probability = logits.softmax(1)

    def dice(prediction, truth):
        prediction = prediction * usable
        truth = truth * usable
        return 1 - (2 * (prediction * truth).sum() + 1e-6) / (prediction.sum() + truth.sum() + 1e-6)

    # Два диагностических Dice: полнота области и качество широкой полосы.
    return 0.25 * ce + 0.75 * tversky, dice(probability[:, 1:].sum(1), foreground), dice(probability[:, 2], boundary & foreground)


def _rates(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return (tp / max(1, tp + fp), tp / max(1, tp + fn), 2 * tp / max(1, 2 * tp + fp + fn))


def _scene_metrics(truth, predicted, probabilities, valid, boundary_truth, boundary_valid, threshold):
    metric = compute_object_f1(ObjectF1Request(y_true_instances=truth, y_pred_instances=predicted))
    pixels = Counter(tp=0, fp=0, fn=0)
    edges = Counter(tp=0, fp=0, fn=0)
    true_areas, pred_areas, intersections = Counter(), Counter(), Counter()
    for y in range(0, truth.shape[0], 256):
        gt, pred, use = truth[y:y+256], predicted[y:y+256], valid[y:y+256]
        pixels.update(tp=int(((gt > 0) & (pred > 0) & use).sum()),
                      fp=int(((gt == 0) & (pred > 0) & use).sum()),
                      fn=int(((gt > 0) & (pred == 0) & use).sum()))
        edge = probabilities[1, y:y+256] >= threshold
        target = boundary_truth[y:y+256]
        supervised = boundary_valid[y:y+256] & use
        edges.update(tp=int((edge & target & supervised).sum()),
                     fp=int((edge & ~target & supervised).sum()),
                     fn=int((~edge & target & supervised).sum()))
        for values, counts in ((gt, true_areas), (pred, pred_areas)):
            ids, amounts = np.unique(values[use], return_counts=True)
            counts.update({int(i): int(n) for i, n in zip(ids, amounts) if i > 0})
        common = (gt > 0) & (pred > 0) & use
        if common.any():
            pairs, amounts = _unique_instance_pairs(gt[common], pred[common])
            intersections.update({(int(pair[0]), int(pair[1])): int(n) for pair, n in zip(pairs, amounts)})
    true_neighbors, pred_neighbors = Counter(), Counter()
    for (gt, pred), amount in intersections.items():
        # Диагностика, не подмена сопоставления IoU: значимое перекрытие обеих фигур.
        if amount >= 0.1 * true_areas[gt] and amount >= 0.1 * pred_areas[pred]:
            true_neighbors[gt] += 1
            pred_neighbors[pred] += 1
    pp, pr, pf = _rates(pixels)
    bp, br, bf = _rates(edges)
    return {
        "tp": metric.true_positive, "fp": metric.false_positive, "fn": metric.false_negative,
        "object_f1": metric.f1, "object_precision": metric.precision, "object_recall": metric.recall,
        "pixel_f1": pf, "pixel_precision": pp, "pixel_recall": pr, "pixel_counts": dict(pixels),
        "boundary_precision": bp, "boundary_recall": br, "boundary_f1": bf,
        "boundary_counts": dict(edges), "boundary_threshold": threshold,
        "true_objects": len(true_areas), "predicted_objects": len(pred_areas),
        "merged_predictions": sum(value > 1 for value in pred_neighbors.values()),
        "split_references": sum(value > 1 for value in true_neighbors.values()),
    }


def _unique_instance_pairs(truth, predicted):
    """Подсчитать пары положительных ID без структурной сортировки двух столбцов."""
    if (truth.dtype.kind in "biu" and predicted.dtype.kind in "biu"
            and truth.max(initial=0) <= 0xFFFFFFFF and predicted.max(initial=0) <= 0xFFFFFFFF):
        encoded = truth.astype(np.uint64)
        encoded <<= np.uint64(32)
        np.bitwise_or(encoded, predicted, out=encoded, dtype=np.uint64, casting="unsafe")
        keys, counts = np.unique(encoded, return_counts=True)
        pairs = np.empty((keys.size, 2), dtype=np.uint32)
        pairs[:, 0] = keys >> np.uint64(32)
        pairs[:, 1] = keys  # Приведение оставляет младшие 32 бита.
        return pairs, counts
    return np.unique(np.column_stack((truth, predicted)), axis=0, return_counts=True)


def evaluate_candidates(scene_id, accumulator, truth, boundary_truth, boundary_valid, separation, candidates, pause_controller, *, initial_result):
    # resegment меняет тот же дисковый массив instances. Сначала учитываем готовый
    # результат finish, иначе очередной порог затрёт его до расчёта метрик.
    initial_threshold = separation.boundary_threshold
    thresholds = ([initial_threshold] if initial_threshold in candidates else []) + [
        value for value in candidates if value != initial_threshold
    ]
    for threshold in thresholds:
        if pause_controller is not None:
            pause_controller.pause_if_requested()
        started = perf_counter()
        logging.getLogger(__name__).info("Снимок %s: порог границы %.3f, разделение и точные метрики", scene_id, threshold)
        result = initial_result if threshold == initial_threshold else accumulator.resegment(
            separation.model_copy(update={"boundary_threshold": threshold}))
        predicted = result.instances
        candidates[threshold].append({"scene_id": scene_id, **_scene_metrics(
            truth, predicted, result.probabilities, result.valid_pixels, boundary_truth, boundary_valid, threshold)})
        logging.getLogger(__name__).info("Снимок %s: порог %.3f проверен за %.1f с, объектов=%s",
                                        scene_id, threshold, perf_counter() - started,
                                        candidates[threshold][-1]["predicted_objects"])
        del predicted


def summarize_candidates(candidates, separation):
    sweep = []
    for threshold, rows in candidates.items():
        objects = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
        precision, recall, f1 = _rates(objects)
        sweep.append({"boundary_threshold": threshold, "object_f1": f1,
                      "object_precision": precision, "object_recall": recall, **objects})
    # При равенстве предпочитаем precision, затем порог, ближайший к штатному 0.5.
    best = max(sweep, key=lambda item: (item["object_f1"], item["object_precision"],
                                      -abs(item["boundary_threshold"] - 0.5), item["boundary_threshold"]))
    rows = candidates[best["boundary_threshold"]]
    pixels = {key: sum(row["pixel_counts"][key] for row in rows) for key in ("tp", "fp", "fn")}
    pp, pr, pf = _rates(pixels)
    chosen = separation.model_copy(update={"boundary_threshold": best["boundary_threshold"]})
    return {
        "quality_f1": best["object_f1"], "quality_precision": best["object_precision"],
        "quality_recall": best["object_recall"], "best_threshold": chosen.foreground_threshold,
        "best_pixel_threshold": chosen.foreground_threshold, "best_threshold_pixel_f1": pf,
        "best_threshold_pixel_precision": pp, "best_threshold_pixel_recall": pr,
        "best_threshold_precision": best["object_precision"], "best_threshold_recall": best["object_recall"],
        "best_threshold_object_f1": best["object_f1"], "best_threshold_object_precision": best["object_precision"],
        "best_threshold_object_recall": best["object_recall"], "per_scene_metrics": rows,
        "object_separation": chosen.model_dump(mode="json"), "object_threshold_sweep": sweep,
    }
