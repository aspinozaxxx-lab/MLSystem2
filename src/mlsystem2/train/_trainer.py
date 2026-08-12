"""PyTorch цикл обучения сегментационной модели."""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from time import perf_counter
from typing import Any

from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request
from mlsystem2.models.api import save_checkpoint
from mlsystem2.models.contracts import SaveCheckpointRequest
from mlsystem2.tile_preparation.contracts import HARD_NEGATIVE_LABEL

from .contracts import CheckpointArtifact, EpochMetrics, TrainError, TrainProgressEvent
from .contracts import TrainProgressSink, TrainRequest, TrainResult


MAX_NONFINITE_GRADIENT_SKIPS_PER_EPOCH = 1
OBJECT_METRIC_MAX_WORKERS = 8
THRESHOLD_CANDIDATES = (0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995)
MULTICLASS_THRESHOLD_CANDIDATES = (0.0, *THRESHOLD_CANDIDATES)


def train_model(
    request: TrainRequest,
    progress_sink: TrainProgressSink | None = None,
) -> TrainResult:
    try:
        import torch
    except ImportError as exc:
        raise TrainError(
            "Для обучения требуется optional dependency torch. "
            "Установите пакет через `pip install -e .[torch]`."
        ) from exc

    model = request.model.model
    config = request.config
    device = torch.device(config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    total_started = perf_counter()
    history: list[EpochMetrics] = []
    best_score = -1.0
    best_metrics: EpochMetrics | None = None
    patience = 0
    checkpoint_dir = _checkpoint_dir(request.checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"

    try:
        for epoch in range(1, config.epochs + 1):
            _emit(progress_sink, epoch, "epoch_started", None)
            epoch_started = perf_counter()

            train_epoch = _train_epoch(
                torch,
                model,
                request.train_loader,
                optimizer,
                device,
                config,
                epoch,
            )
            val = _validate_epoch(torch, model, request.val_loader, device, config, epoch)
            scheduler.step()

            _ensure_finite_scalar(train_epoch["loss"], "train_loss", epoch)
            _ensure_finite_scalar(val["loss"], "val_loss", epoch)
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_epoch["loss"],
                val_loss=val["loss"],
                quality_metric=config.quality_metric,
                val_quality_f1=val["quality_f1"],
                val_quality_precision=val["quality_precision"],
                val_quality_recall=val["quality_recall"],
                val_best_threshold=val["best_threshold"],
                val_best_pixel_threshold=val["best_pixel_threshold"],
                val_best_threshold_pixel_f1=val["best_threshold_pixel_f1"],
                val_best_threshold_pixel_precision=val["best_threshold_pixel_precision"],
                val_best_threshold_pixel_recall=val["best_threshold_pixel_recall"],
                val_best_threshold_precision=val["best_threshold_precision"],
                val_best_threshold_recall=val["best_threshold_recall"],
                val_best_threshold_object_f1=val.get("best_threshold_object_f1"),
                val_best_threshold_object_precision=val.get(
                    "best_threshold_object_precision"
                ),
                val_best_threshold_object_recall=val.get(
                    "best_threshold_object_recall"
                ),
                val_macro_pixel_f1=val.get("macro_pixel_f1"),
                val_macro_pixel_precision=val.get("macro_pixel_precision"),
                val_macro_pixel_recall=val.get("macro_pixel_recall"),
                val_macro_pixel_iou=val.get("macro_pixel_iou"),
                val_micro_pixel_f1=val.get("micro_pixel_f1"),
                val_micro_pixel_precision=val.get("micro_pixel_precision"),
                val_micro_pixel_recall=val.get("micro_pixel_recall"),
                val_foreground_pixel_f1=val.get("foreground_pixel_f1"),
                val_foreground_pixel_precision=val.get("foreground_pixel_precision"),
                val_foreground_pixel_recall=val.get("foreground_pixel_recall"),
                val_per_class_metrics=val.get("per_class_metrics", []),
                val_multiclass_threshold_sweep=val.get("threshold_sweep", {}),
                val_metric_warnings=val.get("metric_warnings", []),
                epoch_time_sec=perf_counter() - epoch_started,
            )
            history.append(metrics)

            score = _checkpoint_score(metrics)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                patience = 0
                _save_training_checkpoint(request, str(best_checkpoint_path), metrics, "best")
            else:
                patience += 1

            _emit(progress_sink, epoch, "epoch_finished", metrics)
            if _training_time_exceeded(config, total_started):
                break
            if patience >= config.early_stopping_patience:
                break

        if not history:
            raise TrainError("Обучение не выполнило ни одной эпохи.")

        _save_training_checkpoint(request, str(final_checkpoint_path), history[-1], "final")
        return TrainResult(
            history=history,
            epochs_total=len(history),
            training_time_sec=perf_counter() - total_started,
            best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
            final_checkpoint_path=str(final_checkpoint_path),
            artifacts=[
                CheckpointArtifact(uri=str(best_checkpoint_path), label="best"),
                CheckpointArtifact(uri=str(final_checkpoint_path), label="final"),
            ],
            task=config.task,
            class_schema=list(config.class_schema),
            best_threshold=(
                best_metrics.val_best_threshold if best_metrics is not None else None
            ),
        )
    except TrainError:
        raise
    except Exception as exc:
        raise TrainError("Ошибка во время обучения модели") from exc


def _train_epoch(
    torch,
    model,
    loader: object,
    optimizer: object,
    device: object,
    config,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    batches = 0
    has_optimizer_step = False
    nonfinite_gradient_skips = 0
    for batch_index, batch in enumerate(loader, start=1):
        images, masks, _meta = _split_batch(batch, epoch, batch_index, "train")
        images = images.to(device=device, dtype=torch.float32)
        masks, hard_negative_pixels = _prepare_supervision_masks(
            torch, masks, config, device
        )
        _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "train")
        _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "train")
        if config.task == "binary":
            _validate_binary_targets(torch, masks, epoch, batch_index, "train")
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(torch, model, images, masks)
        _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "train")
        logits, masks, hard_negative_pixels = _crop_supervision_tensors(
            logits,
            masks,
            hard_negative_pixels,
            config.inference_context,
        )
        if config.task == "multiclass":
            _validate_multiclass_targets(torch, masks, logits.shape[1], epoch, batch_index, "train")
        loss = _loss(torch, logits, masks, config, hard_negative_pixels)
        _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "train")
        loss.backward()
        bad_gradient = _first_nonfinite_gradient(torch, model)
        if bad_gradient is not None:
            nonfinite_gradient_skips += 1
            warnings.warn(
                "Пропущен optimizer step из-за non-finite gradient: "
                f"epoch={epoch}, batch={batch_index}, parameter={bad_gradient}",
                stacklevel=2,
            )
            optimizer.zero_grad(set_to_none=True)
            if nonfinite_gradient_skips > MAX_NONFINITE_GRADIENT_SKIPS_PER_EPOCH:
                raise TrainError(
                    "Слишком много non-finite gradients за эпоху: "
                    f"epoch={epoch}, nonfinite_gradient_skips={nonfinite_gradient_skips}"
                )
            total_loss += float(loss.detach().item())
            batches += 1
            if (
                config.max_train_batches_per_epoch is not None
                and batch_index >= config.max_train_batches_per_epoch
            ):
                break
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _ensure_finite_tensor(torch, grad_norm, "grad_norm", epoch, batch_index, "train")
        optimizer.step()
        has_optimizer_step = True
        total_loss += float(loss.detach().item())
        batches += 1
        if (
            config.max_train_batches_per_epoch is not None
            and batch_index >= config.max_train_batches_per_epoch
        ):
            break
    if batches == 0:
        raise TrainError("Train DataLoader не вернул ни одного batch.")
    if not has_optimizer_step:
        raise TrainError(f"За эпоху {epoch} не выполнено ни одного optimizer step.")
    return {
        "loss": total_loss / batches,
    }


def _validate_epoch(
    torch,
    model,
    loader: object,
    device: object,
    config,
    epoch: int,
) -> dict[str, float | None]:
    if config.task == "multiclass":
        return _validate_multiclass_epoch(torch, model, loader, device, config, epoch)

    model.eval()
    total_loss = 0.0
    batches = 0
    threshold_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in THRESHOLD_CANDIDATES
    }
    object_threshold_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in THRESHOLD_CANDIDATES
    }
    object_instances_seen = False
    object_metric_workers = min(OBJECT_METRIC_MAX_WORKERS, max(1, os.cpu_count() or 1))
    with (
        ThreadPoolExecutor(max_workers=object_metric_workers) as object_metric_executor,
        torch.no_grad(),
    ):
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks, hard_negative_pixels = _prepare_supervision_masks(
                torch, masks, config, device
            )
            _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
            _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "val")
            _validate_binary_targets(torch, masks, epoch, batch_index, "val")
            logits = _forward_logits(torch, model, images, masks)
            _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "val")
            logits, masks, hard_negative_pixels = _crop_supervision_tensors(
                logits,
                masks,
                hard_negative_pixels,
                config.inference_context,
            )
            loss = _loss(torch, logits, masks, config, hard_negative_pixels)
            _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
            total_loss += float(loss.detach().item())
            batches += 1

            probs = torch.sigmoid(logits)
            true = masks >= 0.5
            for threshold, counts in threshold_counts.items():
                threshold_pred = probs >= threshold
                counts["tp"] += int((threshold_pred & true).sum().item())
                counts["fp"] += int((threshold_pred & ~true).sum().item())
                counts["fn"] += int((~threshold_pred & true).sum().item())
            object_instances = meta.get("object_instances") if isinstance(meta, dict) else None
            if object_instances is not None:
                object_instances_seen = True
                object_instances = _crop_spatial(
                    object_instances,
                    config.inference_context,
                )
                _accumulate_object_threshold_counts(
                    object_threshold_counts,
                    _as_numpy_instances(object_instances),
                    probs[:, 0, :, :].detach().cpu().numpy(),
                    object_metric_executor,
                )
            if (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            ):
                break

    if batches == 0:
        raise TrainError("Val DataLoader не вернул ни одного batch.")

    pixel_threshold, pixel_precision, pixel_recall, pixel_f1 = _best_threshold_metrics(
        threshold_counts
    )
    if object_instances_seen:
        object_threshold, object_precision, object_recall, object_f1 = _best_threshold_metrics(
            object_threshold_counts
        )
    else:
        object_threshold = object_precision = object_recall = object_f1 = None
    if config.quality_metric == "objects":
        if object_f1 is None or object_threshold is None:
            raise TrainError(
                "Для объектовой метрики val loader должен передавать маски экземпляров объектов."
            )
        quality_threshold = object_threshold
        quality_precision = float(object_precision)
        quality_recall = float(object_recall)
        quality_f1 = float(object_f1)
    else:
        quality_threshold = pixel_threshold
        quality_precision = pixel_precision
        quality_recall = pixel_recall
        quality_f1 = pixel_f1
    return {
        "loss": total_loss / batches,
        "best_threshold": quality_threshold,
        "best_pixel_threshold": pixel_threshold,
        "best_threshold_pixel_f1": pixel_f1,
        "best_threshold_pixel_precision": pixel_precision,
        "best_threshold_pixel_recall": pixel_recall,
        "best_threshold_precision": quality_precision,
        "best_threshold_recall": quality_recall,
        "best_threshold_object_f1": object_f1,
        "best_threshold_object_precision": object_precision,
        "best_threshold_object_recall": object_recall,
        "quality_f1": quality_f1,
        "quality_precision": quality_precision,
        "quality_recall": quality_recall,
    }


def _accumulate_object_threshold_counts(
    threshold_counts: dict[float, dict[str, int]],
    true_instances: Any,
    probabilities: Any,
    executor: ThreadPoolExecutor,
) -> None:
    tasks: list[tuple[float, Any, Any]] = []
    for threshold in threshold_counts:
        predicted = probabilities >= threshold
        tasks.extend(
            (threshold, true_instances[tile_index], predicted[tile_index])
            for tile_index in range(predicted.shape[0])
        )

    for threshold, true_positive, false_positive, false_negative in executor.map(
        _compute_object_threshold_counts,
        tasks,
    ):
        counts = threshold_counts[threshold]
        counts["tp"] += true_positive
        counts["fp"] += false_positive
        counts["fn"] += false_negative


def _compute_object_threshold_counts(
    task: tuple[float, Any, Any],
) -> tuple[float, int, int, int]:
    threshold, true_instances, predicted = task
    result = compute_object_f1(
        ObjectF1Request(
            y_true_instances=true_instances,
            y_pred_mask=predicted,
        )
    )
    return (
        threshold,
        result.true_positive,
        result.false_positive,
        result.false_negative,
    )


def _validate_multiclass_epoch(
    torch,
    model,
    loader: object,
    device: object,
    config,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    threshold_stats: dict[float, dict[int, dict[str, int]]] = {
        threshold: {} for threshold in MULTICLASS_THRESHOLD_CANDIDATES
    }
    foreground_stats: dict[float, dict[str, int]] = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in MULTICLASS_THRESHOLD_CANDIDATES
    }
    expected_num_classes = len(config.class_schema) + 1

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, _meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks, hard_negative_pixels = _prepare_supervision_masks(
                torch, masks, config, device
            )
            _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
            _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "val")
            logits = _forward_logits(torch, model, images, masks)
            _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "val")
            logits, masks, hard_negative_pixels = _crop_supervision_tensors(
                logits,
                masks,
                hard_negative_pixels,
                config.inference_context,
            )
            num_classes = int(logits.shape[1])
            _validate_multiclass_targets(torch, masks, num_classes, epoch, batch_index, "val")
            if num_classes != expected_num_classes:
                raise TrainError(
                    "Число каналов multiclass-модели не соответствует class_schema: "
                    f"ожидается {expected_num_classes}, получено {num_classes}."
                )
            loss = _loss(torch, logits, masks, config, hard_negative_pixels)
            _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
            total_loss += float(loss.detach().item())
            batches += 1

            probabilities = torch.softmax(logits, dim=1)
            confidence, raw_labels = torch.max(probabilities, dim=1)
            for threshold in MULTICLASS_THRESHOLD_CANDIDATES:
                labels = torch.where(
                    (raw_labels > 0) & (confidence < threshold),
                    torch.zeros_like(raw_labels),
                    raw_labels,
                )
                stats_by_class = threshold_stats[threshold]
                for class_id in range(1, num_classes):
                    predicted = labels == class_id
                    expected = masks == class_id
                    stats = stats_by_class.setdefault(
                        class_id,
                        {"tp": 0, "fp": 0, "fn": 0, "support": 0, "predicted": 0},
                    )
                    stats["tp"] += int((predicted & expected).sum().item())
                    stats["fp"] += int((predicted & ~expected).sum().item())
                    stats["fn"] += int((~predicted & expected).sum().item())
                    stats["support"] += int(expected.sum().item())
                    stats["predicted"] += int(predicted.sum().item())
                predicted_foreground = labels > 0
                expected_foreground = masks > 0
                foreground = foreground_stats[threshold]
                foreground["tp"] += int(
                    (predicted_foreground & expected_foreground).sum().item()
                )
                foreground["fp"] += int(
                    (predicted_foreground & ~expected_foreground).sum().item()
                )
                foreground["fn"] += int(
                    (~predicted_foreground & expected_foreground).sum().item()
                )

            if (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            ):
                break

    if batches == 0:
        raise TrainError("Val DataLoader не вернул ни одного batch.")

    evaluated: dict[float, dict[str, Any]] = {}
    for threshold in MULTICLASS_THRESHOLD_CANDIDATES:
        evaluated[threshold] = _multiclass_metrics(
            threshold_stats[threshold],
            foreground_stats[threshold],
            config,
        )
    best_threshold = max(
        MULTICLASS_THRESHOLD_CANDIDATES,
        key=lambda threshold: (
            evaluated[threshold]["macro_f1"],
            evaluated[threshold]["macro_precision"],
            threshold,
        ),
    )
    best = evaluated[best_threshold]

    return {
        "loss": total_loss / batches,
        "best_threshold": best_threshold,
        "best_pixel_threshold": best_threshold,
        "best_threshold_pixel_f1": best["macro_f1"],
        "best_threshold_pixel_precision": best["macro_precision"],
        "best_threshold_pixel_recall": best["macro_recall"],
        "best_threshold_precision": best["macro_precision"],
        "best_threshold_recall": best["macro_recall"],
        "best_threshold_object_f1": None,
        "best_threshold_object_precision": None,
        "best_threshold_object_recall": None,
        "quality_f1": best["macro_f1"],
        "quality_precision": best["macro_precision"],
        "quality_recall": best["macro_recall"],
        "macro_pixel_f1": best["macro_f1"],
        "macro_pixel_precision": best["macro_precision"],
        "macro_pixel_recall": best["macro_recall"],
        "macro_pixel_iou": best["macro_iou"],
        "micro_pixel_f1": best["micro_f1"],
        "micro_pixel_precision": best["micro_precision"],
        "micro_pixel_recall": best["micro_recall"],
        "foreground_pixel_f1": best["foreground_f1"],
        "foreground_pixel_precision": best["foreground_precision"],
        "foreground_pixel_recall": best["foreground_recall"],
        "per_class_metrics": best["per_class"],
        "metric_warnings": best["warnings"],
        "threshold_sweep": {
            str(threshold): {
                "macro_f1": values["macro_f1"],
                "macro_precision": values["macro_precision"],
                "macro_recall": values["macro_recall"],
                "macro_iou": values["macro_iou"],
                "micro_f1": values["micro_f1"],
                "foreground_f1": values["foreground_f1"],
            }
            for threshold, values in evaluated.items()
        },
    }


def _multiclass_metrics(
    class_stats: dict[int, dict[str, int]],
    foreground_stats: dict[str, int],
    config,
) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []
    warnings_list: list[str] = []
    relevant: list[dict[str, Any]] = []
    schema_by_id = {item.id: item for item in config.class_schema}
    for class_id in sorted(class_stats):
        stats = class_stats[class_id]
        schema = schema_by_id[class_id]
        support = stats["support"]
        predicted = stats["predicted"]
        if support == 0 and predicted == 0:
            precision = recall = f1 = iou = None
            warnings_list.append(
                f"Класс {schema.slug} отсутствует и в разметке, и в предсказаниях validation."
            )
        else:
            precision = _safe_div(stats["tp"], stats["tp"] + stats["fp"])
            recall = _safe_div(stats["tp"], stats["tp"] + stats["fn"])
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
            iou = _safe_div(stats["tp"], stats["tp"] + stats["fp"] + stats["fn"])
        item = {
            "id": class_id,
            "slug": schema.slug,
            "name": schema.name,
            "color": schema.color,
            "priority": schema.priority,
            "true_positive": stats["tp"],
            "false_positive": stats["fp"],
            "false_negative": stats["fn"],
            "support_pixels": support,
            "predicted_pixels": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
        per_class.append(item)
        if support > 0 or predicted > 0:
            relevant.append(item)

    macro_precision = _mean_metric(relevant, "precision")
    macro_recall = _mean_metric(relevant, "recall")
    macro_f1 = _mean_metric(relevant, "f1")
    macro_iou = _mean_metric(relevant, "iou")
    total_tp = sum(item["true_positive"] for item in per_class)
    total_fp = sum(item["false_positive"] for item in per_class)
    total_fn = sum(item["false_negative"] for item in per_class)
    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(
        2.0 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )
    foreground_precision = _safe_div(
        foreground_stats["tp"],
        foreground_stats["tp"] + foreground_stats["fp"],
    )
    foreground_recall = _safe_div(
        foreground_stats["tp"],
        foreground_stats["tp"] + foreground_stats["fn"],
    )
    foreground_f1 = _safe_div(
        2.0 * foreground_precision * foreground_recall,
        foreground_precision + foreground_recall,
    )
    return {
        "per_class": per_class,
        "warnings": warnings_list,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "foreground_precision": foreground_precision,
        "foreground_recall": foreground_recall,
        "foreground_f1": foreground_f1,
    }


def _mean_metric(values: list[dict[str, Any]], key: str) -> float:
    defined = [float(item[key]) for item in values if item.get(key) is not None]
    return sum(defined) / len(defined) if defined else 0.0


def _split_batch(batch: object, epoch: int, batch_index: int, stage: str):
    try:
        batch_len = len(batch)
    except TypeError as exc:
        raise TrainError(
            f"Некорректный batch at stage={stage}, epoch={epoch}, batch={batch_index}: "
            "ожидался batch длины 2 или 3."
        ) from exc

    if batch_len == 2:
        images, masks = batch
        return images, masks, {}
    if batch_len == 3:
        images, masks, meta = batch
        return images, masks, meta
    raise TrainError(
        f"Некорректный batch at stage={stage}, epoch={epoch}, batch={batch_index}: "
        f"ожидалась длина 2 или 3, получено {batch_len}."
    )


def _forward_logits(torch, model, images, masks):
    outputs = model(images)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


def _crop_supervision_tensors(logits, masks, hard_negative_pixels, context: int):
    if context == 0:
        return logits, masks, hard_negative_pixels
    return (
        _crop_spatial(logits, context),
        _crop_spatial(masks, context),
        _crop_spatial(hard_negative_pixels, context),
    )


def _crop_spatial(value, context: int):
    if context == 0:
        return value
    height = int(value.shape[-2])
    width = int(value.shape[-1])
    if height <= 2 * context or width <= 2 * context:
        raise TrainError("Размер supervision mask должен быть больше удвоенного context")
    return value[..., context : height - context, context : width - context]


def _prepare_supervision_masks(torch, masks, config, device):
    if config.task == "multiclass":
        raw = masks.to(device=device)
        if raw.ndim == 4 and raw.shape[1] == 1:
            raw = raw[:, 0, :, :]
        raw = raw.to(dtype=torch.long)
        hard_negative_pixels = raw == HARD_NEGATIVE_LABEL
        target = torch.where(hard_negative_pixels, torch.zeros_like(raw), raw)
        return target, hard_negative_pixels
    raw = masks.to(device=device, dtype=torch.float32)
    hard_negative_pixels = raw == float(HARD_NEGATIVE_LABEL)
    target = torch.where(hard_negative_pixels, torch.zeros_like(raw), raw)
    return target, hard_negative_pixels


def _loss(
    torch,
    logits,
    masks,
    config,
    hard_negative_pixels=None,
):
    if config.task == "multiclass":
        if config.loss not in {"cross_entropy", "cross_entropy_dice"}:
            raise TrainError(
                "multiclass train поддерживает только loss=cross_entropy или cross_entropy_dice"
            )
        weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
        cross_entropy = _weighted_mean(
            torch.nn.functional.cross_entropy(logits, masks, reduction="none"),
            weights,
        )
        if config.loss == "cross_entropy_dice":
            return cross_entropy + _multiclass_dice_loss(
                torch,
                logits,
                masks,
                hard_negative_pixels,
                config,
            )
        return cross_entropy
    if config.loss == "bce_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
            reduction="none",
        )
        return _weighted_mean(bce, weights) + _dice_loss(
            torch,
            logits,
            masks,
            hard_negative_pixels,
            config,
        )
    if config.loss == "focal_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = config.focal_alpha * torch.pow(1.0 - pt, 2.0) * bce
        return _weighted_mean(focal, weights) + _dice_loss(
            torch,
            logits,
            masks,
            hard_negative_pixels,
            config,
        )
    if config.loss == "focal_tversky":
        weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
        focal, _bce = _focal_loss_with_bce(
            torch, logits, masks, config, weights
        )
        return focal + _tversky_loss(
            torch,
            logits,
            masks,
            config,
            hard_negative_pixels,
        )
    raise TrainError(f"Неподдерживаемый loss: {config.loss}")


def _best_threshold_metrics(
    threshold_counts: dict[float, dict[str, int]],
) -> tuple[float, float, float, float]:
    best_threshold = THRESHOLD_CANDIDATES[0]
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    for threshold in THRESHOLD_CANDIDATES:
        counts = threshold_counts[threshold]
        precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        if f1 > best_f1:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_threshold, best_precision, best_recall, max(best_f1, 0.0)


def _pixel_loss_weights(
    torch,
    logits,
    hard_negative_pixels,
    config,
):
    hard_negative_weight = float(getattr(config, "hard_negative_weight", 1.0))
    has_hard_negative_weights = (
        hard_negative_weight != 1.0
        and hard_negative_pixels is not None
        and hasattr(hard_negative_pixels, "to")
    )
    if not has_hard_negative_weights:
        return None
    return 1.0 + (hard_negative_weight - 1.0) * hard_negative_pixels.to(
        device=logits.device,
        dtype=logits.dtype,
    )


def _weighted_mean(values, weights):
    if weights is None:
        return values.mean()
    return (values * weights).mean()


def _focal_loss_with_bce(
    torch,
    logits,
    masks,
    config,
    weights=None,
):
    pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        masks,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(masks > 0.5, probs, 1.0 - probs)
    focal = torch.pow((1.0 - pt).clamp_min(0.0), 2.0) * bce

    alpha = config.focal_alpha
    if alpha is not None:
        alpha_factor = torch.where(
            masks > 0.5,
            torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1.0 - alpha, device=logits.device, dtype=logits.dtype),
        )
        focal = alpha_factor * focal
    return (
        _weighted_mean(focal, weights),
        _weighted_mean(bce, weights),
    )


def _dice_loss(
    torch,
    logits,
    masks,
    hard_negative_pixels=None,
    config=None,
):
    probs = torch.sigmoid(logits)
    probability_weights = _pixel_loss_weights(
        torch, logits, hard_negative_pixels, config
    )
    if probability_weights is not None:
        probs = probs * probability_weights
    smooth = 1.0
    intersection = torch.sum(probs * masks)
    denominator = torch.sum(probs) + torch.sum(masks)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def _multiclass_dice_loss(
    torch,
    logits,
    masks,
    hard_negative_pixels=None,
    config=None,
):
    probs = torch.softmax(logits, dim=1)
    num_classes = int(logits.shape[1])
    if num_classes <= 1:
        return logits.sum() * 0.0
    target = torch.nn.functional.one_hot(
        masks.clamp(min=0, max=num_classes - 1),
        num_classes=num_classes,
    )
    target = target.permute(0, 3, 1, 2).to(device=logits.device, dtype=probs.dtype)
    probs = probs[:, 1:, :, :]
    target = target[:, 1:, :, :]
    probability_weights = _pixel_loss_weights(
        torch, logits, hard_negative_pixels, config
    )
    if probability_weights is not None:
        probs = probs * probability_weights.unsqueeze(1)
    smooth = 1.0
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def _tversky_loss(
    torch,
    logits,
    masks,
    config,
    hard_negative_pixels=None,
):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    true_positive = torch.sum(probs * masks)
    false_positive_pixels = probs * (1.0 - masks)
    weights = _pixel_loss_weights(
        torch, logits, hard_negative_pixels, config
    )
    if weights is not None:
        false_positive_pixels = false_positive_pixels * weights
    false_positive = torch.sum(false_positive_pixels)
    false_negative = torch.sum((1.0 - probs) * masks)
    tversky = (true_positive + smooth) / (
        true_positive
        + config.tversky_alpha * false_positive
        + config.tversky_beta * false_negative
        + smooth
    )
    return 1.0 - tversky


def _ensure_finite_tensor(
    torch,
    tensor,
    name: str,
    epoch: int,
    batch_index: int,
    stage: str,
) -> None:
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(tensor):
        return
    if not bool(torch.isfinite(tensor).all()):
        raise TrainError(
            f"Non-finite tensor at stage={stage}, epoch={epoch}, batch={batch_index}, tensor={name}"
        )


def _ensure_finite_scalar(value: float, name: str, epoch: int) -> None:
    if not isfinite(value):
        raise TrainError(f"Non-finite metric at epoch={epoch}, metric={name}, value={value}")


def _first_nonfinite_gradient(torch, model) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            return name
    return None


def _save_training_checkpoint(
    request: TrainRequest,
    checkpoint_uri: str,
    metrics: EpochMetrics,
    label: str,
) -> None:
    save_checkpoint(
        SaveCheckpointRequest(
            model=request.model,
            checkpoint_uri=checkpoint_uri,
            metadata={
                "label": label,
                "epoch": metrics.epoch,
                "quality_metric": metrics.quality_metric,
                "val_quality_f1": metrics.val_quality_f1,
                "val_quality_precision": metrics.val_quality_precision,
                "val_quality_recall": metrics.val_quality_recall,
                "val_best_threshold": metrics.val_best_threshold,
                "val_best_pixel_threshold": metrics.val_best_pixel_threshold,
                "val_best_threshold_pixel_f1": metrics.val_best_threshold_pixel_f1,
                "val_best_threshold_pixel_precision": metrics.val_best_threshold_pixel_precision,
                "val_best_threshold_pixel_recall": metrics.val_best_threshold_pixel_recall,
                "val_best_threshold_precision": metrics.val_best_threshold_precision,
                "val_best_threshold_recall": metrics.val_best_threshold_recall,
                "val_best_threshold_object_f1": metrics.val_best_threshold_object_f1,
                "val_best_threshold_object_precision": metrics.val_best_threshold_object_precision,
                "val_best_threshold_object_recall": metrics.val_best_threshold_object_recall,
                "task": request.config.task,
                "class_schema": [
                    item.model_dump(mode="json") for item in request.config.class_schema
                ],
                "confidence_threshold": metrics.val_best_threshold,
                "val_macro_pixel_f1": metrics.val_macro_pixel_f1,
                "val_macro_pixel_precision": metrics.val_macro_pixel_precision,
                "val_macro_pixel_recall": metrics.val_macro_pixel_recall,
                "val_macro_pixel_iou": metrics.val_macro_pixel_iou,
                "val_micro_pixel_f1": metrics.val_micro_pixel_f1,
                "val_foreground_pixel_f1": metrics.val_foreground_pixel_f1,
                "val_per_class_metrics": metrics.val_per_class_metrics,
                "val_multiclass_threshold_sweep": metrics.val_multiclass_threshold_sweep,
                "val_metric_warnings": metrics.val_metric_warnings,
                "val_loss": metrics.val_loss,
                "train_loss": metrics.train_loss,
                "sample_size": request.sample_size,
                "inference_context": request.config.inference_context,
                "inference_core_size": (
                    request.sample_size - 2 * request.config.inference_context
                    if request.sample_size is not None
                    else None
                ),
                "seed": request.config.seed,
                "train_config": request.config.model_dump(mode="json"),
            },
        )
    )


def _checkpoint_score(metrics: EpochMetrics) -> float:
    if metrics.val_macro_pixel_f1 is not None:
        return metrics.val_macro_pixel_f1
    return metrics.val_quality_f1


def _as_numpy_instances(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    import numpy as np

    return np.asarray(value)


def _checkpoint_dir(path: str):
    from pathlib import Path

    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _training_time_exceeded(config, total_started: float) -> bool:
    max_training_time_sec = getattr(config, "max_training_time_sec", None)
    if max_training_time_sec is None:
        return False
    return perf_counter() - total_started >= max_training_time_sec


def _emit(
    progress_sink: TrainProgressSink | None,
    epoch: int,
    message: str,
    metrics: EpochMetrics | None,
) -> None:
    if progress_sink is not None:
        progress_sink(TrainProgressEvent(epoch=epoch, message=message, metrics=metrics))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _validate_multiclass_targets(
    torch,
    masks,
    num_classes: int,
    epoch: int,
    batch_index: int,
    stage: str,
) -> None:
    if masks.ndim != 3:
        raise TrainError(
            f"Некорректная multiclass mask at stage={stage}, epoch={epoch}, "
            f"batch={batch_index}: ожидалась форма [B,H,W], получено {tuple(masks.shape)}"
        )
    if int(masks.min().item()) < 0 or int(masks.max().item()) >= num_classes:
        raise TrainError(
            f"Некорректные значения multiclass mask at stage={stage}, epoch={epoch}, "
            f"batch={batch_index}: ожидается диапазон 0..{num_classes - 1}"
        )


def _validate_binary_targets(
    torch,
    masks,
    epoch: int,
    batch_index: int,
    stage: str,
) -> None:
    del torch
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise TrainError(
            f"Некорректная binary mask at stage={stage}, epoch={epoch}, "
            f"batch={batch_index}: ожидалась форма [B,1,H,W], получено {tuple(masks.shape)}"
        )
    min_value = float(masks.min().item())
    max_value = float(masks.max().item())
    if min_value < 0.0 or max_value > 1.0:
        raise TrainError(
            f"Некорректные значения binary mask at stage={stage}, epoch={epoch}, "
            "ожидается диапазон 0..1 после decode supervision mask"
        )
