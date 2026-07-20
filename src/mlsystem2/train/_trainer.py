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
                epoch_time_sec=perf_counter() - epoch_started,
            )
            history.append(metrics)

            score = _checkpoint_score(metrics)
            if score > best_score:
                best_score = score
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
        masks, hard_negative_pixels = _prepare_supervision_masks(torch, masks, config, device)
        _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "train")
        _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "train")
        if config.task == "binary":
            _validate_binary_targets(torch, masks, epoch, batch_index, "train")
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(torch, model, images, masks)
        _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "train")
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
            masks, hard_negative_pixels = _prepare_supervision_masks(torch, masks, config, device)
            _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
            _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "val")
            _validate_binary_targets(torch, masks, epoch, batch_index, "val")
            logits = _forward_logits(torch, model, images, masks)
            _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "val")
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
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    batches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, _meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks, hard_negative_pixels = _prepare_supervision_masks(torch, masks, config, device)
            _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
            _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "val")
            logits = _forward_logits(torch, model, images, masks)
            _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "val")
            num_classes = int(logits.shape[1])
            _validate_multiclass_targets(torch, masks, num_classes, epoch, batch_index, "val")
            loss = _loss(torch, logits, masks, config, hard_negative_pixels)
            _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
            total_loss += float(loss.detach().item())
            batches += 1

            preds = torch.argmax(logits, dim=1)
            pred_foreground = preds > 0
            true_foreground = masks > 0
            true_positive += int((pred_foreground & true_foreground).sum().item())
            false_positive += int((pred_foreground & ~true_foreground).sum().item())
            false_negative += int((~pred_foreground & true_foreground).sum().item())

            if (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            ):
                break

    if batches == 0:
        raise TrainError("Val DataLoader не вернул ни одного batch.")

    foreground_precision = _safe_div(true_positive, true_positive + false_positive)
    foreground_recall = _safe_div(true_positive, true_positive + false_negative)
    foreground_f1 = _safe_div(
        2.0 * foreground_precision * foreground_recall,
        foreground_precision + foreground_recall,
    )

    return {
        "loss": total_loss / batches,
        "best_threshold": 0.0,
        "best_pixel_threshold": 0.0,
        "best_threshold_pixel_f1": foreground_f1,
        "best_threshold_pixel_precision": foreground_precision,
        "best_threshold_pixel_recall": foreground_recall,
        "best_threshold_precision": foreground_precision,
        "best_threshold_recall": foreground_recall,
        "best_threshold_object_f1": None,
        "best_threshold_object_precision": None,
        "best_threshold_object_recall": None,
        "quality_f1": foreground_f1,
        "quality_precision": foreground_precision,
        "quality_recall": foreground_recall,
    }


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


def _loss(torch, logits, masks, config, hard_negative_pixels=None):
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
        focal, _bce = _focal_loss_with_bce(torch, logits, masks, config, weights)
        return focal + _tversky_loss(torch, logits, masks, config, hard_negative_pixels)
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


def _pixel_loss_weights(torch, logits, hard_negative_pixels, config):
    del torch
    hard_negative_weight = float(getattr(config, "hard_negative_weight", 1.0))
    if (
        hard_negative_weight == 1.0
        or hard_negative_pixels is None
        or not hasattr(hard_negative_pixels, "to")
    ):
        return None
    return 1.0 + (
        hard_negative_weight - 1.0
    ) * hard_negative_pixels.to(device=logits.device, dtype=logits.dtype)


def _weighted_mean(values, weights):
    if weights is None:
        return values.mean()
    return (values * weights).mean()


def _focal_loss_with_bce(torch, logits, masks, config, weights=None):
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
    return _weighted_mean(focal, weights), _weighted_mean(bce, weights)


def _dice_loss(torch, logits, masks, hard_negative_pixels=None, config=None):
    probs = torch.sigmoid(logits)
    probability_weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
    if probability_weights is not None:
        probs = probs * probability_weights
    smooth = 1.0
    intersection = torch.sum(probs * masks)
    denominator = torch.sum(probs) + torch.sum(masks)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def _multiclass_dice_loss(torch, logits, masks, hard_negative_pixels=None, config=None):
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
    probability_weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
    if probability_weights is not None:
        probs = probs * probability_weights.unsqueeze(1)
    smooth = 1.0
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def _tversky_loss(torch, logits, masks, config, hard_negative_pixels=None):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    true_positive = torch.sum(probs * masks)
    false_positive_pixels = probs * (1.0 - masks)
    weights = _pixel_loss_weights(torch, logits, hard_negative_pixels, config)
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
                "val_loss": metrics.val_loss,
                "train_loss": metrics.train_loss,
                "sample_size": request.sample_size,
                "train_config": request.config.model_dump(mode="json"),
            },
        )
    )


def _checkpoint_score(metrics: EpochMetrics) -> float:
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
