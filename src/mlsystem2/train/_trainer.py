"""PyTorch цикл обучения сегментационной модели."""

from __future__ import annotations

import warnings
from math import isfinite
from time import perf_counter

from mlsystem2.models.api import save_checkpoint
from mlsystem2.models.contracts import SaveCheckpointRequest

from .contracts import CheckpointArtifact, EpochMetrics, TrainError, TrainProgressEvent
from .contracts import TrainProgressSink, TrainRequest, TrainResult


MAX_SKIPPED_OPTIMIZER_STEPS_PER_EPOCH = 1
THRESHOLD_SWEEP = (0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995)
PROBABILITY_HISTOGRAM_BINS = 1000


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
    best_f1 = -1.0
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
                train_loss_focal=train_epoch["loss_focal"],
                train_loss_tversky=train_epoch["loss_tversky"],
                train_loss_bce=train_epoch["loss_bce"],
                train_loss_dice=train_epoch["loss_dice"],
                train_optimizer_steps=train_epoch["optimizer_steps"],
                train_skipped_optimizer_steps=train_epoch["skipped_optimizer_steps"],
                val_loss=val["loss"],
                val_pixel_precision=val["precision"],
                val_pixel_recall=val["recall"],
                val_pixel_f1=val["f1"],
                val_positive_pixels=val["positive_pixels"],
                val_pred_positive_pixels=val["pred_positive_pixels"],
                val_true_positive=val["true_positive"],
                val_false_positive=val["false_positive"],
                val_false_negative=val["false_negative"],
                val_best_threshold=val["best_threshold"],
                val_best_threshold_pixel_f1=val["best_threshold_pixel_f1"],
                val_best_threshold_precision=val["best_threshold_precision"],
                val_best_threshold_recall=val["best_threshold_recall"],
                val_prob_mean=val["prob_mean"],
                val_prob_min=val["prob_min"],
                val_prob_max=val["prob_max"],
                val_prob_p50=val["prob_p50"],
                val_prob_p90=val["prob_p90"],
                val_prob_p99=val["prob_p99"],
                val_prob_p999=val["prob_p999"],
                val_prob_positive_mean=val["prob_positive_mean"],
                val_prob_positive_p50=val["prob_positive_p50"],
                val_prob_positive_p90=val["prob_positive_p90"],
                val_prob_positive_p99=val["prob_positive_p99"],
                val_prob_negative_mean=val["prob_negative_mean"],
                val_prob_negative_p50=val["prob_negative_p50"],
                val_prob_negative_p90=val["prob_negative_p90"],
                val_prob_negative_p99=val["prob_negative_p99"],
                val_threshold_sweep=val["threshold_sweep"],
                val_macro_f1=val.get("macro_f1"),
                val_mean_iou=val.get("mean_iou"),
                val_pixel_accuracy=val.get("pixel_accuracy"),
                val_per_class_metrics=val.get("per_class_metrics", {}),
                epoch_time_sec=perf_counter() - epoch_started,
            )
            history.append(metrics)

            if metrics.val_pixel_f1 > best_f1:
                best_f1 = metrics.val_pixel_f1
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
) -> dict[str, float | int | None]:
    model.train()
    total_loss = 0.0
    component_totals = {
        "focal": 0.0,
        "tversky": 0.0,
        "bce": 0.0,
        "dice": 0.0,
    }
    component_counts = {name: 0 for name in component_totals}
    batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    for batch_index, batch in enumerate(loader, start=1):
        images, masks, _meta = _split_batch(batch, epoch, batch_index, "train")
        images = images.to(device=device, dtype=torch.float32)
        masks = _prepare_masks(torch, masks, config, device)
        _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "train")
        _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "train")
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(torch, model, images, masks)
        _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "train")
        if config.task == "multiclass":
            _validate_multiclass_targets(torch, masks, logits.shape[1], epoch, batch_index, "train")
        loss_info = _loss_components(torch, logits, masks, config)
        loss = loss_info["loss"]
        _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "train")
        _accumulate_loss_components(component_totals, component_counts, loss_info)
        loss.backward()
        bad_gradient = _first_nonfinite_gradient(torch, model)
        if bad_gradient is not None:
            skipped_optimizer_steps += 1
            warnings.warn(
                "Пропущен optimizer step из-за non-finite gradient: "
                f"epoch={epoch}, batch={batch_index}, parameter={bad_gradient}",
                stacklevel=2,
            )
            optimizer.zero_grad(set_to_none=True)
            if skipped_optimizer_steps > MAX_SKIPPED_OPTIMIZER_STEPS_PER_EPOCH:
                raise TrainError(
                    "Слишком много non-finite gradients за эпоху: "
                    f"epoch={epoch}, skipped_optimizer_steps={skipped_optimizer_steps}"
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
        optimizer_steps += 1
        total_loss += float(loss.detach().item())
        batches += 1
        if (
            config.max_train_batches_per_epoch is not None
            and batch_index >= config.max_train_batches_per_epoch
        ):
            break
    if batches == 0:
        raise TrainError("Train DataLoader не вернул ни одного batch.")
    if optimizer_steps == 0:
        raise TrainError(f"За эпоху {epoch} не выполнено ни одного optimizer step.")
    return {
        "loss": total_loss / batches,
        "loss_focal": _average_component(component_totals, component_counts, "focal"),
        "loss_tversky": _average_component(component_totals, component_counts, "tversky"),
        "loss_bce": _average_component(component_totals, component_counts, "bce"),
        "loss_dice": _average_component(component_totals, component_counts, "dice"),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
    }


def _validate_epoch(
    torch,
    model,
    loader: object,
    device: object,
    config,
    epoch: int,
) -> dict[str, float | int]:
    if config.task == "multiclass":
        return _validate_multiclass_epoch(torch, model, loader, device, config, epoch)

    model.eval()
    total_loss = 0.0
    batches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    positive_pixels = 0
    pred_positive_pixels = 0
    sweep_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in THRESHOLD_SWEEP
    }
    prob_stats = _ProbabilityStats(torch, "prob")
    positive_prob_stats = _ProbabilityStats(torch, "prob_positive")
    negative_prob_stats = _ProbabilityStats(torch, "prob_negative")
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, _meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks = _prepare_masks(torch, masks, config, device)
            _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
            _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "val")
            logits = _forward_logits(torch, model, images, masks)
            _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "val")
            loss = _loss(torch, logits, masks, config)
            _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
            total_loss += float(loss.detach().item())
            batches += 1

            probs = torch.sigmoid(logits)
            pred = probs >= config.threshold
            true = masks >= 0.5
            prob_stats.update(probs)
            positive_prob_stats.update(probs, true)
            negative_prob_stats.update(probs, ~true)
            positive_pixels += int(true.sum().item())
            pred_positive_pixels += int(pred.sum().item())
            true_positive += int((pred & true).sum().item())
            false_positive += int((pred & ~true).sum().item())
            false_negative += int((~pred & true).sum().item())
            for threshold, counts in sweep_counts.items():
                threshold_pred = probs >= threshold
                counts["tp"] += int((threshold_pred & true).sum().item())
                counts["fp"] += int((threshold_pred & ~true).sum().item())
                counts["fn"] += int((~threshold_pred & true).sum().item())
            if (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            ):
                break

    if batches == 0:
        raise TrainError("Val DataLoader не вернул ни одного batch.")

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    threshold_sweep = _threshold_sweep_metrics(sweep_counts)
    best_threshold, best_precision, best_recall, best_f1 = _best_threshold_metrics(threshold_sweep)
    prob_snapshot = {
        **prob_stats.snapshot(),
        **positive_prob_stats.snapshot(),
        **negative_prob_stats.snapshot(),
    }
    return {
        "loss": total_loss / batches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_pixels": positive_pixels,
        "pred_positive_pixels": pred_positive_pixels,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "best_threshold": best_threshold,
        "best_threshold_pixel_f1": best_f1,
        "best_threshold_precision": best_precision,
        "best_threshold_recall": best_recall,
        "threshold_sweep": threshold_sweep,
        **prob_snapshot,
    }


def _validate_multiclass_epoch(
    torch,
    model,
    loader: object,
    device: object,
    config,
    epoch: int,
) -> dict[str, float | int | dict[str, dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    batches = 0
    num_classes = 0
    class_stats: dict[int, dict[str, int]] = {}
    correct_pixels = 0
    total_pixels = 0
    positive_pixels = 0
    pred_positive_pixels = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, _meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks = _prepare_masks(torch, masks, config, device)
            _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "val")
            _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "val")
            logits = _forward_logits(torch, model, images, masks)
            _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "val")
            num_classes = int(logits.shape[1])
            _validate_multiclass_targets(torch, masks, num_classes, epoch, batch_index, "val")
            loss = _loss(torch, logits, masks, config)
            _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
            total_loss += float(loss.detach().item())
            batches += 1

            preds = torch.argmax(logits, dim=1)
            correct_pixels += int((preds == masks).sum().item())
            total_pixels += int(masks.numel())
            positive_pixels += int((masks > 0).sum().item())
            pred_positive_pixels += int((preds > 0).sum().item())
            for class_id in range(1, num_classes):
                stats = class_stats.setdefault(
                    class_id,
                    {"tp": 0, "fp": 0, "fn": 0, "support_pixels": 0, "pred_pixels": 0},
                )
                pred_class = preds == class_id
                true_class = masks == class_id
                stats["tp"] += int((pred_class & true_class).sum().item())
                stats["fp"] += int((pred_class & ~true_class).sum().item())
                stats["fn"] += int((~pred_class & true_class).sum().item())
                stats["support_pixels"] += int(true_class.sum().item())
                stats["pred_pixels"] += int(pred_class.sum().item())

            if (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            ):
                break

    if batches == 0:
        raise TrainError("Val DataLoader не вернул ни одного batch.")

    per_class_metrics = _multiclass_per_class_metrics(class_stats, config)
    foreground_keys = [
        slug
        for slug, values in per_class_metrics.items()
        if values["support_pixels"] > 0
    ]
    macro_f1 = _mean([per_class_metrics[key]["f1"] for key in foreground_keys])
    macro_precision = _mean([per_class_metrics[key]["precision"] for key in foreground_keys])
    macro_recall = _mean([per_class_metrics[key]["recall"] for key in foreground_keys])
    mean_iou = _mean([per_class_metrics[key]["iou"] for key in foreground_keys])
    true_positive = sum(stats["tp"] for stats in class_stats.values())
    false_positive = sum(stats["fp"] for stats in class_stats.values())
    false_negative = sum(stats["fn"] for stats in class_stats.values())

    return {
        "loss": total_loss / batches,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "positive_pixels": positive_pixels,
        "pred_positive_pixels": pred_positive_pixels,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "best_threshold": 0.0,
        "best_threshold_pixel_f1": macro_f1,
        "best_threshold_precision": macro_precision,
        "best_threshold_recall": macro_recall,
        "threshold_sweep": {},
        "prob_mean": 0.0,
        "prob_min": 0.0,
        "prob_max": 0.0,
        "prob_p50": 0.0,
        "prob_p90": 0.0,
        "prob_p99": 0.0,
        "prob_p999": 0.0,
        "prob_positive_mean": 0.0,
        "prob_positive_p50": 0.0,
        "prob_positive_p90": 0.0,
        "prob_positive_p99": 0.0,
        "prob_negative_mean": 0.0,
        "prob_negative_p50": 0.0,
        "prob_negative_p90": 0.0,
        "prob_negative_p99": 0.0,
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
        "pixel_accuracy": _safe_div(correct_pixels, total_pixels),
        "per_class_metrics": per_class_metrics,
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


def _prepare_masks(torch, masks, config, device):
    if config.task == "multiclass":
        prepared = masks.to(device=device)
        if prepared.ndim == 4 and prepared.shape[1] == 1:
            prepared = prepared[:, 0, :, :]
        return prepared.to(dtype=torch.long)
    return masks.to(device=device, dtype=torch.float32)


def _loss(torch, logits, masks, config):
    return _loss_components(torch, logits, masks, config)["loss"]


def _loss_components(torch, logits, masks, config) -> dict[str, object]:
    if config.task == "multiclass":
        if config.loss not in {"cross_entropy", "cross_entropy_dice"}:
            raise TrainError(
                "multiclass train поддерживает только loss=cross_entropy или cross_entropy_dice"
            )
        cross_entropy = torch.nn.functional.cross_entropy(logits, masks)
        if config.loss == "cross_entropy_dice":
            dice = _multiclass_dice_loss(torch, logits, masks)
            loss = cross_entropy + dice
        else:
            dice = None
            loss = cross_entropy
        return {
            "loss": loss,
            "focal": None,
            "tversky": None,
            "bce": None,
            "dice": dice,
        }
    if config.loss == "bce_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
        )
        dice = _dice_loss(torch, logits, masks)
        return {
            "loss": bce + dice,
            "focal": None,
            "tversky": None,
            "bce": bce,
            "dice": dice,
        }
    if config.loss == "focal_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = (config.focal_alpha * torch.pow(1.0 - pt, 2.0) * bce).mean()
        dice = _dice_loss(torch, logits, masks)
        return {
            "loss": focal + dice,
            "focal": focal,
            "tversky": None,
            "bce": bce.mean(),
            "dice": dice,
        }
    if config.loss == "focal_tversky":
        focal, bce = _focal_loss_with_bce(torch, logits, masks, config)
        tversky = _tversky_loss(torch, logits, masks, config)
        return {
            "loss": focal + tversky,
            "focal": focal,
            "tversky": tversky,
            "bce": bce,
            "dice": None,
        }
    raise TrainError(f"Неподдерживаемый loss: {config.loss}")


def _accumulate_loss_components(
    totals: dict[str, float],
    counts: dict[str, int],
    loss_info: dict[str, object],
) -> None:
    for name in totals:
        value = loss_info[name]
        if value is None:
            continue
        totals[name] += float(value.detach().item())
        counts[name] += 1


def _average_component(
    totals: dict[str, float],
    counts: dict[str, int],
    name: str,
) -> float | None:
    count = counts[name]
    if count == 0:
        return None
    return totals[name] / count


class _ProbabilityStats:
    def __init__(self, torch, prefix: str) -> None:
        self._torch = torch
        self._prefix = prefix
        self._histogram = torch.zeros(PROBABILITY_HISTOGRAM_BINS, dtype=torch.long, device="cpu")
        self._count = 0
        self._sum = 0.0
        self._min = 1.0
        self._max = 0.0

    def update(self, probs, selector=None) -> None:
        detached = probs.detach()
        if selector is not None:
            detached = detached[selector.detach()]
        count = int(detached.numel())
        if count == 0:
            return

        self._count += count
        self._sum += float(detached.sum().item())
        self._min = min(self._min, float(detached.min().item()))
        self._max = max(self._max, float(detached.max().item()))

        bins = self._torch.clamp(
            (detached * PROBABILITY_HISTOGRAM_BINS).to(dtype=self._torch.long),
            min=0,
            max=PROBABILITY_HISTOGRAM_BINS - 1,
        )
        histogram = self._torch.bincount(
            bins.reshape(-1).cpu(),
            minlength=PROBABILITY_HISTOGRAM_BINS,
        )
        self._histogram += histogram

    def snapshot(self) -> dict[str, float]:
        if self._count == 0:
            return {
                f"{self._prefix}_mean": 0.0,
                f"{self._prefix}_min": 0.0,
                f"{self._prefix}_max": 0.0,
                f"{self._prefix}_p50": 0.0,
                f"{self._prefix}_p90": 0.0,
                f"{self._prefix}_p99": 0.0,
                f"{self._prefix}_p999": 0.0,
            }
        return {
            f"{self._prefix}_mean": self._sum / self._count,
            f"{self._prefix}_min": self._min,
            f"{self._prefix}_max": self._max,
            f"{self._prefix}_p50": self._percentile(0.50),
            f"{self._prefix}_p90": self._percentile(0.90),
            f"{self._prefix}_p99": self._percentile(0.99),
            f"{self._prefix}_p999": self._percentile(0.999),
        }

    def _percentile(self, fraction: float) -> float:
        target = max(1, int(self._count * fraction))
        cumulative = self._torch.cumsum(self._histogram, dim=0)
        bin_index = int(self._torch.searchsorted(cumulative, target).item())
        return min(1.0, (bin_index + 0.5) / PROBABILITY_HISTOGRAM_BINS)


def _threshold_sweep_metrics(
    sweep_counts: dict[float, dict[str, int]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for threshold in THRESHOLD_SWEEP:
        counts = sweep_counts[threshold]
        precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        metrics[_threshold_key(threshold)] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def _best_threshold_metrics(
    sweep_metrics: dict[str, dict[str, float]],
) -> tuple[float, float, float, float]:
    best_threshold = THRESHOLD_SWEEP[0]
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    for item in sweep_metrics.values():
        threshold = item["threshold"]
        precision = item["precision"]
        recall = item["recall"]
        f1 = item["f1"]
        if f1 > best_f1:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_threshold, best_precision, best_recall, max(best_f1, 0.0)


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.3f}".replace(".", "_")


def _focal_loss(torch, logits, masks, config):
    return _focal_loss_with_bce(torch, logits, masks, config)[0]


def _focal_loss_with_bce(torch, logits, masks, config):
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
    return focal.mean(), bce.mean()


def _dice_loss(torch, logits, masks):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    intersection = torch.sum(probs * masks)
    denominator = torch.sum(probs) + torch.sum(masks)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def _multiclass_dice_loss(torch, logits, masks):
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
    smooth = 1.0
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def _tversky_loss(torch, logits, masks, config):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    true_positive = torch.sum(probs * masks)
    false_positive = torch.sum(probs * (1.0 - masks))
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
                "val_pixel_f1": metrics.val_pixel_f1,
                "val_macro_f1": metrics.val_macro_f1,
                "val_mean_iou": metrics.val_mean_iou,
                "val_pixel_accuracy": metrics.val_pixel_accuracy,
                "val_loss": metrics.val_loss,
                "train_loss": metrics.train_loss,
                "train_optimizer_steps": metrics.train_optimizer_steps,
                "train_skipped_optimizer_steps": metrics.train_skipped_optimizer_steps,
                "train_config": request.config.model_dump(mode="json"),
            },
        )
    )


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


def _multiclass_per_class_metrics(
    class_stats: dict[int, dict[str, int]],
    config,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for class_id in sorted(class_stats):
        stats = class_stats[class_id]
        precision = _safe_div(stats["tp"], stats["tp"] + stats["fp"])
        recall = _safe_div(stats["tp"], stats["tp"] + stats["fn"])
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        iou = _safe_div(stats["tp"], stats["tp"] + stats["fp"] + stats["fn"])
        metrics[_class_slug(config, class_id)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support_pixels": float(stats["support_pixels"]),
            "pred_pixels": float(stats["pred_pixels"]),
        }
    return metrics


def _class_slug(config, class_id: int) -> str:
    index = class_id - 1
    if 0 <= index < len(config.class_slugs):
        return str(config.class_slugs[index])
    return f"class_{class_id}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
