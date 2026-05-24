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
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    for batch_index, batch in enumerate(loader, start=1):
        images, masks, _meta = _split_batch(batch, epoch, batch_index, "train")
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.float32)
        _ensure_finite_tensor(torch, images, "images", epoch, batch_index, "train")
        _ensure_finite_tensor(torch, masks, "masks", epoch, batch_index, "train")
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(torch, model, images, masks)
        _ensure_finite_tensor(torch, logits, "logits", epoch, batch_index, "train")
        loss = _loss(torch, logits, masks, config)
        _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "train")
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
    model.eval()
    total_loss = 0.0
    batches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    positive_pixels = 0
    pred_positive_pixels = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, _meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)
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
            positive_pixels += int(true.sum().item())
            pred_positive_pixels += int(pred.sum().item())
            true_positive += int((pred & true).sum().item())
            false_positive += int((pred & ~true).sum().item())
            false_negative += int((~pred & true).sum().item())
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


def _loss(torch, logits, masks, config):
    if config.loss == "bce_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
        )
        return bce + _dice_loss(torch, logits, masks)
    if config.loss == "focal_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = config.focal_alpha * torch.pow(1.0 - pt, 2.0) * bce
        return focal.mean() + _dice_loss(torch, logits, masks)
    if config.loss == "focal_tversky":
        return torch.pow(_tversky_loss(torch, logits, masks, config), 2.0)
    raise TrainError(f"Неподдерживаемый loss: {config.loss}")


def _dice_loss(torch, logits, masks):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    intersection = torch.sum(probs * masks)
    denominator = torch.sum(probs) + torch.sum(masks)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


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
