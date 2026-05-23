"""PyTorch цикл обучения сегментационной модели."""

from __future__ import annotations

from time import perf_counter

from mlsystem2.models.api import save_checkpoint
from mlsystem2.models.contracts import SaveCheckpointRequest

from .contracts import CheckpointArtifact, EpochMetrics, TrainError, TrainProgressEvent
from .contracts import TrainProgressSink, TrainRequest, TrainResult


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

            train_loss = _train_epoch(torch, model, request.train_loader, optimizer, device, config)
            val = _validate_epoch(torch, model, request.val_loader, device, config)
            scheduler.step()

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val["loss"],
                val_pixel_precision=val["precision"],
                val_pixel_recall=val["recall"],
                val_pixel_f1=val["f1"],
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


def _train_epoch(torch, model, loader: object, optimizer: object, device: object, config) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(torch, model, images, masks)
        loss = _loss(torch, logits, masks, config)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        batches += 1
        if (
            config.max_train_batches_per_epoch is not None
            and batch_index >= config.max_train_batches_per_epoch
        ):
            break
    if batches == 0:
        raise TrainError("Train DataLoader не вернул ни одного batch.")
    return total_loss / batches


def _validate_epoch(torch, model, loader: object, device: object, config) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(loader, start=1):
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)
            logits = _forward_logits(torch, model, images, masks)
            loss = _loss(torch, logits, masks, config)
            total_loss += float(loss.detach().item())
            batches += 1

            probs = torch.sigmoid(logits)
            pred = probs >= config.threshold
            true = masks >= 0.5
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
    }


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
