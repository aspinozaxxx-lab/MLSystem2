"""PyTorch цикл обучения сегментационной модели."""

from __future__ import annotations

import os
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request
from mlsystem2.models.api import load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, SaveCheckpointRequest
from mlsystem2.tile_preparation.contracts import HARD_NEGATIVE_LABEL

from .contracts import CheckpointArtifact, EpochMetrics, TrainError, TrainProgressEvent
from .contracts import TrainProgressSink, TrainRequest, TrainResult


MAX_NONFINITE_GRADIENT_SKIPS_PER_EPOCH = 1
OBJECT_METRIC_MAX_WORKERS = 8
THRESHOLD_CANDIDATES = (0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995)
MULTICLASS_THRESHOLD_CANDIDATES = (0.0, *THRESHOLD_CANDIDATES)
NEXT_GEN_OBJECT_THRESHOLD_CANDIDATES = tuple(
    sorted({*(index / 20.0 for index in range(1, 20)), 0.97, 0.99, 0.995})
)
TRAINING_CONTROL_DIR_ENV = "MLSYSTEM2_TRAINING_CONTROL_DIR"
PAUSE_REQUEST_FILE = "pause.request"
PAUSED_MARKER_FILE = "paused"
STOP_AND_SAVE_BEST_REQUEST_FILE = "stop-and-save-best.request"


class _TrainingStopAndSaveBestRequested(Exception):
    """Внутренний сигнал штатно завершить run с уже сохранённым best.pt."""


class _TrainingPauseController:
    """Кооперативно освобождает GPU, не завершая процесс и MLflow-run."""

    def __init__(self, torch, model, optimizer, device, control_dir: str | None) -> None:
        self._torch = torch
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._control_dir = Path(control_dir) if control_dir else None
        self._paused = False

    def pause_if_requested(self) -> None:
        if self._control_dir is None:
            return
        self._stop_if_requested()
        request_path = self._control_dir / PAUSE_REQUEST_FILE
        if not request_path.is_file():
            return
        marker_path = self._control_dir / PAUSED_MARKER_FILE
        self._control_dir.mkdir(parents=True, exist_ok=True)
        cpu = self._torch.device("cpu")
        try:
            pause_token = request_path.read_text(encoding="utf-8").strip()
            if not pause_token:
                return
            self._model.to(cpu)
            _move_optimizer_state(self._optimizer, cpu)
            _release_training_cuda(self._torch, self._device)
            temporary = marker_path.with_suffix(".tmp")
            temporary.write_text(f"{pause_token}\n", encoding="utf-8")
            os.replace(temporary, marker_path)
            self._paused = True
            while request_path.is_file():
                self._stop_if_requested()
                time.sleep(0.2)
            self._model.to(self._device)
            _move_optimizer_state(self._optimizer, self._device)
            self._stop_if_requested()
        finally:
            marker_path.unlink(missing_ok=True)
            self._paused = False

    def _stop_if_requested(self) -> None:
        if (
            self._control_dir is not None
            and (self._control_dir / STOP_AND_SAVE_BEST_REQUEST_FILE).is_file()
        ):
            raise _TrainingStopAndSaveBestRequested

    def close(self) -> None:
        if self._control_dir is not None:
            (self._control_dir / PAUSED_MARKER_FILE).unlink(missing_ok=True)


def _move_optimizer_state(optimizer: object, device: object) -> None:
    state_by_parameter = getattr(optimizer, "state", {})
    for state in state_by_parameter.values():
        if isinstance(state, dict):
            for key, value in list(state.items()):
                state[key] = _move_state_value(value, device)


def _move_state_value(value: object, device: object) -> object:
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_state_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_value(item, device) for item in value)
    return value


def _release_training_cuda(torch, device: object) -> None:
    if not str(device).startswith("cuda"):
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    cuda.synchronize(device)
    cuda.empty_cache()


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
    if config.pipeline_variant == "next_gen" and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.pipeline_variant == "next_gen":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )

    total_started = perf_counter()
    history: list[EpochMetrics] = []
    best_score = -1.0
    best_metrics: EpochMetrics | None = None
    last_validation_metrics: EpochMetrics | None = None
    patience = 0
    checkpoint_dir = _checkpoint_dir(request.checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"
    pause_controller = _TrainingPauseController(
        torch,
        model,
        optimizer,
        device,
        os.getenv(TRAINING_CONTROL_DIR_ENV),
    )
    stopped_early = False

    try:
        try:
            for epoch in range(1, config.epochs + 1):
                pause_controller.pause_if_requested()
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
                    pause_controller=pause_controller,
                )
                _ensure_finite_scalar(train_epoch["loss"], "train_loss", epoch)
                validate_now = (
                    config.pipeline_variant == "legacy"
                    or epoch == 1
                    or epoch % config.validation_interval_epochs == 0
                    or epoch == config.epochs
                    or _training_time_exceeded(config, total_started)
                )
                if not validate_now:
                    metrics = EpochMetrics(
                        epoch=epoch,
                        validation_performed=False,
                        train_loss=train_epoch["loss"],
                        val_loss=None,
                        quality_metric=config.quality_metric,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        epoch_time_sec=perf_counter() - epoch_started,
                    )
                    history.append(metrics)
                    _emit(progress_sink, epoch, "epoch_finished", metrics)
                    continue

                val = _validate_epoch(
                    torch,
                    model,
                    request.val_loader,
                    device,
                    config,
                    epoch,
                    pause_controller=pause_controller,
                )
                _ensure_finite_scalar(float(val["loss"]), "val_loss", epoch)
                if config.pipeline_variant == "next_gen":
                    scheduler.step(float(val["quality_f1"]))
                else:
                    scheduler.step()

                metrics = EpochMetrics(
                    epoch=epoch,
                    validation_performed=True,
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
                    val_best_threshold_object_precision=val.get("best_threshold_object_precision"),
                    val_best_threshold_object_recall=val.get("best_threshold_object_recall"),
                    val_macro_pixel_f1=val.get("macro_pixel_f1"),
                    val_macro_pixel_precision=val.get("macro_pixel_precision"),
                    val_macro_pixel_recall=val.get("macro_pixel_recall"),
                    val_macro_pixel_iou=val.get("macro_pixel_iou"),
                    val_micro_pixel_f1=val.get("micro_pixel_f1"),
                    val_micro_pixel_precision=val.get("micro_pixel_precision"),
                    val_micro_pixel_recall=val.get("micro_pixel_recall"),
                    val_fixed_0_5_pixel_f1=val.get("fixed_0_5_pixel_f1"),
                    val_fixed_0_5_pixel_precision=val.get("fixed_0_5_pixel_precision"),
                    val_fixed_0_5_pixel_recall=val.get("fixed_0_5_pixel_recall"),
                    val_foreground_pixel_f1=val.get("foreground_pixel_f1"),
                    val_foreground_pixel_precision=val.get("foreground_pixel_precision"),
                    val_foreground_pixel_recall=val.get("foreground_pixel_recall"),
                    val_per_class_metrics=val.get("per_class_metrics", []),
                    val_multiclass_threshold_sweep=val.get("threshold_sweep", {}),
                    val_metric_warnings=val.get("metric_warnings", []),
                    val_per_scene_metrics=val.get("per_scene_metrics", []),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    epoch_time_sec=perf_counter() - epoch_started,
                )
                history.append(metrics)
                last_validation_metrics = metrics

                score = _checkpoint_score(metrics)
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    patience = 0
                    _save_training_checkpoint(
                        request,
                        str(best_checkpoint_path),
                        metrics,
                        "best",
                    )
                else:
                    patience += 1

                _emit(progress_sink, epoch, "epoch_finished", metrics)
                if _training_time_exceeded(config, total_started):
                    break
                if patience >= config.early_stopping_patience:
                    break
        except _TrainingStopAndSaveBestRequested:
            stopped_early = True

        if not history:
            raise TrainError("Обучение не выполнило ни одной эпохи.")
        if stopped_early and not best_checkpoint_path.is_file():
            raise TrainError(
                "Нельзя остановить обучение с сохранением: лучший чекпойнт ещё не создан."
            )

        artifacts = [CheckpointArtifact(uri=str(best_checkpoint_path), label="best")]
        final_checkpoint: str | None = None
        if not stopped_early:
            final_metrics = last_validation_metrics or best_metrics
            if final_metrics is None:
                raise TrainError("Не удалось получить validation для final checkpoint.")
            _save_training_checkpoint(request, str(final_checkpoint_path), final_metrics, "final")
            final_checkpoint = str(final_checkpoint_path)
            artifacts.append(CheckpointArtifact(uri=str(final_checkpoint_path), label="final"))
        diagnostics: dict[str, Any] = {}
        if config.pipeline_variant == "next_gen":
            diagnostics["peak_vram_bytes"] = (
                int(torch.cuda.max_memory_allocated(device))
                if str(device).startswith("cuda")
                else 0
            )
        if config.pipeline_variant == "next_gen" and config.evaluate_gaussian_blend:
            if best_metrics is None:
                raise TrainError("Gaussian A/B требует сохранённый best checkpoint.")
            model.to(torch.device("cpu"))
            _release_training_cuda(torch, device)
            diagnostics["inference_merge_comparison"] = _evaluate_gaussian_merge(
                torch,
                str(best_checkpoint_path),
                request.val_loader,
                device,
                best_metrics,
                request.sample_size,
            )
        return TrainResult(
            history=history,
            epochs_total=len(history),
            training_time_sec=perf_counter() - total_started,
            best_checkpoint_path=str(best_checkpoint_path),
            final_checkpoint_path=final_checkpoint,
            artifacts=artifacts,
            task=config.task,
            class_schema=list(config.class_schema),
            best_threshold=(best_metrics.val_best_threshold if best_metrics is not None else None),
            stopped_early=stopped_early,
            diagnostics=diagnostics,
        )
    except TrainError:
        raise
    except Exception as exc:
        raise TrainError("Ошибка во время обучения модели") from exc
    finally:
        pause_controller.close()


def _evaluate_gaussian_merge(
    torch,
    checkpoint_path: str,
    loader: object,
    device: object,
    best_metrics: EpochMetrics,
    patch_size: int | None,
) -> dict[str, object]:
    import numpy as np

    if patch_size != 512:
        raise TrainError("Gaussian A/B next-gen требует окно 512 px.")
    loaded = load_checkpoint(
        LoadCheckpointRequest(
            checkpoint_uri=checkpoint_path,
            model_spec=None,
            map_location="cpu",
        )
    )
    diagnostic_model = loaded.model.model
    diagnostic_model.to(device)
    diagnostic_model.eval()
    sigma = patch_size / 4.0
    coordinates = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
    axis = np.exp(-(coordinates**2) / (2.0 * sigma**2)).astype(np.float32)
    gaussian = np.outer(axis, axis).astype(np.float32)
    gaussian /= float(gaussian.max())
    scenes: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, meta = _split_batch(
                batch, best_metrics.epoch, batch_index, "gaussian_diagnostic"
            )
            if not isinstance(meta, dict):
                raise TrainError("Gaussian A/B не получила metadata тайлов.")
            scene_ids = meta.get("scene_ids")
            windows = meta.get("windows")
            scene_shapes = meta.get("scene_shapes")
            valid_pixels = meta.get("valid_pixels")
            if not (
                isinstance(scene_ids, list)
                and isinstance(windows, list)
                and isinstance(scene_shapes, list)
                and hasattr(valid_pixels, "shape")
            ):
                raise TrainError("Gaussian A/B не получила scene/window/valid metadata.")
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)
            logits = _forward_logits(torch, diagnostic_model, images, masks)
            probabilities = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
            targets = (masks[:, 0] >= 0.5).detach().cpu().numpy()
            valid_batch = valid_pixels[:, 0].detach().cpu().numpy().astype(bool)
            for sample_index, raw_scene_id in enumerate(scene_ids):
                scene_id = str(raw_scene_id)
                window = dict(windows[sample_index])
                shape = dict(scene_shapes[sample_index])
                width = int(shape["width"])
                height = int(shape["height"])
                accumulator = scenes.get(scene_id)
                if accumulator is None:
                    accumulator = {
                        "probability_sum": np.zeros((height, width), dtype=np.float32),
                        "weight_sum": np.zeros((height, width), dtype=np.float32),
                        "target": np.zeros((height, width), dtype=bool),
                        "valid": np.zeros((height, width), dtype=bool),
                        "window_count": 0,
                        "padding": {"left": False, "top": False, "right": False, "bottom": False},
                    }
                    scenes[scene_id] = accumulator
                elif accumulator["probability_sum"].shape != (height, width):
                    raise TrainError(f"Для сцены {scene_id} получены разные размеры raster.")
                _accumulate_gaussian_window(
                    accumulator,
                    gaussian,
                    probabilities[sample_index],
                    targets[sample_index],
                    valid_batch[sample_index],
                    window,
                    width,
                    height,
                )
            del images, masks, logits, probabilities, targets, valid_batch
    diagnostic_model.to(torch.device("cpu"))
    _release_training_cuda(torch, device)
    if not scenes:
        raise TrainError("Gaussian A/B не получила ни одной validation-сцены.")

    threshold = float(best_metrics.val_best_threshold)
    gaussian_scenes: list[dict[str, object]] = []
    micro_counts = {"tp": 0, "fp": 0, "fn": 0}
    fixed_micro_counts = {"tp": 0, "fp": 0, "fn": 0}
    for scene_id in sorted(scenes):
        accumulator = scenes[scene_id]
        probability_sum = accumulator.pop("probability_sum")
        weight_sum = accumulator.pop("weight_sum")
        target = accumulator.pop("target")
        valid = accumulator.pop("valid")
        merged = np.zeros_like(probability_sum)
        np.divide(probability_sum, weight_sum, out=merged, where=weight_sum > 0)
        uncovered_valid = int(np.count_nonzero(valid & (weight_sum <= 0)))
        if uncovered_valid:
            raise TrainError(
                f"Gaussian merge оставил непокрытые valid pixels: {scene_id}={uncovered_valid}"
            )
        counts = _numpy_binary_counts(merged, target, valid, threshold)
        fixed_counts = _numpy_binary_counts(merged, target, valid, 0.5)
        for key in micro_counts:
            micro_counts[key] += counts[key]
            fixed_micro_counts[key] += fixed_counts[key]
        precision, recall, f1 = _threshold_metrics(counts)
        fixed_precision, fixed_recall, fixed_f1 = _threshold_metrics(fixed_counts)
        gaussian_scenes.append(
            {
                "scene_id": scene_id,
                "threshold": threshold,
                "pixel_precision": precision,
                "pixel_recall": recall,
                "pixel_f1": f1,
                "fixed_0_5_precision": fixed_precision,
                "fixed_0_5_recall": fixed_recall,
                "fixed_0_5_f1": fixed_f1,
                "valid_pixels": int(np.count_nonzero(valid)),
                "uncovered_valid_pixels": uncovered_valid,
                "nodata_predicted_pixels": int(
                    np.count_nonzero(((merged >= threshold) & valid) & ~valid)
                ),
                "window_count": int(accumulator["window_count"]),
                "padding_coverage": dict(accumulator["padding"]),
            }
        )
    micro_precision, micro_recall, micro_f1 = _threshold_metrics(micro_counts)
    fixed_micro_precision, fixed_micro_recall, fixed_micro_f1 = _threshold_metrics(
        fixed_micro_counts
    )
    macro = _macro_scene_metrics(gaussian_scenes, "pixel")
    fixed_macro = _macro_scene_metrics(gaussian_scenes, "fixed_0_5")
    core_scenes = list(best_metrics.val_per_scene_metrics)
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_loaded_without_hub_access": True,
        "window_size": patch_size,
        "stride": 256,
        "sigma": sigma,
        "threshold": threshold,
        "core_crop": {
            "macro_pixel_f1": best_metrics.val_macro_pixel_f1,
            "micro_pixel_f1": best_metrics.val_micro_pixel_f1,
            "scenes": core_scenes,
        },
        "gaussian": {
            **macro,
            "micro_pixel_precision": micro_precision,
            "micro_pixel_recall": micro_recall,
            "micro_pixel_f1": micro_f1,
            **{f"fixed_0_5_{key}": value for key, value in fixed_macro.items()},
            "fixed_0_5_micro_precision": fixed_micro_precision,
            "fixed_0_5_micro_recall": fixed_micro_recall,
            "fixed_0_5_micro_f1": fixed_micro_f1,
            "scenes": gaussian_scenes,
        },
        "production_merge_unchanged": "core_crop",
    }


def _accumulate_gaussian_window(
    accumulator: dict[str, object],
    gaussian,
    probability,
    target,
    valid,
    window: dict[str, object],
    width: int,
    height: int,
) -> None:
    import numpy as np

    x = int(window["x"])
    y = int(window["y"])
    window_width = int(window["width"])
    window_height = int(window["height"])
    if probability.shape != (window_height, window_width):
        raise TrainError("Размер logits не совпадает с Gaussian-окном.")
    destination_x0 = max(0, x)
    destination_y0 = max(0, y)
    destination_x1 = min(width, x + window_width)
    destination_y1 = min(height, y + window_height)
    if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
        return
    source_x0 = destination_x0 - x
    source_y0 = destination_y0 - y
    source_x1 = source_x0 + destination_x1 - destination_x0
    source_y1 = source_y0 + destination_y1 - destination_y0
    destination = np.s_[destination_y0:destination_y1, destination_x0:destination_x1]
    source = np.s_[source_y0:source_y1, source_x0:source_x1]
    valid_patch = valid[source]
    weights = gaussian[source] * valid_patch
    accumulator["probability_sum"][destination] += probability[source] * weights
    accumulator["weight_sum"][destination] += weights
    accumulator["target"][destination] |= target[source] & valid_patch
    accumulator["valid"][destination] |= valid_patch
    accumulator["window_count"] = int(accumulator["window_count"]) + 1
    padding = accumulator["padding"]
    padding["left"] = bool(padding["left"] or x < 0)
    padding["top"] = bool(padding["top"] or y < 0)
    padding["right"] = bool(padding["right"] or x + window_width > width)
    padding["bottom"] = bool(padding["bottom"] or y + window_height > height)


def _numpy_binary_counts(probability, target, valid, threshold: float) -> dict[str, int]:
    predicted = (probability >= threshold) & valid
    true = target & valid
    return {
        "tp": int((predicted & true).sum()),
        "fp": int((predicted & ~true & valid).sum()),
        "fn": int((~predicted & true).sum()),
    }


def _macro_scene_metrics(
    scenes: list[dict[str, object]], prefix: str
) -> dict[str, float]:
    names = ("precision", "recall", "f1")
    return {
        f"macro_pixel_{name}": sum(
            float(scene[f"{prefix}_{name}"]) for scene in scenes
        )
        / len(scenes)
        for name in names
    }


def _train_epoch(
    torch,
    model,
    loader: object,
    optimizer: object,
    device: object,
    config,
    epoch: int,
    pause_controller: "_TrainingPauseController | None" = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    batches = 0
    has_optimizer_step = False
    nonfinite_gradient_skips = 0
    for batch_index, batch in enumerate(loader, start=1):
        images, masks, meta = _split_batch(batch, epoch, batch_index, "train")
        images = images.to(device=device, dtype=torch.float32)
        masks, hard_negative_pixels = _prepare_supervision_masks(torch, masks, config, device)
        valid_pixels = _prepare_valid_pixels(torch, meta, config, device, masks)
        class_hard_negative_pixels = _prepare_class_hard_negative_pixels(
            torch,
            meta,
            config,
            device,
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
        if valid_pixels is not None:
            valid_pixels = _crop_spatial(valid_pixels, config.inference_context)
        if class_hard_negative_pixels is not None:
            class_hard_negative_pixels = _crop_spatial(
                class_hard_negative_pixels,
                config.inference_context,
            )
        if config.task == "multiclass":
            _validate_multiclass_targets(torch, masks, logits.shape[1], epoch, batch_index, "train")
        loss = _loss(
            torch,
            logits,
            masks,
            config,
            hard_negative_pixels,
            class_hard_negative_pixels,
            valid_pixels,
        )
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
            reached_limit = (
                config.max_train_batches_per_epoch is not None
                and batch_index >= config.max_train_batches_per_epoch
            )
            del images, masks, hard_negative_pixels, class_hard_negative_pixels, valid_pixels, logits, loss
            if pause_controller is not None:
                pause_controller.pause_if_requested()
            if reached_limit:
                break
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _ensure_finite_tensor(torch, grad_norm, "grad_norm", epoch, batch_index, "train")
        optimizer.step()
        has_optimizer_step = True
        total_loss += float(loss.detach().item())
        batches += 1
        reached_limit = (
            config.max_train_batches_per_epoch is not None
            and batch_index >= config.max_train_batches_per_epoch
        )
        del images, masks, hard_negative_pixels, class_hard_negative_pixels, valid_pixels, logits, loss, grad_norm
        if pause_controller is not None:
            pause_controller.pause_if_requested()
        if reached_limit:
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
    pause_controller: "_TrainingPauseController | None" = None,
) -> dict[str, float | None]:
    if pause_controller is not None:
        pause_controller.pause_if_requested()
    if config.task == "multiclass":
        return _validate_multiclass_epoch(
            torch,
            model,
            loader,
            device,
            config,
            epoch,
            pause_controller=pause_controller,
        )

    import numpy as np

    model.eval()
    next_gen = config.pipeline_variant == "next_gen"
    total_loss = 0.0
    batches = 0
    threshold_counts = (
        {}
        if next_gen
        else {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in THRESHOLD_CANDIDATES}
    )
    positive_histogram = np.zeros(4096, dtype=np.int64)
    negative_histogram = np.zeros(4096, dtype=np.int64)
    scene_histograms: dict[str, tuple[object, object]] = {}
    configured_fixed_counts = {"tp": 0, "fp": 0, "fn": 0}
    fixed_0_5_counts = {"tp": 0, "fp": 0, "fn": 0}
    scene_configured_fixed_counts: dict[str, dict[str, int]] = {}
    scene_fixed_0_5_counts: dict[str, dict[str, int]] = {}
    if not next_gen:
        object_candidates = THRESHOLD_CANDIDATES
    elif config.quality_metric == "objects":
        object_candidates = tuple(
            sorted({*NEXT_GEN_OBJECT_THRESHOLD_CANDIDATES, float(config.threshold)})
        )
    elif config.threshold_mode == "fixed":
        object_candidates = (float(config.threshold),)
    else:
        object_candidates = ()
    object_threshold_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in object_candidates
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
            valid_pixels = _prepare_valid_pixels(torch, meta, config, device, masks)
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
            if valid_pixels is not None:
                valid_pixels = _crop_spatial(valid_pixels, config.inference_context)
            loss = _loss(
                torch,
                logits,
                masks,
                config,
                hard_negative_pixels,
                None,
                valid_pixels,
            )
            _ensure_finite_tensor(torch, loss, "loss", epoch, batch_index, "val")
            total_loss += float(loss.detach().item())
            batches += 1

            probs = torch.sigmoid(logits)
            true = masks >= 0.5
            if next_gen:
                if valid_pixels is None:
                    raise TrainError("next_gen validation не получила valid_pixels")
                _accumulate_probability_histogram(
                    torch,
                    positive_histogram,
                    negative_histogram,
                    probs[:, 0],
                    true[:, 0],
                    valid_pixels[:, 0],
                )
                _accumulate_exact_threshold_counts(
                    configured_fixed_counts,
                    probs[:, 0],
                    true[:, 0],
                    valid_pixels[:, 0],
                    float(config.threshold),
                )
                _accumulate_exact_threshold_counts(
                    fixed_0_5_counts,
                    probs[:, 0],
                    true[:, 0],
                    valid_pixels[:, 0],
                    0.5,
                )
                scene_ids = meta.get("scene_ids") if isinstance(meta, dict) else None
                if not isinstance(scene_ids, list) or len(scene_ids) != int(probs.shape[0]):
                    raise TrainError("next_gen validation не получила scene_ids")
                for sample_index, scene_id in enumerate(scene_ids):
                    scene_positive, scene_negative = scene_histograms.setdefault(
                        str(scene_id),
                        (
                            np.zeros(4096, dtype=np.int64),
                            np.zeros(4096, dtype=np.int64),
                        ),
                    )
                    _accumulate_probability_histogram(
                        torch,
                        scene_positive,
                        scene_negative,
                        probs[sample_index],
                        true[sample_index],
                        valid_pixels[sample_index],
                    )
                    configured_scene_counts = scene_configured_fixed_counts.setdefault(
                        str(scene_id), {"tp": 0, "fp": 0, "fn": 0}
                    )
                    _accumulate_exact_threshold_counts(
                        configured_scene_counts,
                        probs[sample_index],
                        true[sample_index],
                        valid_pixels[sample_index],
                        float(config.threshold),
                    )
                    fixed_scene_counts = scene_fixed_0_5_counts.setdefault(
                        str(scene_id), {"tp": 0, "fp": 0, "fn": 0}
                    )
                    _accumulate_exact_threshold_counts(
                        fixed_scene_counts,
                        probs[sample_index],
                        true[sample_index],
                        valid_pixels[sample_index],
                        0.5,
                    )
            else:
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
                object_probs = probs[:, 0, :, :]
                true_instances = _as_numpy_instances(object_instances)
                if valid_pixels is not None:
                    object_probs = torch.where(
                        valid_pixels[:, 0],
                        object_probs,
                        torch.full_like(object_probs, -1.0),
                    )
                    true_instances = np.where(
                        valid_pixels[:, 0].detach().cpu().numpy(),
                        true_instances,
                        0,
                    )
                if object_threshold_counts:
                    _accumulate_object_threshold_counts(
                        object_threshold_counts,
                        true_instances,
                        object_probs.detach().cpu().numpy(),
                        object_metric_executor,
                    )
            reached_limit = (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            )
            del images, masks, hard_negative_pixels, valid_pixels, logits, loss, probs, true
            if pause_controller is not None:
                pause_controller.pause_if_requested()
            if reached_limit:
                break

    if batches == 0:
        raise TrainError("Val DataLoader не вернул ни одного batch.")

    per_scene_metrics: list[dict[str, object]] = []
    metric_warnings: list[str] = []
    if next_gen:
        if config.threshold_mode == "fixed":
            pixel_threshold = float(config.threshold)
            pixel_precision, pixel_recall, pixel_f1 = _threshold_metrics(
                configured_fixed_counts
            )
        else:
            pixel_threshold, pixel_precision, pixel_recall, pixel_f1 = (
                _histogram_threshold_metrics(
                    positive_histogram,
                    negative_histogram,
                    mode="optimize",
                    fixed_threshold=float(config.threshold),
                )
            )
        for scene_id in sorted(scene_histograms):
            scene_positive, scene_negative = scene_histograms[scene_id]
            if config.threshold_mode == "fixed":
                optimal_threshold = pixel_threshold
                optimal_precision, optimal_recall, optimal_f1 = _threshold_metrics(
                    scene_configured_fixed_counts[scene_id]
                )
                precision, recall, f1 = optimal_precision, optimal_recall, optimal_f1
                selected_counts = scene_configured_fixed_counts[scene_id]
            else:
                optimal_threshold, optimal_precision, optimal_recall, optimal_f1 = (
                    _histogram_threshold_metrics(
                        scene_positive,
                        scene_negative,
                        mode="optimize",
                        fixed_threshold=float(config.threshold),
                    )
                )
                _, precision, recall, f1 = _histogram_threshold_metrics(
                    scene_positive,
                    scene_negative,
                    mode="fixed",
                    fixed_threshold=pixel_threshold,
                )
                selected_counts = _histogram_counts_at_threshold(
                    scene_positive,
                    scene_negative,
                    pixel_threshold,
                )
            fixed_precision, fixed_recall, fixed_f1 = _threshold_metrics(
                scene_fixed_0_5_counts[scene_id]
            )
            per_scene_metrics.append(
                {
                    "scene_id": scene_id,
                    "threshold": pixel_threshold,
                    "pixel_precision": precision,
                    "pixel_recall": recall,
                    "pixel_f1": f1,
                    "scene_optimal_threshold": optimal_threshold,
                    "scene_optimal_precision": optimal_precision,
                    "scene_optimal_recall": optimal_recall,
                    "scene_optimal_f1": optimal_f1,
                    "fixed_0_5_precision": fixed_precision,
                    "fixed_0_5_recall": fixed_recall,
                    "fixed_0_5_f1": fixed_f1,
                    "true_positive": selected_counts["tp"],
                    "false_positive": selected_counts["fp"],
                    "false_negative": selected_counts["fn"],
                    "valid_pixels": int(scene_positive.sum() + scene_negative.sum()),
                    "fixed_0_5_true_positive": scene_fixed_0_5_counts[scene_id]["tp"],
                    "fixed_0_5_false_positive": scene_fixed_0_5_counts[scene_id]["fp"],
                    "fixed_0_5_false_negative": scene_fixed_0_5_counts[scene_id]["fn"],
                }
            )
    else:
        pixel_threshold, pixel_precision, pixel_recall, pixel_f1 = _best_threshold_metrics(
            threshold_counts
        )
    macro_pixel_precision = (
        sum(float(item["pixel_precision"]) for item in per_scene_metrics)
        / len(per_scene_metrics)
        if per_scene_metrics
        else None
    )
    macro_pixel_recall = (
        sum(float(item["pixel_recall"]) for item in per_scene_metrics)
        / len(per_scene_metrics)
        if per_scene_metrics
        else None
    )
    macro_pixel_f1 = (
        sum(float(item["pixel_f1"]) for item in per_scene_metrics) / len(per_scene_metrics)
        if per_scene_metrics
        else None
    )
    if object_instances_seen:
        if not next_gen:
            object_threshold, object_precision, object_recall, object_f1 = (
                _best_threshold_metrics(object_threshold_counts)
            )
        elif config.quality_metric != "objects":
            object_threshold = pixel_threshold
            if object_threshold in object_threshold_counts:
                exact_object_counts = object_threshold_counts[object_threshold]
            else:
                exact_object_counts = _validate_object_threshold(
                    torch,
                    model,
                    loader,
                    device,
                    config,
                    epoch,
                    object_threshold,
                    pause_controller,
                )
            object_precision, object_recall, object_f1 = _threshold_metrics(
                exact_object_counts
            )
        else:
            object_threshold, object_precision, object_recall, object_f1 = (
                _best_threshold_metrics_for_candidates(object_threshold_counts)
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
        quality_precision = (
            macro_pixel_precision if macro_pixel_precision is not None else pixel_precision
        )
        quality_recall = macro_pixel_recall if macro_pixel_recall is not None else pixel_recall
        quality_f1 = macro_pixel_f1 if macro_pixel_f1 is not None else pixel_f1
    fixed_0_5_precision, fixed_0_5_recall, fixed_0_5_f1 = _threshold_metrics(
        fixed_0_5_counts
    ) if next_gen else (None, None, None)
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
        "macro_pixel_f1": macro_pixel_f1,
        "macro_pixel_precision": macro_pixel_precision,
        "macro_pixel_recall": macro_pixel_recall,
        "micro_pixel_f1": pixel_f1 if next_gen else None,
        "micro_pixel_precision": pixel_precision if next_gen else None,
        "micro_pixel_recall": pixel_recall if next_gen else None,
        "fixed_0_5_pixel_f1": fixed_0_5_f1,
        "fixed_0_5_pixel_precision": fixed_0_5_precision,
        "fixed_0_5_pixel_recall": fixed_0_5_recall,
        "per_scene_metrics": per_scene_metrics,
        "metric_warnings": metric_warnings,
    }


def _accumulate_probability_histogram(
    torch,
    positive_histogram,
    negative_histogram,
    probabilities,
    true,
    valid,
) -> None:
    bins = torch.clamp((probabilities * 4095.0).to(dtype=torch.long), 0, 4095)
    positive = bins[true & valid]
    negative = bins[(~true) & valid]
    if int(positive.numel()):
        positive_histogram += torch.bincount(positive, minlength=4096).cpu().numpy()
    if int(negative.numel()):
        negative_histogram += torch.bincount(negative, minlength=4096).cpu().numpy()


def _accumulate_exact_threshold_counts(
    counts: dict[str, int],
    probabilities,
    true,
    valid,
    threshold: float,
) -> None:
    predicted = probabilities >= threshold
    counts["tp"] += int((predicted & true & valid).sum().item())
    counts["fp"] += int((predicted & ~true & valid).sum().item())
    counts["fn"] += int((~predicted & true & valid).sum().item())


def _histogram_threshold_metrics(
    positive_histogram,
    negative_histogram,
    *,
    mode: str,
    fixed_threshold: float,
) -> tuple[float, float, float, float]:
    import numpy as np

    positive = np.asarray(positive_histogram, dtype=np.int64)
    negative = np.asarray(negative_histogram, dtype=np.int64)
    tp_by_bin = np.cumsum(positive[::-1], dtype=np.int64)[::-1]
    fp_by_bin = np.cumsum(negative[::-1], dtype=np.int64)[::-1]
    positives = int(positive.sum())
    if mode == "fixed":
        index = min(4095, max(0, int(math.ceil(fixed_threshold * 4095.0))))
        precision, recall, f1 = _threshold_metrics(
            {
                "tp": int(tp_by_bin[index]),
                "fp": int(fp_by_bin[index]),
                "fn": positives - int(tp_by_bin[index]),
            }
        )
        return float(fixed_threshold), precision, recall, f1
    best = (0.0, 0.0, 0.0, 0.0)
    best_key = (-1.0, -1.0, -1.0)
    for index in range(4096):
        threshold = index / 4095.0
        precision, recall, f1 = _threshold_metrics(
            {
                "tp": int(tp_by_bin[index]),
                "fp": int(fp_by_bin[index]),
                "fn": positives - int(tp_by_bin[index]),
            }
        )
        key = (f1, precision, threshold)
        if key > best_key:
            best_key = key
            best = (threshold, precision, recall, f1)
    return best


def _histogram_counts_at_threshold(
    positive_histogram,
    negative_histogram,
    threshold: float,
) -> dict[str, int]:
    import numpy as np

    positive = np.asarray(positive_histogram, dtype=np.int64)
    negative = np.asarray(negative_histogram, dtype=np.int64)
    index = min(4095, max(0, int(round(threshold * 4095.0))))
    true_positive = int(positive[index:].sum())
    return {
        "tp": true_positive,
        "fp": int(negative[index:].sum()),
        "fn": int(positive.sum()) - true_positive,
    }


def _validate_object_threshold(
    torch,
    model,
    loader: object,
    device: object,
    config,
    epoch: int,
    threshold: float,
    pause_controller: "_TrainingPauseController | None",
) -> dict[str, int]:
    import numpy as np

    threshold_counts = {threshold: {"tp": 0, "fp": 0, "fn": 0}}
    object_metric_workers = min(OBJECT_METRIC_MAX_WORKERS, max(1, os.cpu_count() or 1))
    with (
        ThreadPoolExecutor(max_workers=object_metric_workers) as executor,
        torch.no_grad(),
    ):
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, meta = _split_batch(
                batch,
                epoch,
                batch_index,
                "val_object_threshold",
            )
            if not isinstance(meta, dict) or meta.get("object_instances") is None:
                continue
            images = images.to(device=device, dtype=torch.float32)
            masks, _ = _prepare_supervision_masks(torch, masks, config, device)
            valid_pixels = _prepare_valid_pixels(torch, meta, config, device, masks)
            logits = _forward_logits(torch, model, images, masks)
            logits = _crop_spatial(logits, config.inference_context)
            object_instances = _crop_spatial(
                meta["object_instances"],
                config.inference_context,
            )
            if valid_pixels is not None:
                valid_pixels = _crop_spatial(valid_pixels, config.inference_context)
            probabilities = torch.sigmoid(logits[:, 0])
            true_instances = _as_numpy_instances(object_instances)
            if valid_pixels is not None:
                probabilities = torch.where(
                    valid_pixels[:, 0],
                    probabilities,
                    torch.full_like(probabilities, -1.0),
                )
                true_instances = np.where(
                    valid_pixels[:, 0].detach().cpu().numpy(),
                    true_instances,
                    0,
                )
            _accumulate_object_threshold_counts(
                threshold_counts,
                true_instances,
                probabilities.detach().cpu().numpy(),
                executor,
            )
            if pause_controller is not None:
                pause_controller.pause_if_requested()
    return threshold_counts[threshold]


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
    pause_controller: "_TrainingPauseController | None" = None,
) -> dict[str, Any]:
    if pause_controller is not None:
        pause_controller.pause_if_requested()
    model.eval()
    total_loss = 0.0
    batches = 0
    threshold_stats: dict[float, dict[int, dict[str, int]]] = {
        threshold: {} for threshold in MULTICLASS_THRESHOLD_CANDIDATES
    }
    foreground_stats: dict[float, dict[str, int]] = {
        threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in MULTICLASS_THRESHOLD_CANDIDATES
    }
    expected_num_classes = len(config.class_schema) + 1

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images, masks, meta = _split_batch(batch, epoch, batch_index, "val")
            images = images.to(device=device, dtype=torch.float32)
            masks, hard_negative_pixels = _prepare_supervision_masks(torch, masks, config, device)
            class_hard_negative_pixels = _prepare_class_hard_negative_pixels(
                torch,
                meta,
                config,
                device,
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
            if class_hard_negative_pixels is not None:
                class_hard_negative_pixels = _crop_spatial(
                    class_hard_negative_pixels,
                    config.inference_context,
                )
            num_classes = int(logits.shape[1])
            _validate_multiclass_targets(torch, masks, num_classes, epoch, batch_index, "val")
            if num_classes != expected_num_classes:
                raise TrainError(
                    "Число каналов multiclass-модели не соответствует class_schema: "
                    f"ожидается {expected_num_classes}, получено {num_classes}."
                )
            loss = _loss(
                torch,
                logits,
                masks,
                config,
                hard_negative_pixels,
                class_hard_negative_pixels,
            )
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
                    valid = (
                        torch.ones_like(expected, dtype=torch.bool)
                        if class_hard_negative_pixels is None
                        else (
                            ~class_hard_negative_pixels.any(dim=1)
                            | class_hard_negative_pixels[:, class_id - 1, :, :]
                        )
                    )
                    stats = stats_by_class.setdefault(
                        class_id,
                        {"tp": 0, "fp": 0, "fn": 0, "support": 0, "predicted": 0},
                    )
                    stats["tp"] += int((predicted & expected & valid).sum().item())
                    stats["fp"] += int((predicted & ~expected & valid).sum().item())
                    stats["fn"] += int((~predicted & expected & valid).sum().item())
                    stats["support"] += int((expected & valid).sum().item())
                    stats["predicted"] += int((predicted & valid).sum().item())
                predicted_foreground = labels > 0
                expected_foreground = masks > 0
                foreground_valid = (
                    torch.ones_like(expected_foreground, dtype=torch.bool)
                    if class_hard_negative_pixels is None
                    else ~class_hard_negative_pixels.any(dim=1)
                )
                foreground = foreground_stats[threshold]
                foreground["tp"] += int(
                    (predicted_foreground & expected_foreground & foreground_valid).sum().item()
                )
                foreground["fp"] += int(
                    (predicted_foreground & ~expected_foreground & foreground_valid).sum().item()
                )
                foreground["fn"] += int(
                    (~predicted_foreground & expected_foreground & foreground_valid).sum().item()
                )

            reached_limit = (
                config.max_val_batches_per_epoch is not None
                and batch_index >= config.max_val_batches_per_epoch
            )
            del images, masks, hard_negative_pixels, class_hard_negative_pixels
            del logits, loss, probabilities
            del confidence, raw_labels, labels, predicted, expected
            del predicted_foreground, expected_foreground, foreground_valid, valid
            if pause_controller is not None:
                pause_controller.pause_if_requested()
            if reached_limit:
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


def _prepare_class_hard_negative_pixels(torch, meta, config, device):
    if config.task != "multiclass" or not isinstance(meta, dict):
        return None
    raw = meta.get("class_hard_negative_masks")
    if raw is None:
        return None
    value = raw.to(device=device, dtype=torch.bool)
    expected_channels = len(config.class_schema)
    if value.ndim != 4 or int(value.shape[1]) != expected_channels:
        raise TrainError(
            f"Классовая hard negative маска должна иметь форму B×{expected_channels}×H×W."
        )
    return value


def _prepare_valid_pixels(torch, meta, config, device, masks):
    if config.pipeline_variant != "next_gen":
        return None
    if not isinstance(meta, dict) or meta.get("valid_pixels") is None:
        raise TrainError("next_gen batch не содержит valid_pixels")
    value = meta["valid_pixels"].to(device=device, dtype=torch.bool)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != masks.ndim or value.shape[-2:] != masks.shape[-2:]:
        raise TrainError("valid_pixels не совпадает с формой supervision mask")
    return value


def _loss(
    torch,
    logits,
    masks,
    config,
    hard_negative_pixels=None,
    class_hard_negative_pixels=None,
    valid_pixels=None,
):
    if config.task == "multiclass":
        if config.loss not in {"cross_entropy", "cross_entropy_dice"}:
            raise TrainError(
                "multiclass train поддерживает только loss=cross_entropy или cross_entropy_dice"
            )
        weights = _multiclass_base_loss_weights(
            torch,
            logits,
            masks,
            hard_negative_pixels,
            class_hard_negative_pixels,
            config,
        )
        cross_entropy = _weighted_mean(
            torch.nn.functional.cross_entropy(logits, masks, reduction="none"),
            weights,
        )
        class_hard_negative_loss = _class_hard_negative_loss(
            torch,
            logits,
            class_hard_negative_pixels,
            config,
        )
        if config.loss == "cross_entropy_dice":
            return (
                cross_entropy
                + class_hard_negative_loss
                + _multiclass_dice_loss(
                    torch,
                    logits,
                    masks,
                    hard_negative_pixels,
                    config,
                    class_hard_negative_pixels,
                )
            )
        return cross_entropy + class_hard_negative_loss
    if config.loss == "bce_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        weights = _pixel_loss_weights(
            torch, logits, masks, hard_negative_pixels, config, valid_pixels
        )
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
            reduction="none",
        )
        return _weighted_mean(
            bce,
            weights,
            normalize_by_weights=config.pipeline_variant == "next_gen",
        ) + _dice_loss(
            torch,
            logits,
            masks,
            hard_negative_pixels,
            config,
            valid_pixels,
        )
    if config.loss == "focal_dice":
        pos_weight = torch.tensor([config.pos_weight], device=logits.device, dtype=logits.dtype)
        weights = _pixel_loss_weights(
            torch, logits, masks, hard_negative_pixels, config, valid_pixels
        )
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = config.focal_alpha * torch.pow(1.0 - pt, 2.0) * bce
        return _weighted_mean(
            focal,
            weights,
            normalize_by_weights=config.pipeline_variant == "next_gen",
        ) + _dice_loss(
            torch,
            logits,
            masks,
            hard_negative_pixels,
            config,
            valid_pixels,
        )
    if config.loss == "focal_tversky":
        weights = _pixel_loss_weights(
            torch, logits, masks, hard_negative_pixels, config, valid_pixels
        )
        focal, _bce = _focal_loss_with_bce(
            torch,
            logits,
            masks,
            config,
            weights,
            normalize_by_weights=config.pipeline_variant == "next_gen",
        )
        return focal + _tversky_loss(
            torch,
            logits,
            masks,
            config,
            hard_negative_pixels,
            valid_pixels,
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


def _best_threshold_metrics_for_candidates(
    threshold_counts: dict[float, dict[str, int]],
) -> tuple[float, float, float, float]:
    best = (0.0, 0.0, 0.0, 0.0)
    best_key = (-1.0, -1.0, -1.0)
    for threshold in sorted(threshold_counts):
        precision, recall, f1 = _threshold_metrics(threshold_counts[threshold])
        key = (f1, precision, threshold)
        if key > best_key:
            best_key = key
            best = (threshold, precision, recall, f1)
    return best


def _threshold_metrics(counts: dict[str, int]) -> tuple[float, float, float]:
    precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _pixel_loss_weights(
    torch,
    logits,
    masks,
    hard_negative_pixels,
    config,
    valid_pixels=None,
):
    background_weight = float(getattr(config, "background_weight", 1.0))
    hard_negative_weight = float(getattr(config, "hard_negative_weight", 1.0))
    has_hard_negative_weights = (
        hard_negative_weight != 1.0
        and hard_negative_pixels is not None
        and hasattr(hard_negative_pixels, "to")
    )
    if (
        background_weight == 1.0
        and not has_hard_negative_weights
        and valid_pixels is None
    ):
        return None
    weights = torch.ones_like(
        masks,
        device=logits.device,
        dtype=logits.dtype,
    )
    if background_weight != 1.0:
        background_pixels = masks == 0
        weights = torch.where(
            background_pixels,
            torch.as_tensor(
                background_weight,
                device=logits.device,
                dtype=logits.dtype,
            ),
            weights,
        )
    if has_hard_negative_weights:
        weights = weights * (
            1.0
            + (hard_negative_weight - 1.0)
            * hard_negative_pixels.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        )
    if valid_pixels is not None:
        weights = weights * valid_pixels.to(
            device=logits.device,
            dtype=logits.dtype,
        )
    return weights


def _weighted_mean(values, weights, *, normalize_by_weights: bool = False):
    if weights is None:
        return values.mean()
    if normalize_by_weights:
        return (values * weights).sum() / weights.sum().clamp_min(1.0)
    return (values * weights).mean()


def _multiclass_base_loss_weights(
    torch,
    logits,
    masks,
    hard_negative_pixels,
    class_hard_negative_pixels,
    config,
):
    weights = _pixel_loss_weights(torch, logits, masks, hard_negative_pixels, config)
    if class_hard_negative_pixels is None:
        return weights
    valid = (~class_hard_negative_pixels.any(dim=1)).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    return valid if weights is None else weights * valid


def _class_hard_negative_loss(
    torch,
    logits,
    class_hard_negative_pixels,
    config,
):
    if class_hard_negative_pixels is None:
        return logits.sum() * 0.0
    probabilities = torch.softmax(logits, dim=1)[:, 1:, :, :]
    masks = class_hard_negative_pixels.to(
        device=logits.device,
        dtype=logits.dtype,
    )
    epsilon = torch.finfo(logits.dtype).eps
    complement_loss = -torch.log((1.0 - probabilities).clamp_min(epsilon))
    pixel_count = max(1, int(logits.shape[0] * logits.shape[2] * logits.shape[3]))
    return (
        complement_loss.mul(masks).sum()
        / pixel_count
        * float(getattr(config, "background_weight", 1.0))
        * float(getattr(config, "hard_negative_weight", 1.0))
    )


def _focal_loss_with_bce(
    torch,
    logits,
    masks,
    config,
    weights=None,
    normalize_by_weights: bool = False,
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
        _weighted_mean(focal, weights, normalize_by_weights=normalize_by_weights),
        _weighted_mean(bce, weights, normalize_by_weights=normalize_by_weights),
    )


def _dice_loss(
    torch,
    logits,
    masks,
    hard_negative_pixels=None,
    config=None,
    valid_pixels=None,
):
    probs = torch.sigmoid(logits)
    if valid_pixels is not None:
        masks = masks * valid_pixels.to(device=logits.device, dtype=logits.dtype)
    probability_weights = _pixel_loss_weights(
        torch,
        logits,
        masks,
        hard_negative_pixels,
        config,
        valid_pixels,
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
    class_hard_negative_pixels=None,
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
    if class_hard_negative_pixels is not None:
        valid = (
            (~class_hard_negative_pixels.any(dim=1))
            .unsqueeze(1)
            .to(
                device=logits.device,
                dtype=probs.dtype,
            )
        )
        probs = probs * valid
        target = target * valid
    probability_weights = _pixel_loss_weights(
        torch,
        logits,
        masks,
        hard_negative_pixels,
        config,
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
    valid_pixels=None,
):
    probs = torch.sigmoid(logits)
    if valid_pixels is not None:
        valid = valid_pixels.to(device=logits.device, dtype=logits.dtype)
        probs = probs * valid
        masks = masks * valid
    smooth = 1.0
    true_positive = torch.sum(probs * masks)
    false_positive_pixels = probs * (1.0 - masks)
    weights = _pixel_loss_weights(
        torch,
        logits,
        masks,
        hard_negative_pixels,
        config,
        valid_pixels,
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
    metadata = {
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
        "train_config": _checkpoint_train_config(request.config),
    }
    if request.config.pipeline_variant == "next_gen":
        metadata.update(
            {
                "pipeline_variant": "next_gen",
                "run_metadata": dict(request.run_metadata),
                "validation_performed": metrics.validation_performed,
                "val_per_scene_metrics": metrics.val_per_scene_metrics,
                "val_fixed_0_5_pixel_f1": metrics.val_fixed_0_5_pixel_f1,
                "val_fixed_0_5_pixel_precision": metrics.val_fixed_0_5_pixel_precision,
                "val_fixed_0_5_pixel_recall": metrics.val_fixed_0_5_pixel_recall,
                "learning_rate": metrics.learning_rate,
                "model_parameters": request.model.spec.parameters,
            }
        )
    save_checkpoint(
        SaveCheckpointRequest(
            model=request.model,
            checkpoint_uri=checkpoint_uri,
            metadata=metadata,
        )
    )


def _checkpoint_train_config(config) -> dict[str, object]:
    excluded = (
        {"pipeline_variant", "validation_interval_epochs", "threshold_mode", "evaluate_gaussian_blend"}
        if config.pipeline_variant == "legacy"
        else set()
    )
    return config.model_dump(mode="json", exclude=excluded)


def _checkpoint_score(metrics: EpochMetrics) -> float:
    if metrics.quality_metric == "objects":
        return metrics.val_quality_f1
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
