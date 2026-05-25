"""Tiny-overfit диагностика train path для segmentation."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest
from mlsystem2.models.api import create_model
from mlsystem2.models.contracts import ModelHandle, ModelSpec
from mlsystem2.settings.api import get_settings, load_settings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest
from mlsystem2.train import _trainer
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainRequest


N_POSITIVE = 16
N_NEGATIVE = 16
SYNTHETIC_POSITIVE = 8
SYNTHETIC_NEGATIVE = 8
DEFAULT_EPOCHS = 20
DEFAULT_STEPS = 300
DEFAULT_MATRIX_STEPS = 500
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_TINY_UNET_LEARNING_RATE = 1e-3
DEFAULT_MIN_F1 = 0.8
MODEL_CHOICES = (
    "segformer_b0",
    "segformer_b2",
    "smp_segformer_b0",
    "smp_segformer_b2",
    "tiny_unet_4ch",
)
MATRIX_MODELS = ("segformer_b0", "smp_segformer_b0", "tiny_unet_4ch")
MATRIX_LOSSES = ("bce_dice", "focal_tversky")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-overfit-test")
    parser.add_argument("--config", required=True, help="Путь к YAML-конфигу train pipeline.")
    parser.add_argument("--model", default="segformer_b0", choices=MODEL_CHOICES)
    parser.add_argument("--loss", default=None, choices=["bce_dice", "focal_dice", "focal_tversky"])
    parser.add_argument("--report", default=None, help="Путь к JSON-отчету.")
    parser.add_argument("--device", default=None, help="Переопределение train.device.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--samples-output-dir", default=None)
    parser.add_argument("--export-samples-only", action="store_true")
    parser.add_argument(
        "--positive-selection",
        choices=["first", "largest", "both"],
        default="both",
    )
    parser.add_argument("--export-negative-samples", action="store_true")
    args = parser.parse_args(argv)

    started = perf_counter()
    config_path = Path(args.config)
    default_report = "overfit_matrix_report.json" if args.matrix else "overfit_test_report.json"
    report_path = Path(args.report) if args.report else config_path.parent / default_report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    load_settings(config_path)
    settings = get_settings()
    device = args.device or settings.train.device
    torch = _torch()

    if args.synthetic:
        images, masks, collected, sample_records = _make_synthetic_dataset(
            torch,
            settings.train.input_channels,
        )
        sample_dir = report_path.parent / "synthetic_overfit_samples"
    else:
        images, masks, collected, sample_records = _collect_real_dataset(settings)
        sample_dir = report_path.parent / "overfit_samples"
    sample_previews = _save_sample_previews(images, masks, sample_dir)

    sample_export_report = None
    if args.samples_output_dir is not None:
        groups = _collect_export_groups(
            settings=settings,
            tiny_records=sample_records,
            positive_selection=args.positive_selection,
            export_negative_samples=args.export_negative_samples,
            synthetic=args.synthetic,
        )
        sample_export_report = _export_sample_groups(
            groups=groups,
            output_dir=Path(args.samples_output_dir),
            config_path=config_path,
        )
        if args.export_samples_only:
            print(json.dumps(sample_export_report, ensure_ascii=False, indent=2))
            return 0

    steps = args.steps or (DEFAULT_MATRIX_STEPS if args.matrix else DEFAULT_STEPS)
    if args.matrix:
        report = _run_matrix(
            torch=torch,
            settings=settings,
            images=images,
            masks=masks,
            collected=collected,
            sample_previews=sample_previews,
            report_path=report_path,
            device=device,
            steps=steps,
            epochs=args.epochs,
            min_f1=args.min_f1,
            started=started,
            synthetic=args.synthetic,
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["smp_focal_tversky_passed"] else 1

    loss_name = args.loss or settings.train.loss
    case = _run_case(
        torch=torch,
        settings=settings,
        images=images,
        masks=masks,
        model_name=args.model,
        loss_name=loss_name,
        learning_rate=args.learning_rate,
        steps=steps,
        epochs=args.epochs,
        device=device,
        report_dir=report_path.parent,
        case_name=f"{args.model}_{loss_name}",
        min_f1=args.min_f1,
        catch_errors=True,
    )
    report = {
        **case,
        "config": str(config_path),
        "synthetic": args.synthetic,
        "threshold_sweep_enabled": args.threshold_sweep,
        "positive_tiles_used": collected["positive_tiles"],
        "negative_tiles_used": collected["negative_tiles"],
        "sample_previews": sample_previews,
        "sample_export_report": sample_export_report,
        "elapsed_sec": perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed", False) else 1


def _run_matrix(
    *,
    torch,
    settings,
    images,
    masks,
    collected: dict[str, int],
    sample_previews: list[dict[str, object]],
    report_path: Path,
    device: str,
    steps: int,
    epochs: int | None,
    min_f1: float,
    started: float,
    synthetic: bool,
) -> dict[str, object]:
    cases = []
    for model_name in MATRIX_MODELS:
        for loss_name in MATRIX_LOSSES:
            learning_rate = (
                DEFAULT_TINY_UNET_LEARNING_RATE
                if model_name == "tiny_unet_4ch"
                else DEFAULT_LEARNING_RATE
            )
            cases.append(
                _run_case(
                    torch=torch,
                    settings=settings,
                    images=images,
                    masks=masks,
                    model_name=model_name,
                    loss_name=loss_name,
                    learning_rate=learning_rate,
                    steps=steps,
                    epochs=epochs,
                    device=device,
                    report_dir=report_path.parent,
                    case_name=f"{model_name}_{loss_name}",
                    min_f1=min_f1,
                    catch_errors=True,
                )
            )
    smp_focal = next(
        (
            item
            for item in cases
            if item["model"] == "smp_segformer_b0" and item["loss"] == "focal_tversky"
        ),
        None,
    )
    return {
        "status": "ok" if smp_focal is not None and smp_focal.get("passed") else "failed",
        "synthetic": synthetic,
        "requested_steps": steps,
        "success_min_best_threshold_f1": min_f1,
        "positive_tiles_used": collected["positive_tiles"],
        "negative_tiles_used": collected["negative_tiles"],
        "smp_focal_tversky_passed": bool(smp_focal is not None and smp_focal.get("passed")),
        "cases": cases,
        "sample_previews": sample_previews,
        "elapsed_sec": perf_counter() - started,
    }


def _run_case(
    *,
    torch,
    settings,
    images,
    masks,
    model_name: str,
    loss_name: str,
    learning_rate: float,
    steps: int,
    epochs: int | None,
    device: str,
    report_dir: Path,
    case_name: str,
    min_f1: float,
    catch_errors: bool,
) -> dict[str, object]:
    case_started = perf_counter()
    try:
        tiny_loader, eval_loader = _make_loaders(torch, images, masks, settings.tile_preparation.seed)
        effective_epochs = epochs or max(DEFAULT_EPOCHS, ceil(steps / len(tiny_loader)))
        model = _create_overfit_model(
            torch=torch,
            model_name=model_name,
            input_channels=settings.train.input_channels,
            output_channels=settings.train.output_channels,
        )
        model.model.to(torch.device(device))
        train_config = TrainConfig(
            epochs=effective_epochs,
            batch_size=DEFAULT_BATCH_SIZE,
            device=device,
            learning_rate=learning_rate,
            weight_decay=settings.train.weight_decay,
            loss=loss_name,
            focal_alpha=settings.train.focal_alpha,
            pos_weight=settings.train.pos_weight,
            tversky_alpha=settings.train.tversky_alpha,
            tversky_beta=settings.train.tversky_beta,
            threshold=settings.train.threshold,
            early_stopping_patience=effective_epochs,
        )
        initial = _trainer._validate_epoch(
            torch,
            model.model,
            eval_loader,
            torch.device(device),
            train_config,
            0,
        )
        result = train_model(
            TrainRequest(
                model=model,
                train_loader=tiny_loader,
                val_loader=eval_loader,
                config=train_config,
                checkpoint_dir=str(report_dir / "overfit_checkpoints" / case_name),
            )
        )
        final = result.history[-1]
        best_epoch = max(result.history, key=lambda item: item.val_best_threshold_pixel_f1)
        best_threshold_f1 = best_epoch.val_best_threshold_pixel_f1
        return {
            "status": "ok" if best_threshold_f1 >= min_f1 else "failed",
            "passed": best_threshold_f1 >= min_f1,
            "model": model_name,
            "loss": loss_name,
            "device": device,
            "learning_rate": learning_rate,
            "requested_steps": steps,
            "actual_optimizer_steps": sum(item.train_optimizer_steps for item in result.history),
            "epochs_total": result.epochs_total,
            "initial_f1": initial["f1"],
            "initial_precision": initial["precision"],
            "initial_recall": initial["recall"],
            "final_f1": final.val_pixel_f1,
            "final_precision": final.val_pixel_precision,
            "final_recall": final.val_pixel_recall,
            "final_best_threshold": final.val_best_threshold,
            "final_best_threshold_f1": final.val_best_threshold_pixel_f1,
            "best_epoch": best_epoch.epoch,
            "best_threshold": best_epoch.val_best_threshold,
            "best_threshold_f1": best_threshold_f1,
            "best_threshold_precision": best_epoch.val_best_threshold_precision,
            "best_threshold_recall": best_epoch.val_best_threshold_recall,
            "final_prob_positive_mean": final.val_prob_positive_mean,
            "final_prob_negative_mean": final.val_prob_negative_mean,
            "final_train_loss_focal": final.train_loss_focal,
            "final_train_loss_tversky": final.train_loss_tversky,
            "final_train_loss_bce": final.train_loss_bce,
            "final_train_loss_dice": final.train_loss_dice,
            "train_loss_history": [item.train_loss for item in result.history],
            "f1_history": [item.val_pixel_f1 for item in result.history],
            "best_threshold_f1_history": [
                item.val_best_threshold_pixel_f1 for item in result.history
            ],
            "training_time_sec": result.training_time_sec,
            "elapsed_sec": perf_counter() - case_started,
            "best_checkpoint_path": result.best_checkpoint_path,
            "final_checkpoint_path": result.final_checkpoint_path,
        }
    except Exception as exc:
        if not catch_errors:
            raise
        return {
            "status": "error",
            "passed": False,
            "model": model_name,
            "loss": loss_name,
            "error": str(exc),
            "elapsed_sec": perf_counter() - case_started,
        }


def _collect_real_dataset(settings):
    dataset_result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=settings.dataset.images_dir,
            scenes_file=settings.dataset.scenes_file,
            annotation_file=settings.dataset.annotation_file,
            val_fraction=settings.dataset.val_fraction,
        )
    )
    if dataset_result.dataset is None:
        raise SystemExit(f"dataset_preparing failed: {dataset_result.report.errors}")

    # mode=val отключает train augmentation и shuffle, но использует тот же train VRT.
    source_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=dataset_result.dataset.train_vrt_xml,
            annotation_file=dataset_result.dataset.annotation_file,
            batch_size=settings.train.batch_size,
            mode="val",
        )
    )
    try:
        return _collect_tiny_dataset(source_loader)
    finally:
        _close_loader(source_loader)


def _collect_tiny_dataset(loader: object):
    torch = _torch()
    positive_images = []
    positive_masks = []
    positive_records = []
    negative_images = []
    negative_masks = []
    negative_records = []

    for batch in loader:
        images, masks = batch[0], batch[1]
        positive_by_tile = (masks > 0.5).flatten(1).sum(dim=1) > 0
        for tile_index, positive in enumerate(positive_by_tile.tolist()):
            if positive and len(positive_images) < N_POSITIVE:
                image = images[tile_index].detach().cpu()
                mask = masks[tile_index].detach().cpu()
                positive_images.append(image)
                positive_masks.append(mask)
                positive_records.append(_sample_record(image, mask, source="overfit_collector"))
            elif not positive and len(negative_images) < N_NEGATIVE:
                image = images[tile_index].detach().cpu()
                mask = masks[tile_index].detach().cpu()
                negative_images.append(image)
                negative_masks.append(mask)
                negative_records.append(_sample_record(image, mask, source="overfit_collector"))
        if len(positive_images) >= N_POSITIVE and len(negative_images) >= N_NEGATIVE:
            break

    if len(positive_images) < N_POSITIVE or len(negative_images) < N_NEGATIVE:
        raise RuntimeError(
            "Не удалось собрать tiny-overfit dataset: "
            f"positive={len(positive_images)}/{N_POSITIVE}, "
            f"negative={len(negative_images)}/{N_NEGATIVE}."
        )

    images = torch.stack([*positive_images, *negative_images], dim=0)
    masks = torch.stack([*positive_masks, *negative_masks], dim=0)
    return images, masks, {
        "positive_tiles": len(positive_images),
        "negative_tiles": len(negative_images),
    }, [*positive_records, *negative_records]


def _make_synthetic_dataset(torch, input_channels: int):
    size = 128
    images = torch.zeros(
        (SYNTHETIC_POSITIVE + SYNTHETIC_NEGATIVE, input_channels, size, size),
        dtype=torch.float32,
    )
    masks = torch.zeros((SYNTHETIC_POSITIVE + SYNTHETIC_NEGATIVE, 1, size, size), dtype=torch.float32)
    for index in range(SYNTHETIC_POSITIVE):
        y = 16 + (index % 4) * 18
        x = 12 + (index // 4) * 42
        height = 26 + (index % 3) * 4
        width = 30 + (index % 2) * 6
        masks[index, :, y : y + height, x : x + width] = 1.0
        images[index, 0, y : y + height, x : x + width] = 255.0
        if input_channels > 1:
            images[index, 1] = 25.0
            images[index, 1, y : y + height, x : x + width] = 180.0
        if input_channels > 2:
            images[index, 2, :, :] = 10.0 + index
            images[index, 2, y : y + height, x : x + width] = 120.0
        if input_channels > 3:
            images[index, 3, y : y + height, x : x + width] = 220.0
    records = [
        _sample_record(
            images[index],
            masks[index],
            source="synthetic",
            source_index=index,
        )
        for index in range(int(images.shape[0]))
    ]
    return images, masks, {
        "positive_tiles": SYNTHETIC_POSITIVE,
        "negative_tiles": SYNTHETIC_NEGATIVE,
    }, records


def _make_loaders(torch, images, masks, seed: int):
    tiny_dataset = torch.utils.data.TensorDataset(images, masks)
    generator = torch.Generator()
    generator.manual_seed(seed)
    tiny_loader = torch.utils.data.DataLoader(
        tiny_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    eval_loader = torch.utils.data.DataLoader(
        tiny_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
    )
    return tiny_loader, eval_loader


def _collect_export_groups(
    *,
    settings,
    tiny_records: list[dict[str, object]],
    positive_selection: str,
    export_negative_samples: bool,
    synthetic: bool,
) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    if positive_selection in {"first", "both"}:
        groups["positive_first_16"] = [
            record for record in tiny_records if bool(record["positive"])
        ][:N_POSITIVE]
    if positive_selection in {"largest", "both"}:
        if synthetic:
            largest = [record for record in tiny_records if bool(record["positive"])]
            largest = sorted(
                largest,
                key=lambda item: int(item["positive_mask_pixels"]),
                reverse=True,
            )
        else:
            largest = _collect_largest_positive_records(settings)
        groups["positive_largest_16"] = largest[:N_POSITIVE]
    if export_negative_samples:
        groups["negative_16"] = [
            record for record in tiny_records if not bool(record["positive"])
        ][:N_NEGATIVE]
    return groups


def _collect_largest_positive_records(settings) -> list[dict[str, object]]:
    torch = _torch()
    dataset_result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=settings.dataset.images_dir,
            scenes_file=settings.dataset.scenes_file,
            annotation_file=settings.dataset.annotation_file,
            val_fraction=settings.dataset.val_fraction,
        )
    )
    if dataset_result.dataset is None:
        raise SystemExit(f"dataset_preparing failed: {dataset_result.report.errors}")

    source_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=dataset_result.dataset.train_vrt_xml,
            annotation_file=dataset_result.dataset.annotation_file,
            batch_size=settings.train.batch_size,
            mode="val",
        )
    )
    dataset = getattr(source_loader, "dataset", None)
    if dataset is None:
        raise RuntimeError("DataLoader не вернул dataset для diagnostic samples export.")

    try:
        sequential_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=settings.train.batch_size,
            shuffle=False,
            num_workers=settings.tile_preparation.num_workers,
            collate_fn=_collate_export_batch,
        )
        best_records: list[dict[str, object]] = []
        for batch_index, batch in enumerate(sequential_loader):
            images, masks = batch[0], batch[1]
            positive_pixels = (masks > 0.5).flatten(1).sum(dim=1)
            for tile_index, pixel_count_tensor in enumerate(positive_pixels.tolist()):
                pixel_count = int(pixel_count_tensor)
                if pixel_count <= 0:
                    continue
                source_index = batch_index * settings.train.batch_size + tile_index
                image = images[tile_index].detach().cpu()
                mask = masks[tile_index].detach().cpu()
                best_records.append(
                    _sample_record(
                        image,
                        mask,
                        source="largest_positive_scan",
                        source_index=source_index,
                        window=_dataset_window(dataset, source_index),
                    )
                )
                best_records.sort(
                    key=lambda item: int(item["positive_mask_pixels"]),
                    reverse=True,
                )
                del best_records[N_POSITIVE:]
        return best_records
    finally:
        _close_loader(source_loader)


def _collate_export_batch(samples):
    torch = _torch()
    images = torch.stack(
        [torch.as_tensor(sample[0], dtype=torch.float32) for sample in samples],
        dim=0,
    )
    masks = torch.stack(
        [torch.as_tensor(sample[1], dtype=torch.float32) for sample in samples],
        dim=0,
    )
    return images, masks


def _dataset_window(dataset: object, index: int) -> dict[str, int] | None:
    windows = getattr(dataset, "_windows", None)
    if windows is None or index >= len(windows):
        return None
    window = windows[index]
    return {
        "x": int(getattr(window, "x")),
        "y": int(getattr(window, "y")),
        "width": int(getattr(window, "width")),
        "height": int(getattr(window, "height")),
    }


def _sample_record(
    image,
    mask,
    *,
    source: str,
    source_index: int | None = None,
    window: dict[str, int] | None = None,
) -> dict[str, object]:
    positive_mask_pixels = int((mask > 0.5).sum().item())
    height = int(mask.shape[-2])
    width = int(mask.shape[-1])
    return {
        "image": image.detach().cpu().clone(),
        "mask": mask.detach().cpu().clone(),
        "source": source,
        "source_index": source_index,
        "window": window,
        "positive": positive_mask_pixels > 0,
        "positive_mask_pixels": positive_mask_pixels,
        "positive_fraction": positive_mask_pixels / float(height * width),
    }


def _export_sample_groups(
    *,
    groups: dict[str, list[dict[str, object]]],
    output_dir: Path,
    config_path: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_name in ("positive_first_16", "positive_largest_16", "negative_16"):
        group_dir = output_dir / group_name
        if group_dir.exists():
            import shutil

            shutil.rmtree(group_dir)

    report_groups = {}
    for group_name, records in groups.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        for index, record in enumerate(records):
            exported.append(_export_sample_record(record, group_dir, group_name, index))
        report_groups[group_name] = _group_summary(exported)

    report = {
        "status": "ok",
        "config": str(config_path),
        "output_dir": str(output_dir),
        "groups": report_groups,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _export_sample_record(
    record: dict[str, object],
    group_dir: Path,
    group_name: str,
    index: int,
) -> dict[str, object]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Для экспорта PNG требуется pillow.") from exc

    torch = _torch()
    image = record["image"]
    mask = record["mask"]
    image_np = image.numpy()
    mask_np = mask.numpy()

    sample_dir = group_dir / f"sample_{index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    preview = _preview_rgb_uint8(image_np)
    mask_u8 = (mask_np[0].clip(0.0, 1.0) * 255).astype("uint8")
    edge = _mask_edge(mask_np)
    overlay = preview.copy()
    overlay[edge] = [255, 0, 0]

    Image.fromarray(preview, mode="RGB").save(sample_dir / "preview_rgb.png")
    Image.fromarray(overlay, mode="RGB").save(sample_dir / "preview_mask_overlay.png")
    Image.fromarray(mask_u8, mode="L").save(sample_dir / "mask.png")
    for channel_index, channel in enumerate(image_np):
        Image.fromarray(_stretch_channel_to_uint8(channel), mode="L").save(
            sample_dir / f"channel_{channel_index:02d}.png"
        )

    meta = _sample_export_meta(record, group_name, index)
    torch.save(
        {
            "image": image,
            "mask": mask,
            "meta": meta,
        },
        sample_dir / "sample.pt",
    )
    (sample_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def _sample_export_meta(
    record: dict[str, object],
    group_name: str,
    index: int,
) -> dict[str, object]:
    image = record["image"]
    mask = record["mask"]
    image_np = image.numpy()
    mask_np = mask.numpy()
    return {
        "group": group_name,
        "sample_index": index,
        "source": record.get("source"),
        "source_index": record.get("source_index"),
        "positive": bool(record["positive"]),
        "positive_mask_pixels": int(record["positive_mask_pixels"]),
        "positive_fraction": float(record["positive_fraction"]),
        "image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "image_dtype": str(image_np.dtype),
        "mask_dtype": str(mask_np.dtype),
        "channel_min": _channel_stat(image_np, "min"),
        "channel_max": _channel_stat(image_np, "max"),
        "channel_mean": _channel_stat(image_np, "mean"),
        "all_channels_equal": _all_channels_equal(image_np),
        "nonzero_channel_count": _nonzero_channel_count(image_np),
        "window": record.get("window"),
        "alignment_stats": _sample_alignment_stats(image_np, mask_np),
    }


def _group_summary(records: list[dict[str, object]]) -> dict[str, object]:
    fractions = [float(item["positive_fraction"]) for item in records]
    pixels = [int(item["positive_mask_pixels"]) for item in records]
    return {
        "count": len(records),
        "positive_mask_pixels_min": min(pixels) if pixels else None,
        "positive_mask_pixels_max": max(pixels) if pixels else None,
        "positive_mask_pixels_mean": sum(pixels) / len(pixels) if pixels else None,
        "positive_fraction_min": min(fractions) if fractions else None,
        "positive_fraction_max": max(fractions) if fractions else None,
        "positive_fraction_mean": sum(fractions) / len(fractions) if fractions else None,
    }


def _create_overfit_model(
    *,
    torch,
    model_name: str,
    input_channels: int,
    output_channels: int,
) -> ModelHandle:
    if model_name == "tiny_unet_4ch":
        return ModelHandle(
            spec=ModelSpec(name=model_name, input_channels=input_channels, output_channels=output_channels),
            model=_TinyUNet(input_channels=input_channels, output_channels=output_channels, torch=torch),
        )
    return create_model(
        ModelSpec(
            name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            pretrained=False,
        )
    )


class _TinyUNet:
    def __new__(cls, *, input_channels: int, output_channels: int, torch):
        class TinyUNetImpl(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(16, output_channels, kernel_size=1),
                )

            def forward(self, images):
                return self.net(images)

        return TinyUNetImpl()


def _close_loader(loader: object) -> None:
    dataset = getattr(loader, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _save_sample_previews(images, masks, output_dir: Path) -> list[dict[str, object]]:
    try:
        from PIL import Image
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    max_samples = min(16, int(images.shape[0]))
    for index in range(max_samples):
        image = images[index].numpy()
        mask = masks[index].numpy()
        preview = _preview_rgb_uint8(image)
        mask_u8 = (mask[0].clip(0.0, 1.0) * 255).astype("uint8")
        edge = _mask_edge(mask)
        overlay = preview.copy()
        overlay[edge] = [255, 0, 0]
        stats = _sample_alignment_stats(image, mask)
        preview_path = output_dir / f"sample_{index:04d}_preview_rgb.png"
        mask_path = output_dir / f"sample_{index:04d}_mask.png"
        overlay_path = output_dir / f"sample_{index:04d}_overlay.png"
        stats_path = output_dir / f"sample_{index:04d}_stats.json"
        Image.fromarray(preview, mode="RGB").save(preview_path)
        Image.fromarray(mask_u8, mode="L").save(mask_path)
        Image.fromarray(overlay, mode="RGB").save(overlay_path)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "sample_index": index,
                "positive": bool(mask.sum() > 0),
                "preview_rgb": str(preview_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "stats": str(stats_path),
                "stats_summary": stats,
            }
        )
    return records


def _sample_alignment_stats(image_chw, mask_chw) -> dict[str, object]:
    import numpy as np

    mask = mask_chw[0] > 0.5
    inside_pixels = int(np.count_nonzero(mask))
    outside_pixels = int(mask.size - inside_pixels)
    inside_means = []
    outside_means = []
    abs_diffs = []
    for channel in image_chw:
        finite = np.isfinite(channel)
        inside = channel[mask & finite]
        outside = channel[(~mask) & finite]
        inside_mean = float(inside.mean()) if inside.size else None
        outside_mean = float(outside.mean()) if outside.size else None
        inside_means.append(inside_mean)
        outside_means.append(outside_mean)
        abs_diffs.append(
            None
            if inside_mean is None or outside_mean is None
            else abs(inside_mean - outside_mean)
        )
    return {
        "positive_mask_pixels": inside_pixels,
        "negative_mask_pixels": outside_pixels,
        "inside_channel_mean": inside_means,
        "outside_channel_mean": outside_means,
        "inside_outside_abs_diff": abs_diffs,
    }


def _channel_stat(image_chw, stat: str) -> list[float | None]:
    import numpy as np

    values: list[float | None] = []
    for channel in image_chw:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            values.append(None)
            continue
        if stat == "min":
            values.append(float(finite.min()))
        elif stat == "max":
            values.append(float(finite.max()))
        elif stat == "mean":
            values.append(float(finite.mean()))
        else:
            raise RuntimeError(f"Неподдерживаемая статистика канала: {stat}")
    return values


def _all_channels_equal(image_chw) -> bool:
    import numpy as np

    if image_chw.shape[0] <= 1:
        return False
    first = image_chw[0]
    return all(bool(np.array_equal(first, image_chw[index])) for index in range(1, image_chw.shape[0]))


def _nonzero_channel_count(image_chw) -> int:
    import numpy as np

    return sum(1 for channel in image_chw if int(np.count_nonzero(channel)) > 0)


def _preview_rgb_uint8(image_chw):
    import numpy as np

    if image_chw.shape[0] >= 3:
        image_hwc = image_chw[:3].transpose(1, 2, 0)
    else:
        image_hwc = np.repeat(image_chw[0:1], 3, axis=0).transpose(1, 2, 0)
    channels = [_stretch_channel_to_uint8(image_hwc[:, :, index]) for index in range(3)]
    return np.stack(channels, axis=2)


def _stretch_channel_to_uint8(channel):
    import numpy as np

    image = channel.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    nonzero = finite[finite > 0]
    values = nonzero if nonzero.size else finite
    min_value = float(np.percentile(values, 1))
    max_value = float(np.percentile(values, 99))
    if max_value <= min_value:
        min_value = float(values.min())
        max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - min_value) / (max_value - min_value) * 255.0, 0.0, 255.0).astype(
        np.uint8
    )


def _mask_edge(mask_chw):
    import numpy as np

    mask = mask_chw[0] > 0.5
    if not bool(np.any(mask)):
        return np.zeros(mask.shape, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    return mask & ~interior


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Для tiny-overfit диагностики требуется PyTorch.") from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
