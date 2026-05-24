"""Tiny-overfit диагностика train path для segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest
from mlsystem2.models.api import create_model
from mlsystem2.models.contracts import ModelSpec
from mlsystem2.settings.api import get_settings, load_settings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainRequest
from mlsystem2.train import _trainer


N_POSITIVE = 16
N_NEGATIVE = 16
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-overfit-test")
    parser.add_argument("--config", required=True, help="Путь к YAML-конфигу train pipeline.")
    parser.add_argument("--model", default="segformer_b0", choices=["segformer_b0", "segformer_b2"])
    parser.add_argument("--report", default=None, help="Путь к JSON-отчету.")
    parser.add_argument("--device", default=None, help="Переопределение train.device.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args(argv)

    started = perf_counter()
    config_path = Path(args.config)
    report_path = Path(args.report) if args.report else config_path.parent / "overfit_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    load_settings(config_path)
    settings = get_settings()
    device = args.device or settings.train.device

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
        images, masks, collected = _collect_tiny_dataset(source_loader)
    finally:
        _close_loader(source_loader)

    torch = _torch()
    tiny_dataset = torch.utils.data.TensorDataset(images, masks)
    generator = torch.Generator()
    generator.manual_seed(settings.tile_preparation.seed)
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

    model = create_model(
        ModelSpec(
            name=args.model,
            input_channels=settings.train.input_channels,
            output_channels=settings.train.output_channels,
            pretrained=False,
        )
    )
    model.model.to(torch.device(device))
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=settings.train.weight_decay,
        loss=settings.train.loss,
        focal_alpha=settings.train.focal_alpha,
        pos_weight=settings.train.pos_weight,
        tversky_alpha=settings.train.tversky_alpha,
        tversky_beta=settings.train.tversky_beta,
        threshold=settings.train.threshold,
        early_stopping_patience=args.epochs,
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
            checkpoint_dir=str(report_path.parent / "overfit_checkpoints"),
        )
    )
    final = result.history[-1]
    report = {
        "status": "ok",
        "config": str(config_path),
        "model": args.model,
        "device": device,
        "positive_tiles_used": collected["positive_tiles"],
        "negative_tiles_used": collected["negative_tiles"],
        "initial_f1": initial["f1"],
        "initial_precision": initial["precision"],
        "initial_recall": initial["recall"],
        "final_f1": final.val_pixel_f1,
        "final_precision": final.val_pixel_precision,
        "final_recall": final.val_pixel_recall,
        "final_best_threshold": final.val_best_threshold,
        "final_best_threshold_f1": final.val_best_threshold_pixel_f1,
        "train_loss_history": [item.train_loss for item in result.history],
        "f1_history": [item.val_pixel_f1 for item in result.history],
        "best_threshold_f1_history": [item.val_best_threshold_pixel_f1 for item in result.history],
        "epochs_total": result.epochs_total,
        "training_time_sec": result.training_time_sec,
        "elapsed_sec": perf_counter() - started,
        "best_checkpoint_path": result.best_checkpoint_path,
        "final_checkpoint_path": result.final_checkpoint_path,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _collect_tiny_dataset(loader: object):
    torch = _torch()
    positive_images = []
    positive_masks = []
    negative_images = []
    negative_masks = []

    for batch in loader:
        images, masks = batch[0], batch[1]
        positive_by_tile = (masks > 0.5).flatten(1).sum(dim=1) > 0
        for tile_index, positive in enumerate(positive_by_tile.tolist()):
            if positive and len(positive_images) < N_POSITIVE:
                positive_images.append(images[tile_index].detach().cpu())
                positive_masks.append(masks[tile_index].detach().cpu())
            elif not positive and len(negative_images) < N_NEGATIVE:
                negative_images.append(images[tile_index].detach().cpu())
                negative_masks.append(masks[tile_index].detach().cpu())
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
    }


def _close_loader(loader: object) -> None:
    dataset = getattr(loader, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Для tiny-overfit диагностики требуется PyTorch.") from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
