"""Проверка нарезки тайлов на полностью черные tiles."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import yaml

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import (
    DatasetClassRequest,
    DatasetPreparationRequest,
    PreparedDataset,
)
from mlsystem2.settings.api import load_settings
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import (
    TileClassAnnotation,
    TileDataloaderRequest,
    TileSplitRequest,
)


DEFAULT_EPS = 1e-6
DEFAULT_PROGRESS_EVERY_BATCHES = 100
DEFAULT_MAX_EXAMPLES = 20


@dataclass
class BlackTileExample:
    split: str
    batch_index: int
    tile_index_in_batch: int
    sequential_tile_index: int
    reason: str


@dataclass
class SplitScanReport:
    split: str
    dataset_tiles: int | None = None
    batches: int = 0
    tiles: int = 0
    black_tiles: int = 0
    nonfinite_tiles: int = 0
    elapsed_sec: float = 0.0
    examples: list[BlackTileExample] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mlsystem2.cli.tiling_test_for_black",
        description="Проверить, что tile_preparation не возвращает полностью черные tiles.",
    )
    parser.add_argument("--config", required=True, help="Путь к YAML-конфигу.")
    parser.add_argument(
        "--report",
        default=None,
        help="Путь к JSON-отчету. По умолчанию пишется в runtime.logs_root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size проверки. По умолчанию используется train.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Переопределить tile_preparation.num_workers для проверки.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help="Порог, ниже которого pixel считается черным.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_BATCHES,
        help="Как часто печатать прогресс по batch.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Сколько примеров проблемных tiles сохранить в отчет.",
    )
    args = parser.parse_args(argv)

    total_started = perf_counter()
    config_path = Path(args.config).resolve()
    settings = load_settings(config_path)
    report_path = _report_path(settings, args.report)
    scan_config_path = report_path.with_suffix(".scan.yaml")
    timings: dict[str, float] = {}
    split_reports: list[SplitScanReport] = []

    try:
        dataset_started = perf_counter()
        dataset_result = prepare_dataset(_dataset_request(settings))
        timings["dataset_preparing_sec"] = perf_counter() - dataset_started

        if dataset_result.dataset is None or dataset_result.report.status != "ok":
            report = _final_report(
                status="error",
                config_path=config_path,
                scan_config_path=scan_config_path,
                timings=timings,
                total_started=total_started,
                dataset_report=dataset_result.report.model_dump(mode="json"),
                split_reports=split_reports,
                error="dataset_preparing не вернул готовый датасет.",
            )
            _write_json(report_path, report)
            print(f"status=error report={report_path}")
            return 1

        _write_scan_settings(
            settings,
            scan_config_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        scan_settings = load_settings(scan_config_path)

        for split in ("train", "val"):
            loader_started = perf_counter()
            loader = create_tile_dataloader(
                _tile_request(
                    scan_settings,
                    dataset_result.dataset,
                    batch_size=args.batch_size or scan_settings.train.batch_size,
                    split=split,
                )
            )
            timings[f"{split}_dataloader_creation_sec"] = perf_counter() - loader_started
            split_reports.append(
                _scan_loader(
                    loader,
                    split=split,
                    eps=args.eps,
                    progress_every_batches=max(1, args.progress_every_batches),
                    max_examples=max(0, args.max_examples),
                )
            )
            _close_loader_dataset(loader)

        black_tiles = sum(item.black_tiles for item in split_reports)
        nonfinite_tiles = sum(item.nonfinite_tiles for item in split_reports)
        status = "ok" if black_tiles == 0 and nonfinite_tiles == 0 else "error"
        error = None
        if status == "error":
            error = (
                "Найдены пустые или некорректные tiles: "
                f"black_tiles={black_tiles}, nonfinite_tiles={nonfinite_tiles}."
            )
        report = _final_report(
            status=status,
            config_path=config_path,
            scan_config_path=scan_config_path,
            timings=timings,
            total_started=total_started,
            dataset_report=dataset_result.report.model_dump(mode="json"),
            split_reports=split_reports,
            error=error,
        )
        _write_json(report_path, report)
        print(f"status={status} report={report_path}")
        print(f"black_tiles={black_tiles} nonfinite_tiles={nonfinite_tiles}")
        return 0 if status == "ok" else 1
    except Exception as exc:  # noqa: BLE001
        report = _final_report(
            status="error",
            config_path=config_path,
            scan_config_path=scan_config_path,
            timings=timings,
            total_started=total_started,
            dataset_report=None,
            split_reports=split_reports,
            error=str(exc),
        )
        _write_json(report_path, report)
        print(f"status=error report={report_path} error={exc}")
        return 1


def _dataset_request(settings: SystemSettings) -> DatasetPreparationRequest:
    dataset = settings.dataset
    if dataset.classes:
        return DatasetPreparationRequest(
            images_dir=dataset.images_dir,
            classes=[
                DatasetClassRequest(
                    slug=item.slug,
                    name=item.name,
                    scenes_file=item.scenes_file,
                    annotation_file=item.annotation_file,
                    priority=item.priority,
                )
                for item in dataset.classes
            ],
            val_fraction=dataset.val_fraction,
        )
    return DatasetPreparationRequest(
        images_dir=dataset.images_dir,
        scenes_file=dataset.scenes_file,
        annotation_file=dataset.annotation_file,
        val_fraction=dataset.val_fraction,
    )


def _tile_request(
    settings: SystemSettings,
    dataset: PreparedDataset,
    *,
    batch_size: int,
    split: Literal["train", "val"],
) -> TileDataloaderRequest:
    kwargs: dict[str, Any] = {
        "vrt_xml": _vrt_xml(dataset, split),
        "batch_size": batch_size,
        "mode": split,
        "tile_split": _tile_split(settings),
    }
    if dataset.class_annotations:
        kwargs["class_annotations"] = [
            TileClassAnnotation(
                class_id=item.class_id,
                slug=item.slug,
                name=item.name,
                annotation_file=item.annotation_file,
                priority=item.priority,
            )
            for item in dataset.class_annotations
        ]
    else:
        kwargs["annotation_file"] = dataset.annotation_file
    return TileDataloaderRequest(**kwargs)


def _vrt_xml(
    dataset: PreparedDataset,
    split: Literal["train", "val"],
) -> str:
    return dataset.pool_vrt_xml or (dataset.train_vrt_xml if split == "train" else dataset.val_vrt_xml)


def _tile_split(settings: SystemSettings) -> TileSplitRequest:
    return TileSplitRequest(
        val_fraction=settings.dataset.val_fraction,
        seed=settings.tile_preparation.seed,
    )


def _scan_loader(
    loader: object,
    *,
    split: str,
    eps: float,
    progress_every_batches: int,
    max_examples: int,
) -> SplitScanReport:
    started = perf_counter()
    report = SplitScanReport(split=split, dataset_tiles=_safe_dataset_len(loader))
    for batch_index, batch in enumerate(loader, start=1):
        images = batch[0]
        black_flags, nonfinite_flags = _tile_quality_flags(images, eps=eps)
        batch_tiles = len(black_flags)
        seen_before = report.tiles
        report.batches += 1
        report.tiles += batch_tiles
        report.black_tiles += sum(1 for item in black_flags if item)
        report.nonfinite_tiles += sum(1 for item in nonfinite_flags if item)
        _append_examples(
            report,
            batch_index=batch_index,
            seen_before=seen_before,
            black_flags=black_flags,
            nonfinite_flags=nonfinite_flags,
            max_examples=max_examples,
        )
        if batch_index % progress_every_batches == 0:
            print(
                f"{split}: batches={report.batches} tiles={report.tiles} "
                f"black={report.black_tiles} nonfinite={report.nonfinite_tiles}"
            )
    report.elapsed_sec = perf_counter() - started
    print(
        f"{split}: done batches={report.batches} tiles={report.tiles} "
        f"black={report.black_tiles} nonfinite={report.nonfinite_tiles} "
        f"elapsed_sec={report.elapsed_sec:.3f}"
    )
    return report


def _tile_quality_flags(images: object, *, eps: float) -> tuple[list[bool], list[bool]]:
    import torch

    tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
    if tensor.ndim < 2:
        raise ValueError(f"Ожидался batch tensor с размерностью >=2, получено {tensor.ndim}.")
    flat = tensor.reshape(tensor.shape[0], -1)
    finite = torch.isfinite(flat)
    has_signal = torch.any(finite & (torch.abs(flat) > eps), dim=1)
    all_finite = torch.all(finite, dim=1)
    return (~has_signal).cpu().tolist(), (~all_finite).cpu().tolist()


def _append_examples(
    report: SplitScanReport,
    *,
    batch_index: int,
    seen_before: int,
    black_flags: list[bool],
    nonfinite_flags: list[bool],
    max_examples: int,
) -> None:
    if len(report.examples) >= max_examples:
        return
    for index, (is_black, is_nonfinite) in enumerate(zip(black_flags, nonfinite_flags, strict=True)):
        if not is_black and not is_nonfinite:
            continue
        reason = "black"
        if is_black and is_nonfinite:
            reason = "black+nonfinite"
        elif is_nonfinite:
            reason = "nonfinite"
        report.examples.append(
            BlackTileExample(
                split=report.split,
                batch_index=batch_index,
                tile_index_in_batch=index,
                sequential_tile_index=seen_before + index,
                reason=reason,
            )
        )
        if len(report.examples) >= max_examples:
            return


def _write_scan_settings(
    settings: SystemSettings,
    path: Path,
    *,
    batch_size: int | None,
    num_workers: int | None,
) -> None:
    payload = settings.model_dump(mode="json")
    payload["tile_preparation"]["augmentation_level"] = 0
    if num_workers is not None:
        payload["tile_preparation"]["num_workers"] = num_workers
    if batch_size is not None:
        payload["train"]["batch_size"] = batch_size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _report_path(settings: SystemSettings, raw_report_path: str | None) -> Path:
    if raw_report_path:
        return Path(raw_report_path).resolve()
    return Path(settings.runtime.logs_root).resolve() / "tiling_test_for_black_report.json"


def _final_report(
    *,
    status: Literal["ok", "error"],
    config_path: Path,
    scan_config_path: Path,
    timings: dict[str, float],
    total_started: float,
    dataset_report: dict[str, Any] | None,
    split_reports: list[SplitScanReport],
    error: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "error": error,
        "config_path": str(config_path),
        "scan_config_path": str(scan_config_path),
        "timings": {
            **timings,
            "total_sec": perf_counter() - total_started,
        },
        "dataset_report": dataset_report,
        "splits": [asdict(item) for item in split_reports],
    }


def _safe_dataset_len(loader: object) -> int | None:
    dataset = getattr(loader, "dataset", None)
    try:
        return int(len(dataset))
    except TypeError:
        return None


def _close_loader_dataset(loader: object) -> None:
    dataset = getattr(loader, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
