"""Тестовый запуск подготовки датасета."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest


def main() -> int:
    args = _parse_args()
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=args.images_dir,
            scenes_file=args.scenes_file,
            annotation_file=args.annotation_file,
            val_fraction=args.val_fraction,
        )
    )
    finished_at = datetime.now().isoformat(timespec="seconds")
    elapsed_sec = time.perf_counter() - started

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "preparation_report.json", result.report.model_dump(mode="json"))
    _write_json(
        out_dir / "preparation_timing.json",
        {
            "elapsed_sec": elapsed_sec,
            "started_at": started_at,
            "finished_at": finished_at,
        },
    )

    if result.dataset is not None:
        (out_dir / "train.vrt").write_text(result.dataset.train_vrt_xml, encoding="utf-8")
        (out_dir / "val.vrt").write_text(result.dataset.val_vrt_xml, encoding="utf-8")

    return 0 if result.report.status == "ok" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Тестовый запуск подготовки датасета.")
    parser.add_argument("--images-dir", required=True, help="Директория raster-снимков.")
    parser.add_argument("--scenes-file", required=True, help="Файл списка сцен.")
    parser.add_argument("--annotation-file", required=True, help="Файл разметки.")
    parser.add_argument("--val-fraction", required=True, type=float, help="Доля validation.")
    parser.add_argument("--out-dir", default=r"E:\Projects\test", help="Директория выходных файлов.")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
