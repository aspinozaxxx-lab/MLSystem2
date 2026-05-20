"""Локальная проверка dataset_preparing через публичный API."""

from __future__ import annotations

import json
from pathlib import Path

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest


IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestation")
DATASET_DIR = Path(r"D:\Projects\MLMarkup\Вырубки")
OUT_DIR = Path(r"D:\Projects\test")

SCENES_FILE = DATASET_DIR / "deforestation.txt"
ANNOTATION_FILE = DATASET_DIR / "deforestation.geojson"

VAL_FRACTION = 0.2


def main() -> int:
    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(IMAGES_DIR),
            scenes_file=str(SCENES_FILE),
            annotation_file=str(ANNOTATION_FILE),
            val_fraction=VAL_FRACTION,
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "preparation_report.json").write_text(
        json.dumps(result.report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if result.dataset is None:
        return 1

    (OUT_DIR / "train.vrt").write_text(result.dataset.train_vrt_xml, encoding="utf-8")
    (OUT_DIR / "val.vrt").write_text(result.dataset.val_vrt_xml, encoding="utf-8")

    return 0 if result.report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())