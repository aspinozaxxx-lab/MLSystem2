"""Чтение публичного инвентаря MLMarkup для UI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .contracts import ClassInfo, DatasetInfo


CUSTOM_KEY = "custom"
CUSTOM_NAME = "Custom"


def list_datasets(mlmarkup_root: Path) -> list[DatasetInfo]:
    datasets = [_dataset_from_folder(path) for path in _class_dirs(mlmarkup_root)]
    datasets = [item for item in datasets if item is not None]
    datasets.sort(key=lambda item: item.name.lower())
    datasets.append(DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True))
    return datasets


def list_classes(mlmarkup_root: Path) -> list[ClassInfo]:
    classes = [
        ClassInfo(
            key=path.name,
            name=path.name,
            updated_at=_path_updated_at(path),
            is_custom=False,
        )
        for path in _class_dirs(mlmarkup_root)
    ]
    classes.sort(key=lambda item: item.name.lower())
    classes.append(ClassInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True))
    return classes


def find_class(mlmarkup_root: Path, class_key: str) -> ClassInfo | None:
    for item in list_classes(mlmarkup_root):
        if item.key == class_key:
            return item
    return None


def _class_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]


def _dataset_from_folder(path: Path) -> DatasetInfo | None:
    scenes_file = _first_file(path, ".txt")
    annotation_file = _first_file(path, ".geojson")
    return DatasetInfo(
        key=path.name,
        name=path.name,
        path=str(path),
        scenes_file=str(scenes_file) if scenes_file else None,
        annotation_file=str(annotation_file) if annotation_file else None,
        updated_at=_path_updated_at(path),
    )


def _first_file(path: Path, suffix: str) -> Path | None:
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)
    return files[0] if files else None


def _path_updated_at(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None

