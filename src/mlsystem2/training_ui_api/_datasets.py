"""Чтение публичного инвентаря MLMarkup для UI."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .contracts import ClassInfo, DatasetInfo


CUSTOM_KEY = "custom"
CUSTOM_NAME = "Custom"
DEFAULT_VARIANT = "main"


def list_datasets(mlmarkup_root: Path) -> list[DatasetInfo]:
    datasets = [
        variant
        for class_dir in _class_dirs(mlmarkup_root)
        for variant in _datasets_from_class_folder(class_dir, mlmarkup_root)
    ]
    datasets.sort(key=lambda item: item.name.lower())
    datasets.append(DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True))
    return datasets


def list_classes(mlmarkup_root: Path) -> list[ClassInfo]:
    classes: list[ClassInfo] = []
    for path in _class_dirs(mlmarkup_root):
        variants = _datasets_from_class_folder(path, mlmarkup_root)
        if not variants:
            continue
        classes.append(
            ClassInfo(
                key=path.name,
                name=path.name,
                updated_at=_latest_updated_at(variants),
                variants=variants,
                is_custom=False,
            )
        )
    classes.sort(key=lambda item: item.name.lower())
    custom_dataset = DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True)
    classes.append(
        ClassInfo(
            key=CUSTOM_KEY,
            name=CUSTOM_NAME,
            variants=[custom_dataset],
            is_custom=True,
        )
    )
    return classes


def find_class(mlmarkup_root: Path, class_key: str) -> ClassInfo | None:
    for item in list_classes(mlmarkup_root):
        if item.key == class_key:
            return item
    return None


def find_dataset(mlmarkup_root: Path, dataset_key: str) -> DatasetInfo | None:
    for item in list_datasets(mlmarkup_root):
        if item.key == dataset_key:
            return item
    return None


def _class_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]


def _datasets_from_class_folder(class_path: Path, repo_root: Path) -> list[DatasetInfo]:
    variant_paths = [
        path
        for path in class_path.iterdir()
        if path.is_dir() and not path.name.startswith(".") and _looks_like_dataset_folder(path)
    ]
    if not variant_paths and _looks_like_dataset_folder(class_path):
        variant_paths = [class_path]
    variant_paths.sort(key=lambda item: (item.name != DEFAULT_VARIANT, item.name.lower()))
    return [
        _dataset_from_variant_folder(
            class_path=class_path,
            variant_path=variant_path,
            repo_root=repo_root,
        )
        for variant_path in variant_paths
    ]


def _dataset_from_variant_folder(
    *,
    class_path: Path,
    variant_path: Path,
    repo_root: Path,
) -> DatasetInfo:
    variant_name = variant_path.name if variant_path != class_path else DEFAULT_VARIANT
    dataset_name = _dataset_display_name(class_path.name, variant_name)
    scenes_file = _first_file(variant_path, ".txt")
    annotation_file = _first_file(variant_path, ".geojson")
    return DatasetInfo(
        key=dataset_name,
        name=dataset_name,
        class_key=class_path.name,
        class_name=class_path.name,
        variant_key=variant_name,
        variant_name=variant_name,
        path=str(variant_path),
        scenes_file=str(scenes_file) if scenes_file else None,
        annotation_file=str(annotation_file) if annotation_file else None,
        updated_at=_path_updated_at(variant_path, repo_root),
    )


def _dataset_display_name(class_name: str, variant_name: str) -> str:
    return f"{class_name}\\{variant_name}"


def _looks_like_dataset_folder(path: Path) -> bool:
    return _first_file(path, ".txt") is not None or _first_file(path, ".geojson") is not None


def _first_file(path: Path, suffix: str) -> Path | None:
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)
    return files[0] if files else None


def _latest_updated_at(datasets: list[DatasetInfo]) -> datetime | None:
    values = [item.updated_at for item in datasets if item.updated_at is not None]
    return max(values) if values else None


def _path_updated_at(path: Path, repo_root: Path) -> datetime | None:
    return _git_path_updated_at(path, repo_root) or _filesystem_path_updated_at(path)


def _git_path_updated_at(path: Path, repo_root: Path) -> datetime | None:
    root = Path(repo_root)
    if not (root / ".git").exists():
        return None
    try:
        relative_path = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None

    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.resolve()}",
                "-C",
                str(root),
                "log",
                "-1",
                "--format=%cI",
                "--",
                relative_path,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None

    value = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _filesystem_path_updated_at(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
