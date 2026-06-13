"""Чтение публичного инвентаря MLMarkup для UI."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .contracts import ClassInfo, DatasetInfo, ImageFolderInfo


CUSTOM_KEY = "custom"
CUSTOM_NAME = "Custom"
DEFAULT_VARIANT = "main"
RASTER_SUFFIXES = (".tif", ".tiff")


def list_datasets(mlmarkup_root: Path, images_root: Path | None = None) -> list[DatasetInfo]:
    image_index = _image_index(images_root) if images_root is not None else None
    datasets = [
        variant
        for class_dir in _class_dirs(mlmarkup_root)
        for variant in _datasets_from_class_folder(class_dir, mlmarkup_root, image_index)
    ]
    datasets.sort(key=lambda item: item.name.lower())
    datasets.append(DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True))
    return datasets


def list_classes(mlmarkup_root: Path, images_root: Path | None = None) -> list[ClassInfo]:
    image_index = _image_index(images_root) if images_root is not None else None
    classes: list[ClassInfo] = []
    for path in _class_dirs(mlmarkup_root):
        variants = _datasets_from_class_folder(path, mlmarkup_root, image_index)
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


def list_image_folders(images_root: Path) -> list[ImageFolderInfo]:
    root = Path(images_root)
    if not root.exists() or not root.is_dir():
        return []
    folders: list[ImageFolderInfo] = []
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: item.as_posix().lower()):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        count = _direct_raster_count(path)
        if count <= 0:
            continue
        key = path.relative_to(root).as_posix()
        folders.append(
            ImageFolderInfo(
                key=key,
                name=key,
                path=str(path),
                image_count=count,
            )
        )
    return folders


def find_image_folder(images_root: Path, folder_key: str) -> ImageFolderInfo | None:
    for item in list_image_folders(images_root):
        if item.key == folder_key:
            return item
    return None


def find_class(mlmarkup_root: Path, class_key: str, images_root: Path | None = None) -> ClassInfo | None:
    for item in list_classes(mlmarkup_root, images_root):
        if item.key == class_key:
            return item
    return None


def find_dataset(mlmarkup_root: Path, dataset_key: str, images_root: Path | None = None) -> DatasetInfo | None:
    for item in list_datasets(mlmarkup_root, images_root):
        if item.key == dataset_key:
            return item
    return None


def _class_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]


def _datasets_from_class_folder(
    class_path: Path,
    repo_root: Path,
    image_index: dict[str, list[Path]] | None,
) -> list[DatasetInfo]:
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
            image_index=image_index,
        )
        for variant_path in variant_paths
    ]


def _dataset_from_variant_folder(
    *,
    class_path: Path,
    variant_path: Path,
    repo_root: Path,
    image_index: dict[str, list[Path]] | None,
) -> DatasetInfo:
    variant_name = variant_path.name if variant_path != class_path else DEFAULT_VARIANT
    dataset_name = _dataset_display_name(class_path.name, variant_name)
    scenes_file = _first_file(variant_path, ".txt")
    annotation_file = _first_file(variant_path, ".geojson")
    updated_at, version = _path_metadata(variant_path, repo_root)
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
        image_count=_dataset_image_count(scenes_file, image_index),
        version=version,
        updated_at=updated_at,
    )


def _dataset_display_name(class_name: str, variant_name: str) -> str:
    return f"{class_name}\\{variant_name}"


def _looks_like_dataset_folder(path: Path) -> bool:
    return _first_file(path, ".txt") is not None or _first_file(path, ".geojson") is not None


def _first_file(path: Path, suffix: str) -> Path | None:
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)
    return files[0] if files else None


def _direct_raster_count(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in RASTER_SUFFIXES)


def _dataset_image_count(scenes_file: Path | None, image_index: dict[str, list[Path]] | None) -> int | None:
    if scenes_file is None or image_index is None:
        return None
    try:
        scenes = [line.strip() for line in scenes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    found: set[Path] = set()
    for scene in scenes:
        found.update(_find_images(scene, image_index))
    return len(found)


def _image_index(images_root: Path) -> dict[str, list[Path]]:
    root = Path(images_root)
    if not root.exists() or not root.is_dir():
        return {}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
    ]
    index: dict[str, list[Path]] = {}
    for path in sorted(files):
        for key in _scene_lookup_keys(path.name):
            _add_index_path(index, key, path)
        for key in _scene_lookup_keys(path.stem):
            _add_index_path(index, key, path)
        for parent in path.parents:
            if parent == root:
                break
            for key in _scene_lookup_keys(parent.name):
                _add_index_path(index, key, path)
            try:
                relative_parent = parent.relative_to(root).as_posix()
            except ValueError:
                continue
            for key in _scene_lookup_keys(relative_parent):
                _add_index_path(index, key, path)
    return index


def _add_index_path(index: dict[str, list[Path]], key: str, path: Path) -> None:
    paths = index.setdefault(key, [])
    if path not in paths:
        paths.append(path)


def _find_images(scene: str, index: dict[str, list[Path]]) -> list[Path]:
    found: list[Path] = []
    for key in _scene_lookup_keys(scene):
        for path in index.get(key, []):
            if path not in found:
                found.append(path)
    return sorted(found)


def _scene_lookup_keys(value: str) -> set[str]:
    raw = value.strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.strip("/")
    if not raw:
        return set()

    path = PurePosixPath(raw)
    variants = {raw, path.name, path.stem, _strip_raster_suffix(raw), _strip_raster_suffix(path.name)}
    keys: set[str] = set()
    for variant in variants:
        if not variant:
            continue
        keys.add(variant.lower())
        if variant.endswith("_cog"):
            keys.add(variant[:-4].lower())
        else:
            keys.add(f"{variant}_cog".lower())
    return keys


def _strip_raster_suffix(value: str) -> str:
    lowered = value.lower()
    for suffix in (".tiff", ".tif"):
        if lowered.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _latest_updated_at(datasets: list[DatasetInfo]) -> datetime | None:
    values = [item.updated_at for item in datasets if item.updated_at is not None]
    return max(values) if values else None


def _path_metadata(path: Path, repo_root: Path) -> tuple[datetime | None, str | None]:
    return _git_path_metadata(path, repo_root) or _filesystem_path_metadata(path)


def _git_path_metadata(path: Path, repo_root: Path) -> tuple[datetime | None, str] | None:
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
                "--format=%H%x00%cI",
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
    parts = value.split("\x00", 1)
    if len(parts) != 2:
        return None
    commit_sha, committed_at = parts
    try:
        return datetime.fromisoformat(committed_at.replace("Z", "+00:00")), f"git:{commit_sha}"
    except ValueError:
        return None


def _filesystem_path_metadata(path: Path) -> tuple[datetime | None, str | None]:
    max_mtime_ns: int | None = None
    for item in Path(path).rglob("*"):
        if not item.is_file():
            continue
        try:
            mtime_ns = item.stat().st_mtime_ns
        except OSError:
            continue
        max_mtime_ns = mtime_ns if max_mtime_ns is None else max(max_mtime_ns, mtime_ns)
    if max_mtime_ns is None:
        try:
            max_mtime_ns = Path(path).stat().st_mtime_ns
        except OSError:
            return None, None
    return datetime.fromtimestamp(max_mtime_ns / 1_000_000_000, tz=timezone.utc), f"fs:{max_mtime_ns}"


def _filesystem_path_updated_at(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
