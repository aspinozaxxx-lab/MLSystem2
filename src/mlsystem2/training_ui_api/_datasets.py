"""Чтение публичного инвентаря MLMarkup для UI."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .contracts import ClassInfo, DatasetInfo, ImageFolderInfo


CUSTOM_KEY = "custom"
CUSTOM_NAME = "Custom"
DEFAULT_DATASET_NAME = "main"
RASTER_SUFFIXES = (".tif", ".tiff")
IMAGERY_FOLDERS = {"kanopus": "kanopus", "ortho": "orto"}
IMAGERY_CHANNELS = {"kanopus": 4, "ortho": 3}


def list_datasets(mlmarkup_root: Path, images_root: Path | None = None) -> list[DatasetInfo]:
    image_index = (
        _image_index(imagery_images_dir(images_root, "kanopus"))
        if images_root is not None
        else None
    )
    datasets = [
        dataset
        for class_dir in _class_dirs(mlmarkup_root)
        for dataset in _datasets_from_class_folder(class_dir, mlmarkup_root, image_index)
    ]
    datasets.sort(key=lambda item: item.name.lower())
    datasets.append(DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True))
    return datasets


def list_classes(mlmarkup_root: Path, images_root: Path | None = None) -> list[ClassInfo]:
    image_index = (
        _image_index(imagery_images_dir(images_root, "kanopus"))
        if images_root is not None
        else None
    )
    classes: list[ClassInfo] = []
    for path in _class_dirs(mlmarkup_root):
        datasets = _datasets_from_class_folder(path, mlmarkup_root, image_index)
        if not datasets:
            continue
        classes.append(
            ClassInfo(
                key=path.name,
                name=path.name,
                updated_at=_latest_updated_at(datasets),
                datasets=datasets,
                is_custom=False,
                imagery_type="kanopus",
                primary_dataset_key=next(
                    (item.key for item in datasets if item.dataset_name == DEFAULT_DATASET_NAME),
                    None,
                ),
            )
        )
    classes.sort(key=lambda item: item.name.lower())
    custom_dataset = DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True)
    classes.append(
        ClassInfo(
            key=CUSTOM_KEY,
            name=CUSTOM_NAME,
            datasets=[custom_dataset],
            is_custom=True,
        )
    )
    return classes


def list_image_folders(images_root: Path) -> list[ImageFolderInfo]:
    root = Path(images_root).resolve()
    if not root.exists() or not root.is_dir():
        return []
    folders: list[ImageFolderInfo] = []
    for imagery_type, folder_name in IMAGERY_FOLDERS.items():
        imagery_root = (root / folder_name).resolve()
        if not imagery_root.is_dir() or not _is_within_root(imagery_root, root):
            continue
        directories = [imagery_root, *(item for item in imagery_root.rglob("*") if item.is_dir())]
        for path in sorted(directories, key=lambda item: item.as_posix().casefold()):
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            count = _direct_raster_count(path)
            if count <= 0:
                continue
            key = relative.as_posix()
            folders.append(
                ImageFolderInfo(
                    key=key,
                    name=key,
                    path=str(path),
                    image_count=count,
                    imagery_type=imagery_type,
                )
            )
    folders.sort(key=lambda item: item.key.casefold())
    return folders


def find_image_folder(
    images_root: Path,
    folder_key: str,
    imagery_type: str | None = None,
) -> ImageFolderInfo | None:
    for item in list_image_folders(images_root):
        if item.key == folder_key and (
            imagery_type is None or item.imagery_type.value == imagery_type
        ):
            return item
    return None


def imagery_images_dir(images_root: Path, imagery_type: str) -> Path:
    try:
        folder = IMAGERY_FOLDERS[imagery_type]
    except KeyError as exc:
        raise ValueError(f"Неизвестный тип снимков: {imagery_type}") from exc
    return (Path(images_root) / folder).resolve()


def build_image_index(images_root: Path) -> dict[str, list[Path]]:
    return _image_index(images_root)


def resolve_scenes_file_images(scenes_file: Path, images_root: Path) -> list[Path]:
    index = _image_index(images_root)
    try:
        scenes = [
            line.strip()
            for line in Path(scenes_file).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except OSError:
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for scene in scenes:
        for path in _find_images(scene, index):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return found


def count_scenes_file_images(
    scenes_file: Path | None,
    images_root: Path,
    image_index: dict[str, list[Path]] | None = None,
) -> int | None:
    return _dataset_image_count(scenes_file, image_index if image_index is not None else _image_index(images_root))


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
    dataset_paths = [
        path
        for path in class_path.iterdir()
        if path.is_dir() and not path.name.startswith(".") and _looks_like_dataset_folder(path)
    ]
    if not dataset_paths and _looks_like_dataset_folder(class_path):
        dataset_paths = [class_path]
    dataset_paths.sort(
        key=lambda item: (item.name != DEFAULT_DATASET_NAME, item.name.casefold())
    )
    return [
        _dataset_from_folder(
            class_path=class_path,
            dataset_path=dataset_path,
            repo_root=repo_root,
            image_index=image_index,
        )
        for dataset_path in dataset_paths
    ]


def _dataset_from_folder(
    *,
    class_path: Path,
    dataset_path: Path,
    repo_root: Path,
    image_index: dict[str, list[Path]] | None,
) -> DatasetInfo:
    short_name = dataset_path.name if dataset_path != class_path else DEFAULT_DATASET_NAME
    display_name = _dataset_display_name(class_path.name, short_name)
    scenes_file = _first_file(dataset_path, ".txt")
    annotation_file, hard_negative_annotation_file, diagnostics = _annotation_files(dataset_path)
    updated_at, version = _path_metadata(dataset_path, repo_root)
    return DatasetInfo(
        key=display_name,
        name=display_name,
        dataset_name=short_name,
        class_key=class_path.name,
        class_name=class_path.name,
        path=str(dataset_path),
        scenes_file=str(scenes_file) if scenes_file else None,
        annotation_file=str(annotation_file) if annotation_file else None,
        hard_negative_annotation_file=(
            str(hard_negative_annotation_file) if hard_negative_annotation_file else None
        ),
        image_count=_dataset_image_count(scenes_file, image_index),
        version=version,
        updated_at=updated_at,
        imagery_type="kanopus",
        input_channels=IMAGERY_CHANNELS["kanopus"],
        diagnostics=diagnostics,
    )


def _dataset_display_name(class_name: str, dataset_name: str) -> str:
    return f"{class_name}\\{dataset_name}"


def _looks_like_dataset_folder(path: Path) -> bool:
    return _first_file(path, ".txt") is not None or _first_file(path, ".geojson") is not None


def _first_file(path: Path, suffix: str) -> Path | None:
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)
    return files[0] if files else None


def _annotation_files(path: Path) -> tuple[Path | None, Path | None, list[str]]:
    geojson_files = sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".geojson"),
        key=lambda item: item.name.casefold(),
    )
    hard_negative_files = [
        item for item in geojson_files if item.name.casefold() == "hard_negative.geojson"
    ]
    positive_files = [item for item in geojson_files if item not in hard_negative_files]
    diagnostics: list[str] = []
    annotation_file: Path | None = None
    hard_negative_annotation_file: Path | None = None

    if len(positive_files) == 1:
        annotation_file = positive_files[0]
    elif len(positive_files) > 1:
        diagnostics.append(
            "В папке датасета найдено несколько positive GeoJSON, выбор разметки неоднозначен."
        )

    if len(hard_negative_files) == 1:
        hard_negative_annotation_file = hard_negative_files[0]
    elif len(hard_negative_files) > 1:
        diagnostics.append(
            "В папке датасета найдено несколько hard_negative.geojson, выбор разметки неоднозначен."
        )

    return annotation_file, hard_negative_annotation_file, diagnostics


def _direct_raster_count(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in RASTER_SUFFIXES)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.name
        for key in _scene_lookup_keys(relative_path):
            _add_index_path(index, key, path)
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
    normalized = _normalized_scene(scene)
    if "/" in normalized:
        exact = _paths_for_keys(_exact_scene_lookup_keys(normalized), index)
        if exact:
            return exact
    return _paths_for_keys(_scene_lookup_keys(normalized), index)


def _paths_for_keys(
    keys: set[str],
    index: dict[str, list[Path]],
) -> list[Path]:
    found: list[Path] = []
    for key in keys:
        for path in index.get(key, []):
            if path not in found:
                found.append(path)
    return sorted(found)


def _scene_lookup_keys(value: str) -> set[str]:
    raw = _normalized_scene(value)
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


def _exact_scene_lookup_keys(value: str) -> set[str]:
    raw = _normalized_scene(value)
    if not raw:
        return set()
    variants = {raw, _strip_raster_suffix(raw)}
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


def _normalized_scene(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.strip("/")


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
