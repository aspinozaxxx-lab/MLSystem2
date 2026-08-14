"""Сопоставление per-image GeoJSON с подготовленными TIFF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._scene_matching import IMAGE_EXTENSIONS, index_image_files


PER_IMAGE_FOOTPRINT_SUFFIX = "_footprint.geojson"


@dataclass(frozen=True, slots=True)
class PerImageAnnotationMatch:
    scene_id: str
    image_path: Path
    annotation_file: Path
    footprint_file: Path | None = None


@dataclass(frozen=True, slots=True)
class PerImageAnnotationResolution:
    matches: tuple[PerImageAnnotationMatch, ...]
    missing_annotations: tuple[str, ...]
    ambiguous_annotations: dict[str, tuple[Path, ...]]
    annotation_collisions: dict[str, tuple[Path, ...]]


def per_image_annotation_name(image_path: str | Path) -> str:
    path = Path(image_path)
    return f"{path.parent.name}_{path.stem}.geojson"


def per_image_footprint_name(image_path: str | Path) -> str:
    return footprint_name_for_annotation(per_image_annotation_name(image_path))


def footprint_name_for_annotation(annotation_file: str | Path) -> str:
    name = Path(annotation_file).name
    if Path(name).suffix.casefold() != ".geojson" or is_per_image_footprint_name(name):
        raise ValueError(f"Некорректное имя per-image разметки: {name}")
    return f"{Path(name).stem}_footprint.geojson"


def is_per_image_footprint_name(value: str | Path) -> bool:
    return Path(value).name.casefold().endswith(PER_IMAGE_FOOTPRINT_SUFFIX)


def per_image_annotation_files(annotations_dir: str | Path) -> list[Path]:
    root = Path(annotations_dir)
    return sorted(
        (
            path.resolve()
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".geojson"
            and not is_per_image_footprint_name(path)
        ),
        key=lambda path: path.name.casefold(),
    )


def resolve_per_image_annotations(
    images_dir: str | Path,
    annotations_dir: str | Path,
) -> PerImageAnnotationResolution:
    images_root = Path(images_dir).resolve()
    annotation_root = Path(annotations_dir).resolve()
    if not annotation_root.exists():
        raise FileNotFoundError(f"Директория разметки не существует: {annotation_root}")
    if not annotation_root.is_dir():
        raise NotADirectoryError(f"Путь разметки не является директорией: {annotation_root}")

    image_index = index_image_files(images_root, IMAGE_EXTENSIONS)
    images_by_annotation: dict[str, list[Path]] = {}
    for image_path in image_index["paths"]:
        key = per_image_annotation_name(image_path).casefold()
        images_by_annotation.setdefault(key, []).append(Path(image_path).resolve())

    annotation_files = per_image_annotation_files(annotation_root)
    matches: list[PerImageAnnotationMatch] = []
    missing: list[str] = []
    ambiguous: dict[str, tuple[Path, ...]] = {}
    annotations_by_name: dict[str, list[Path]] = {}
    for annotation_file in annotation_files:
        annotations_by_name.setdefault(annotation_file.name.casefold(), []).append(
            annotation_file
        )
    collisions = {
        paths[0].name: tuple(paths)
        for paths in annotations_by_name.values()
        if len(paths) > 1
    }
    collided_paths = {
        path
        for paths in collisions.values()
        for path in paths
    }
    for annotation_file in annotation_files:
        if annotation_file in collided_paths:
            continue
        candidates = images_by_annotation.get(annotation_file.name.casefold(), [])
        if not candidates:
            missing.append(annotation_file.name)
            continue
        if len(candidates) != 1:
            ambiguous[annotation_file.name] = tuple(
                sorted(candidates, key=lambda path: path.as_posix().casefold())
            )
            continue
        image_path = candidates[0]
        scene_id = image_path.relative_to(images_root).with_suffix("").as_posix()
        matches.append(
            PerImageAnnotationMatch(
                scene_id=scene_id,
                image_path=image_path,
                annotation_file=annotation_file,
                footprint_file=(
                    annotation_root / footprint_name_for_annotation(annotation_file)
                    if (annotation_root / footprint_name_for_annotation(annotation_file)).is_file()
                    else None
                ),
            )
        )
    return PerImageAnnotationResolution(
        matches=tuple(matches),
        missing_annotations=tuple(missing),
        ambiguous_annotations=ambiguous,
        annotation_collisions=collisions,
    )


__all__ = [
    "PerImageAnnotationMatch",
    "PerImageAnnotationResolution",
    "PER_IMAGE_FOOTPRINT_SUFFIX",
    "footprint_name_for_annotation",
    "is_per_image_footprint_name",
    "per_image_annotation_name",
    "per_image_annotation_files",
    "per_image_footprint_name",
    "resolve_per_image_annotations",
]
