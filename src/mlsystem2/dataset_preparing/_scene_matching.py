"""Поиск снимков по списку сцен."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

IMAGE_EXTENSIONS = (".tif", ".tiff")


@dataclass(frozen=True)
class SceneFilterResult:
    existing_scenes: list[str]
    missing_scenes: list[str]
    scene_to_image: dict[str, Path]
    matched_by: dict[str, str]
    ambiguous_scenes: dict[str, list[Path]] = field(default_factory=dict)


def read_scene_list(txt_path: Path) -> list[str]:
    scenes: list[str] = []
    for raw_line in Path(txt_path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line.split()[0])
    return scenes


def index_image_files(
    images_dir: Path,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Директория снимков не существует: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Путь снимков не является директорией: {root}")

    normalized_ext = {item.lower() for item in extensions}
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_ext
    ]
    paths.sort(key=lambda item: str(item).casefold())

    by_name: dict[str, list[Path]] = defaultdict(list)
    by_name_casefold: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    by_stem_casefold: dict[str, list[Path]] = defaultdict(list)
    by_normalized: dict[str, list[Path]] = defaultdict(list)

    for path in paths:
        by_name[path.name].append(path)
        by_name_casefold[path.name.casefold()].append(path)
        by_stem[path.stem].append(path)
        by_stem_casefold[path.stem.casefold()].append(path)
        by_normalized[_normalized_scene_key(path.name)].append(path)

    return {
        "paths": paths,
        "by_name": dict(by_name),
        "by_name_casefold": dict(by_name_casefold),
        "by_stem": dict(by_stem),
        "by_stem_casefold": dict(by_stem_casefold),
        "by_normalized": dict(by_normalized),
    }


def filter_existing_scenes(scene_names: list[str], image_index: dict[str, Any]) -> SceneFilterResult:
    existing: list[str] = []
    missing: list[str] = []
    mapping: dict[str, Path] = {}
    matched_by: dict[str, str] = {}
    ambiguous: dict[str, list[Path]] = {}

    for scene in scene_names:
        name = _scene_basename(scene)
        stem = _scene_stem(scene)
        candidates: list[tuple[str, list[Path]]] = [
            ("filename_exact", image_index.get("by_name", {}).get(name, [])),
            ("stem_exact", image_index.get("by_stem", {}).get(stem, [])),
            ("filename_casefold", image_index.get("by_name_casefold", {}).get(name.casefold(), [])),
            ("stem_casefold", image_index.get("by_stem_casefold", {}).get(stem.casefold(), [])),
            ("normalized_scene", image_index.get("by_normalized", {}).get(_normalized_scene_key(scene), [])),
        ]

        ordered_matches: list[tuple[str, Path]] = []
        seen_paths: set[Path] = set()
        for reason, paths in candidates:
            for path in paths:
                if path in seen_paths:
                    continue
                ordered_matches.append((reason, path))
                seen_paths.add(path)

        if not ordered_matches:
            missing.append(scene)
            continue
        if len(ordered_matches) > 1:
            ambiguous[scene] = [path for _, path in ordered_matches]
            continue

        reason, path = ordered_matches[0]
        existing.append(scene)
        mapping[scene] = path
        matched_by[scene] = reason

    return SceneFilterResult(existing, missing, mapping, matched_by, ambiguous)


def scene_match_key(value: str) -> str:
    return _normalized_scene_key(value)


def scene_basename(value: str) -> str:
    return _scene_basename(value)


def scene_stem(value: str) -> str:
    return _scene_stem(value)


def _scene_basename(value: str) -> str:
    text = str(value).strip().strip('"').strip("'")
    text = text.replace("\\", "/")
    return PurePath(text).name


def _scene_stem(value: str) -> str:
    name = _scene_basename(value)
    suffix = Path(name).suffix
    return Path(name).stem if suffix.lower() in IMAGE_EXTENSIONS else name


def _normalized_scene_key(value: str) -> str:
    name = _scene_stem(value).casefold()
    name = re.sub(r"\.aux\.xml$", "", name)
    name = re.sub(r"[_\-. ]?cog$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)
