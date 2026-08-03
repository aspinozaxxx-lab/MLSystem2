"""Единое разрешение записей списка сцен в подготовленные TIFF."""

from __future__ import annotations

from pathlib import Path

from ._prepare import _resolve_ambiguous_scenes
from ._scene_matching import (
    expand_scene_entries,
    filter_existing_scenes,
    index_image_files,
    read_scene_list,
)
from .contracts import (
    ResolvedSceneImage,
    SceneImageResolution,
    SceneImageResolutionRequest,
)


def resolve_scene_images(request: SceneImageResolutionRequest) -> SceneImageResolution:
    """Найти точные TIFF для строк списка сцен с optional разрешением по разметке."""

    images_root = Path(request.images_dir).resolve()
    scene_entries = read_scene_list(Path(request.scenes_file))
    image_index = index_image_files(images_root)
    annotation_files = [Path(value) for value in request.annotation_files]

    requests_by_path: dict[Path, list[str]] = {}
    missing_scenes: list[str] = []
    ambiguous_scenes: dict[str, list[str]] = {}
    for request_scene in scene_entries:
        expanded_scenes = expand_scene_entries([request_scene], image_index)
        filtered = filter_existing_scenes(expanded_scenes, image_index)
        resolved = dict(filtered.scene_to_image)
        if annotation_files:
            resolved.update(
                _resolve_ambiguous_scenes(filtered.ambiguous_scenes, annotation_files)
            )

        if filtered.missing_scenes:
            missing_scenes.append(request_scene)
        unresolved_paths = {
            path.resolve()
            for scene, paths in filtered.ambiguous_scenes.items()
            if scene not in resolved
            for path in paths
        }
        if unresolved_paths:
            ambiguous_scenes[request_scene] = [
                path.as_posix()
                for path in sorted(unresolved_paths, key=lambda item: item.as_posix().casefold())
            ]

        for scene in expanded_scenes:
            image_path = resolved.get(scene)
            if image_path is None:
                continue
            requests_by_path.setdefault(image_path.resolve(), []).append(request_scene)

    images = [
        ResolvedSceneImage(
            scene_id=image_path.stem,
            image_path=image_path.as_posix(),
            request_scenes=request_scenes,
        )
        for image_path, request_scenes in sorted(
            requests_by_path.items(),
            key=lambda item: item[0].relative_to(images_root).as_posix().casefold(),
        )
    ]
    return SceneImageResolution(
        input_scene_count=len(scene_entries),
        images=images,
        missing_scenes=missing_scenes,
        ambiguous_scenes=ambiguous_scenes,
    )


__all__ = ["resolve_scene_images"]
