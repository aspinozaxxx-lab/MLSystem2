"""Единое разрешение записей списка сцен в подготовленные TIFF."""

from __future__ import annotations

from pathlib import Path

from ._per_image import resolve_per_image_annotations
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
    if request.annotations_dir is not None:
        return _resolve_per_image_scene_images(images_root, Path(request.annotations_dir))
    if request.scenes_file is None:
        raise AssertionError("scenes_file должен быть провалидирован")
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


def _resolve_per_image_scene_images(
    images_root: Path,
    annotations_dir: Path,
) -> SceneImageResolution:
    resolution = resolve_per_image_annotations(images_root, annotations_dir)
    return SceneImageResolution(
        input_scene_count=(
            len(resolution.matches)
            + len(resolution.missing_annotations)
            + len(resolution.ambiguous_annotations)
            + len(resolution.annotation_collisions)
        ),
        images=[
            ResolvedSceneImage(
                scene_id=item.scene_id,
                image_path=item.image_path.as_posix(),
                annotation_file=item.annotation_file.as_posix(),
                footprint_file=(
                    item.footprint_file.as_posix() if item.footprint_file is not None else None
                ),
                request_scenes=[item.annotation_file.name],
            )
            for item in resolution.matches
        ],
        missing_scenes=list(resolution.missing_annotations),
        ambiguous_scenes={
            name: [path.as_posix() for path in paths]
            for name, paths in {
                **resolution.ambiguous_annotations,
                **resolution.annotation_collisions,
            }.items()
        },
    )


__all__ = ["resolve_scene_images"]
