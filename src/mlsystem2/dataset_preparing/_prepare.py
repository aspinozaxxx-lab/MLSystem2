"""Реализация подготовки датасета."""

from __future__ import annotations

from pathlib import Path

from ._manifest import (
    load_dataset_manifest,
    manifest_path,
    validate_multiclass_annotation,
)
from ._object_counts import (
    ImageGeometryScore,
    SceneObjectCount,
    count_per_image_annotation_roles,
    count_objects_per_scene,
    score_images_by_annotation_geometry,
)
from ._per_image import resolve_per_image_annotations
from ._raster_validation import RasterValidationResult, validate_rasters
from ._scene_matching import (
    expand_scene_entries,
    filter_existing_scenes,
    index_image_files,
    read_scene_list,
)
from .contracts import (
    DatasetClassAnnotation,
    DatasetClassRequest,
    DatasetPreparationReport,
    DatasetPreparationRequest,
    DatasetPreparationResult,
    DatasetSceneReport,
    PreparedDataset,
    PreparedScene,
)

def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    if request.classes:
        return _prepare_multiclass_dataset(request)
    if request.annotations_dir is not None:
        return _prepare_per_image_dataset(request)
    return _prepare_binary_dataset(request)


def _prepare_binary_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    images_dir = Path(request.images_dir)
    if request.scenes_file is None or request.annotation_file is None:
        raise AssertionError("binary request должен быть провалидирован до подготовки")
    scenes_file = Path(request.scenes_file)
    annotation_file = Path(request.annotation_file)
    hard_negative_file = (
        Path(request.hard_negative_annotation_file)
        if request.hard_negative_annotation_file is not None
        else None
    )

    errors: list[str] = []
    scenes = _read_scenes_or_collect_error(scenes_file, errors)
    if not scenes:
        errors.append("Список сцен пуст.")
    if not images_dir.exists():
        errors.append(f"Директория снимков не существует: {images_dir}")
    elif not images_dir.is_dir():
        errors.append(f"Путь снимков не является директорией: {images_dir}")
    if not annotation_file.exists():
        errors.append(f"Файл разметки не существует: {annotation_file}")
    elif not annotation_file.is_file():
        errors.append(f"Путь разметки не является файлом: {annotation_file}")
    if hard_negative_file is not None:
        if not hard_negative_file.exists():
            errors.append(f"Файл hard negative разметки не существует: {hard_negative_file}")
        elif not hard_negative_file.is_file():
            errors.append(f"Путь hard negative разметки не является файлом: {hard_negative_file}")

    if errors:
        report = _build_report(
            scenes=scenes,
            positive_rows=[],
            hard_negative_rows=[],
            scene_to_image={},
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    image_index = _index_images_or_collect_error(images_dir, errors)
    if image_index is None:
        report = _build_report(
            scenes=scenes,
            positive_rows=[],
            hard_negative_rows=[],
            scene_to_image={},
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    scenes = expand_scene_entries(scenes, image_index)
    filtered = filter_existing_scenes(scenes, image_index)
    missing_files = list(filtered.missing_scenes)
    scene_to_image = {
        scene: path.resolve()
        for scene, path in filtered.scene_to_image.items()
    }
    resolved_ambiguous = _resolve_ambiguous_scenes(
        filtered.ambiguous_scenes,
        _annotation_files(annotation_file, hard_negative_file),
    )
    scene_to_image.update({
        scene: path.resolve()
        for scene, path in resolved_ambiguous.items()
    })
    if missing_files:
        errors.append(f"Не найдены снимки для сцен: {', '.join(missing_files)}")
    for scene, paths in filtered.ambiguous_scenes.items():
        if scene in resolved_ambiguous:
            continue
        joined = "; ".join(path.resolve().as_posix() for path in paths)
        errors.append(f"Сцена неоднозначно сопоставлена со снимками: {scene}: {joined}")

    positive_rows = _count_objects_or_collect_error(scenes, scene_to_image, annotation_file, errors)
    hard_negative_rows = _count_optional_objects_or_collect_error(
        scenes,
        scene_to_image,
        hard_negative_file,
        errors,
    )
    found_rows = [row for row in positive_rows if row.scene_name in scene_to_image]
    pool_scene_ids = [row.scene_name for row in found_rows]

    if not found_rows:
        errors.append("Не найдено ни одного снимка из списка сцен.")

    selected_scene_to_image = {
        scene: scene_to_image[scene]
        for scene in pool_scene_ids
        if scene in scene_to_image
    }
    validation = (
        validate_rasters(
            selected_scene_to_image,
            expected_band_count=request.expected_band_count,
            expected_dtype=request.expected_dtype,
        )
        if selected_scene_to_image
        else None
    )
    if validation is not None:
        errors.extend(validation.errors)

    dataset: PreparedDataset | None = None
    if not errors and validation is not None:
        raster_by_scene = {raster.scene_id: raster for raster in validation.rasters}
        dataset = PreparedDataset(
            format="legacy_binary",
            scenes=[
                PreparedScene(
                    scene_id=scene,
                    image_path=raster_by_scene[scene].path.resolve().as_posix(),
                )
                for scene in pool_scene_ids
            ],
            annotation_file=annotation_file.resolve().as_posix(),
            hard_negative_annotation_file=(
                hard_negative_file.resolve().as_posix()
                if hard_negative_file is not None
                else None
            ),
        )

    report = _build_report(
        scenes=scenes,
        positive_rows=positive_rows,
        hard_negative_rows=hard_negative_rows,
        scene_to_image=scene_to_image,
        missing_files=missing_files,
        errors=errors,
        validation=validation,
    )
    if errors:
        dataset = None
    return DatasetPreparationResult(dataset=dataset, report=report)


def _prepare_per_image_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    images_dir = Path(request.images_dir)
    if request.annotations_dir is None:
        raise AssertionError("annotations_dir должен быть провалидирован до подготовки")
    annotations_dir = Path(request.annotations_dir)
    errors: list[str] = []
    manifest = None
    try:
        manifest = load_dataset_manifest(annotations_dir)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Не удалось загрузить схему per-image датасета: {exc}")
    try:
        resolution = resolve_per_image_annotations(images_dir, annotations_dir)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Не удалось сопоставить per-image разметку: {exc}")
        report = _build_report(
            scenes=[],
            positive_rows=[],
            hard_negative_rows=[],
            scene_to_image={},
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    scenes = [
        *(item.scene_id for item in resolution.matches),
        *resolution.missing_annotations,
        *resolution.ambiguous_annotations,
        *resolution.annotation_collisions,
    ]
    missing_files = list(resolution.missing_annotations)
    if missing_files:
        errors.append(
            "Не найдены снимки для файлов разметки: " + ", ".join(missing_files)
        )
    for annotation_name, paths in resolution.ambiguous_annotations.items():
        joined = "; ".join(path.as_posix() for path in paths)
        errors.append(
            f"Файл разметки неоднозначно сопоставлен со снимками: "
            f"{annotation_name}: {joined}"
        )
    for annotation_name, paths in resolution.annotation_collisions.items():
        joined = "; ".join(path.as_posix() for path in paths)
        errors.append(
            f"Коллизия имён файлов per-image разметки: "
            f"{annotation_name}: {joined}"
        )
    if not resolution.matches:
        errors.append("В per-image датасете не найдено ни одной сопоставленной сцены.")
    scene_to_image = {item.scene_id: item.image_path for item in resolution.matches}
    positive_rows: list[SceneObjectCount] = []
    hard_negative_rows: list[SceneObjectCount] = []
    class_counts_by_scene: dict[str, dict[str, int]] = {}
    for item in resolution.matches:
        try:
            if manifest is not None:
                multiclass_counts = validate_multiclass_annotation(
                    item.annotation_file,
                    manifest,
                )
                class_counts_by_scene[item.scene_id] = multiclass_counts.by_class
            positive_count, hard_negative_count = count_per_image_annotation_roles(
                item.annotation_file,
                item.image_path,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"Не удалось прочитать per-image разметку {item.annotation_file}: {exc}"
            )
            positive_count = 0
            hard_negative_count = 0
        positive_rows.append(
            SceneObjectCount(item.scene_id, item.image_path, positive_count)
        )
        hard_negative_rows.append(
            SceneObjectCount(item.scene_id, item.image_path, hard_negative_count)
        )

    validation = (
        validate_rasters(
            scene_to_image,
            expected_band_count=request.expected_band_count,
            expected_dtype=request.expected_dtype,
        )
        if scene_to_image
        else None
    )
    if validation is not None:
        errors.extend(validation.errors)

    dataset: PreparedDataset | None = None
    if not errors and validation is not None:
        annotation_by_scene = {
            item.scene_id: item.annotation_file for item in resolution.matches
        }
        dataset = PreparedDataset(
            format=("per_image_multiclass" if manifest is not None else "per_image_binary"),
            scenes=[
                PreparedScene(
                    scene_id=raster.scene_id,
                    image_path=raster.path.resolve().as_posix(),
                    annotation_file=annotation_by_scene[
                        raster.scene_id
                    ].resolve().as_posix(),
                )
                for raster in validation.rasters
            ],
            classes=(list(manifest.classes) if manifest is not None else []),
            manifest_file=(
                manifest_path(annotations_dir).resolve().as_posix()
                if manifest is not None
                else None
            ),
        )

    report = _build_report(
        scenes=scenes,
        positive_rows=positive_rows,
        hard_negative_rows=hard_negative_rows,
        scene_to_image=scene_to_image,
        missing_files=missing_files,
        errors=errors,
        validation=validation,
        class_counts_by_scene=class_counts_by_scene,
    )
    return DatasetPreparationResult(dataset=dataset, report=report)


def _prepare_multiclass_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    images_dir = Path(request.images_dir)
    classes = list(request.classes or [])
    errors: list[str] = []
    scenes_by_class: dict[str, list[str]] = {}
    annotation_by_slug: dict[str, Path] = {}
    hard_negative_by_slug: dict[str, Path] = {}

    for class_request in classes:
        scenes_file = Path(class_request.scenes_file)
        annotation_file = Path(class_request.annotation_file)
        hard_negative_file = (
            Path(class_request.hard_negative_annotation_file)
            if class_request.hard_negative_annotation_file is not None
            else None
        )
        scenes = _read_scenes_or_collect_error(scenes_file, errors)
        scenes_by_class[class_request.slug] = scenes
        annotation_by_slug[class_request.slug] = annotation_file
        if not annotation_file.exists():
            errors.append(
                f"Файл разметки класса {class_request.slug} не существует: {annotation_file}"
            )
        elif not annotation_file.is_file():
            errors.append(
                f"Путь разметки класса {class_request.slug} не является файлом: {annotation_file}"
            )
        if hard_negative_file is not None:
            hard_negative_by_slug[class_request.slug] = hard_negative_file
            if not hard_negative_file.exists():
                errors.append(
                    f"Файл hard negative разметки класса {class_request.slug} не существует: "
                    f"{hard_negative_file}"
                )
            elif not hard_negative_file.is_file():
                errors.append(
                    f"Путь hard negative разметки класса {class_request.slug} не является файлом: "
                    f"{hard_negative_file}"
                )

    scenes = _unique_preserving_order(
        scene
        for class_scenes in scenes_by_class.values()
        for scene in class_scenes
    )
    if not scenes:
        errors.append("Список сцен пуст.")
    if not images_dir.exists():
        errors.append(f"Директория снимков не существует: {images_dir}")
    elif not images_dir.is_dir():
        errors.append(f"Путь снимков не является директорией: {images_dir}")

    if errors:
        report = _build_report(
            scenes=scenes,
            positive_rows=[],
            hard_negative_rows=[],
            scene_to_image={},
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    image_index = _index_images_or_collect_error(images_dir, errors)
    if image_index is None:
        report = _build_report(
            scenes=scenes,
            positive_rows=[],
            hard_negative_rows=[],
            scene_to_image={},
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    scenes_by_class = {
        slug: expand_scene_entries(class_scenes, image_index)
        for slug, class_scenes in scenes_by_class.items()
    }
    scenes = _unique_preserving_order(
        scene
        for class_scenes in scenes_by_class.values()
        for scene in class_scenes
    )
    filtered = filter_existing_scenes(scenes, image_index)
    missing_files = list(filtered.missing_scenes)
    scene_to_image = {
        scene: path.resolve()
        for scene, path in filtered.scene_to_image.items()
    }
    resolved_ambiguous = _resolve_ambiguous_scenes(
        filtered.ambiguous_scenes,
        [*annotation_by_slug.values(), *hard_negative_by_slug.values()],
    )
    scene_to_image.update({
        scene: path.resolve()
        for scene, path in resolved_ambiguous.items()
    })
    if missing_files:
        errors.append(f"Не найдены снимки для сцен: {', '.join(missing_files)}")
    for scene, paths in filtered.ambiguous_scenes.items():
        if scene in resolved_ambiguous:
            continue
        joined = "; ".join(path.resolve().as_posix() for path in paths)
        errors.append(f"Сцена неоднозначно сопоставлена со снимками: {scene}: {joined}")

    positive_rows = _count_multiclass_objects_or_collect_errors(
        classes,
        scenes,
        scene_to_image,
        annotation_by_slug,
        errors,
    )
    hard_negative_rows = _count_multiclass_optional_objects_or_collect_errors(
        classes,
        scenes,
        scene_to_image,
        hard_negative_by_slug,
        errors,
    )
    found_rows = [row for row in positive_rows if row.scene_name in scene_to_image]
    pool_scene_ids = [row.scene_name for row in found_rows]

    if not found_rows:
        errors.append("Не найдено ни одного снимка из списка сцен.")

    selected_scene_to_image = {
        scene: scene_to_image[scene]
        for scene in pool_scene_ids
        if scene in scene_to_image
    }
    validation = (
        validate_rasters(
            selected_scene_to_image,
            expected_band_count=request.expected_band_count,
            expected_dtype=request.expected_dtype,
        )
        if selected_scene_to_image
        else None
    )
    if validation is not None:
        errors.extend(validation.errors)

    dataset: PreparedDataset | None = None
    if not errors and validation is not None:
        raster_by_scene = {raster.scene_id: raster for raster in validation.rasters}
        dataset = PreparedDataset(
            format="legacy_multiclass",
            scenes=[
                PreparedScene(
                    scene_id=scene,
                    image_path=raster_by_scene[scene].path.resolve().as_posix(),
                )
                for scene in pool_scene_ids
            ],
            annotation_file=None,
            class_annotations=[
                DatasetClassAnnotation(
                    class_id=class_id,
                    slug=class_request.slug,
                    name=class_request.name,
                    annotation_file=Path(class_request.annotation_file).resolve().as_posix(),
                    hard_negative_annotation_file=(
                        hard_negative_by_slug[class_request.slug].resolve().as_posix()
                        if class_request.slug in hard_negative_by_slug
                        else None
                    ),
                    priority=class_request.priority,
                )
                for class_id, class_request in enumerate(classes, start=1)
            ],
        )

    report = _build_report(
        scenes=scenes,
        positive_rows=positive_rows,
        hard_negative_rows=hard_negative_rows,
        scene_to_image=scene_to_image,
        missing_files=missing_files,
        errors=errors,
        validation=validation,
    )
    if errors:
        dataset = None
    return DatasetPreparationResult(dataset=dataset, report=report)


def _read_scenes_or_collect_error(path: Path, errors: list[str]) -> list[str]:
    if not path.exists():
        errors.append(f"Файл списка сцен не существует: {path}")
        return []
    if not path.is_file():
        errors.append(f"Путь списка сцен не является файлом: {path}")
        return []
    try:
        return read_scene_list(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Не удалось прочитать список сцен: {path}: {exc}")
        return []


def _index_images_or_collect_error(images_dir: Path, errors: list[str]) -> dict[str, object] | None:
    try:
        return index_image_files(images_dir)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Не удалось проиндексировать снимки: {images_dir}: {exc}")
        return None


def _count_objects_or_collect_error(
    scenes: list[str],
    scene_to_image: dict[str, Path],
    annotation_file: Path,
    errors: list[str],
) -> list[SceneObjectCount]:
    try:
        return count_objects_per_scene(scenes, scene_to_image, annotation_file)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Не удалось посчитать объекты по разметке: {annotation_file}: {exc}")
        return [
            SceneObjectCount(scene_name=scene, image_path=scene_to_image.get(scene), object_count=0)
            for scene in scenes
        ]


def _count_optional_objects_or_collect_error(
    scenes: list[str],
    scene_to_image: dict[str, Path],
    annotation_file: Path | None,
    errors: list[str],
) -> list[SceneObjectCount]:
    if annotation_file is None:
        return _zero_object_rows(scenes, scene_to_image)
    return _count_objects_or_collect_error(scenes, scene_to_image, annotation_file, errors)


def _count_multiclass_objects_or_collect_errors(
    classes: list[DatasetClassRequest],
    scenes: list[str],
    scene_to_image: dict[str, Path],
    annotation_by_slug: dict[str, Path],
    errors: list[str],
) -> list[SceneObjectCount]:
    counts_by_scene = {
        scene: SceneObjectCount(
            scene_name=scene,
            image_path=scene_to_image.get(scene),
            object_count=0,
        )
        for scene in scenes
    }
    for class_request in classes:
        class_rows = _count_objects_or_collect_error(
            scenes,
            scene_to_image,
            annotation_by_slug[class_request.slug],
            errors,
        )
        for row in class_rows:
            existing = counts_by_scene[row.scene_name]
            counts_by_scene[row.scene_name] = SceneObjectCount(
                scene_name=row.scene_name,
                image_path=existing.image_path or row.image_path,
                object_count=existing.object_count + row.object_count,
            )
    return [counts_by_scene[scene] for scene in scenes]


def _count_multiclass_optional_objects_or_collect_errors(
    classes: list[DatasetClassRequest],
    scenes: list[str],
    scene_to_image: dict[str, Path],
    annotation_by_slug: dict[str, Path],
    errors: list[str],
) -> list[SceneObjectCount]:
    counts_by_scene = {
        scene: SceneObjectCount(
            scene_name=scene,
            image_path=scene_to_image.get(scene),
            object_count=0,
        )
        for scene in scenes
    }
    for class_request in classes:
        annotation_file = annotation_by_slug.get(class_request.slug)
        if annotation_file is None:
            continue
        class_rows = _count_objects_or_collect_error(
            scenes,
            scene_to_image,
            annotation_file,
            errors,
        )
        for row in class_rows:
            existing = counts_by_scene[row.scene_name]
            counts_by_scene[row.scene_name] = SceneObjectCount(
                scene_name=row.scene_name,
                image_path=existing.image_path or row.image_path,
                object_count=existing.object_count + row.object_count,
            )
    return [counts_by_scene[scene] for scene in scenes]


def _zero_object_rows(
    scenes: list[str],
    scene_to_image: dict[str, Path],
) -> list[SceneObjectCount]:
    return [
        SceneObjectCount(scene_name=scene, image_path=scene_to_image.get(scene), object_count=0)
        for scene in scenes
    ]


def _resolve_ambiguous_scenes(
    ambiguous_scenes: dict[str, list[Path]],
    annotation_files: list[Path],
) -> dict[str, Path]:
    if not ambiguous_scenes:
        return {}

    candidate_paths = _unique_paths(
        path
        for paths in ambiguous_scenes.values()
        for path in paths
    )
    scores_by_path: dict[Path, ImageGeometryScore] = {}
    for annotation_file in annotation_files:
        annotation_scores = score_images_by_annotation_geometry(candidate_paths, annotation_file)
        for path, score in annotation_scores.items():
            existing = scores_by_path.get(path)
            if existing is None:
                scores_by_path[path] = score
                continue
            scores_by_path[path] = ImageGeometryScore(
                image_path=path,
                object_count=existing.object_count + score.object_count,
                distance_to_annotation=min(
                    existing.distance_to_annotation,
                    score.distance_to_annotation,
                ),
            )

    resolved: dict[str, Path] = {}
    for scene, paths in ambiguous_scenes.items():
        scored_paths = [
            scores_by_path[path]
            for path in paths
            if path in scores_by_path
        ]
        if not scored_paths:
            continue
        best = sorted(
            scored_paths,
            key=lambda item: (
                -item.object_count,
                item.distance_to_annotation,
                item.image_path.as_posix().casefold(),
            ),
        )[0]
        resolved[scene] = best.image_path
    return resolved


def _unique_paths(values) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _unique_preserving_order(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_report(
    *,
    scenes: list[str],
    positive_rows: list[SceneObjectCount],
    hard_negative_rows: list[SceneObjectCount],
    scene_to_image: dict[str, Path],
    missing_files: list[str],
    errors: list[str],
    validation: RasterValidationResult | None = None,
    class_counts_by_scene: dict[str, dict[str, int]] | None = None,
) -> DatasetPreparationReport:
    positive_by_scene = {row.scene_name: row.object_count for row in positive_rows}
    hard_negative_by_scene = {row.scene_name: row.object_count for row in hard_negative_rows}
    resolved_class_counts = class_counts_by_scene or {}
    scene_reports = [
        DatasetSceneReport(
            scene_id=scene,
            image_path=scene_to_image[scene].as_posix() if scene in scene_to_image else None,
            positive_objects=max(0, int(positive_by_scene.get(scene, 0))),
            hard_negative_objects=max(0, int(hard_negative_by_scene.get(scene, 0))),
            object_count=max(0, int(positive_by_scene.get(scene, 0)))
            + max(0, int(hard_negative_by_scene.get(scene, 0))),
            class_counts={
                slug: max(0, int(count))
                for slug, count in resolved_class_counts.get(scene, {}).items()
            },
        )
        for scene in scenes
    ]
    positive_objects = sum(item.positive_objects for item in scene_reports)
    hard_negative_objects = sum(item.hard_negative_objects for item in scene_reports)
    class_counts: dict[str, int] = {}
    for item in scene_reports:
        for slug, count in item.class_counts.items():
            class_counts[slug] = class_counts.get(slug, 0) + count
    first_raster = validation.rasters[0] if validation and validation.rasters else None
    return DatasetPreparationReport(
        status="error" if errors else "ok",
        scenes_total=len(scenes),
        scenes_found=len(scene_to_image),
        positive_objects=positive_objects,
        hard_negative_objects=hard_negative_objects,
        objects_total=positive_objects + hard_negative_objects,
        class_counts=class_counts,
        band_count=first_raster.band_count if first_raster is not None else None,
        dtypes=list(first_raster.dtypes) if first_raster is not None else [],
        scenes=scene_reports,
        missing_files=missing_files,
        errors=errors,
    )


def _annotation_files(
    annotation_file: Path,
    hard_negative_file: Path | None,
) -> list[Path]:
    if hard_negative_file is None:
        return [annotation_file]
    return [annotation_file, hard_negative_file]
