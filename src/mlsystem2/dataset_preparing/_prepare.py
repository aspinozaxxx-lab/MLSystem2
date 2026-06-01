"""Реализация подготовки датасета."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ._object_counts import SceneObjectCount, count_objects_per_scene
from ._raster_validation import validate_rasters
from ._scene_matching import filter_existing_scenes, index_image_files, read_scene_list
from ._split import split_train_val_by_object_counts
from ._vrt import build_vrt_xml
from .contracts import (
    DatasetClassAnnotation,
    DatasetClassRequest,
    DatasetPreparationReport,
    DatasetPreparationRequest,
    DatasetPreparationResult,
    DatasetSceneReport,
    PreparedDataset,
)

SPLIT_SEED = 42


@dataclass(frozen=True)
class _SceneSelection:
    rows: list[SceneObjectCount]
    positive_count: int
    negative_count: int


def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    if request.classes:
        return _prepare_multiclass_dataset(request)
    return _prepare_binary_dataset(request)


def _prepare_binary_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    images_dir = Path(request.images_dir)
    if request.scenes_file is None or request.annotation_file is None:
        raise AssertionError("binary request должен быть провалидирован до подготовки")
    scenes_file = Path(request.scenes_file)
    annotation_file = Path(request.annotation_file)

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

    if errors:
        report = _build_report(
            scenes=scenes,
            rows=[],
            scene_to_image={},
            train_names=set(),
            val_names=set(),
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    image_index = _index_images_or_collect_error(images_dir, errors)
    if image_index is None:
        report = _build_report(
            scenes=scenes,
            rows=[],
            scene_to_image={},
            train_names=set(),
            val_names=set(),
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    filtered = filter_existing_scenes(scenes, image_index)
    missing_files = list(filtered.missing_scenes)
    scene_to_image = {
        scene: path.resolve()
        for scene, path in filtered.scene_to_image.items()
    }
    if missing_files:
        errors.append(f"Не найдены снимки для сцен: {', '.join(missing_files)}")
    for scene, paths in filtered.ambiguous_scenes.items():
        joined = "; ".join(path.resolve().as_posix() for path in paths)
        errors.append(f"Сцена неоднозначно сопоставлена со снимками: {scene}: {joined}")

    rows = _count_objects_or_collect_error(scenes, scene_to_image, annotation_file, errors)
    found_rows = [row for row in rows if row.scene_name in scene_to_image]
    selection = _select_scene_rows(
        found_rows,
        negative_scene_limit=request.negative_scene_limit,
        seed=SPLIT_SEED,
    )
    split = (
        split_train_val_by_object_counts(
            selection.rows,
            target_val_fraction=request.val_fraction,
            seed=SPLIT_SEED,
        )
        if request.split_granularity == "scene"
        else None
    )
    train_scene_ids = [row.scene_name for row in split.train] if split is not None else []
    val_scene_ids = [row.scene_name for row in split.val] if split is not None else []
    train_names = set(train_scene_ids)
    val_names = set(val_scene_ids)
    pool_scene_ids = [row.scene_name for row in selection.rows]
    pool_names = set(pool_scene_ids) if request.split_granularity == "tile" else set()

    if not found_rows:
        errors.append("Не найдено ни одного снимка из списка сцен.")
    elif not selection.rows:
        errors.append("После отбора сцен датасет пуст.")
    elif request.split_granularity == "scene" and split is not None and (
        not split.train or not split.val
    ):
        errors.append("Недостаточно найденных сцен для построения train и val VRT.")

    selected_scene_to_image = {
        scene: scene_to_image[scene]
        for scene in pool_scene_ids
        if scene in scene_to_image
    }
    validation = validate_rasters(selected_scene_to_image) if selected_scene_to_image else None
    if validation is not None:
        errors.extend(validation.errors)

    dataset: PreparedDataset | None = None
    if not errors and validation is not None:
        raster_by_scene = {raster.scene_id: raster for raster in validation.rasters}
        try:
            if request.split_granularity == "tile":
                pool_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in pool_scene_ids])
                train_vrt_xml = pool_vrt_xml
                val_vrt_xml = pool_vrt_xml
            else:
                pool_vrt_xml = None
                train_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in train_scene_ids])
                val_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in val_scene_ids])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Не удалось построить VRT: {exc}")
        else:
            dataset = PreparedDataset(
                train_vrt_xml=train_vrt_xml,
                val_vrt_xml=val_vrt_xml,
                pool_vrt_xml=pool_vrt_xml,
                annotation_file=annotation_file.resolve().as_posix(),
            )

    report = _build_report(
        scenes=scenes,
        rows=rows,
        scene_to_image=scene_to_image,
        train_names=train_names,
        val_names=val_names,
        pool_names=pool_names,
        missing_files=missing_files,
        errors=errors,
        split_granularity=request.split_granularity,
        negative_scene_limit=request.negative_scene_limit,
        selected_positive_scenes_count=selection.positive_count,
        selected_negative_scenes_count=selection.negative_count,
    )
    if errors:
        dataset = None
    return DatasetPreparationResult(dataset=dataset, report=report)


def _prepare_multiclass_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    images_dir = Path(request.images_dir)
    classes = list(request.classes or [])
    errors: list[str] = []
    scenes_by_class: dict[str, list[str]] = {}
    annotation_by_slug: dict[str, Path] = {}

    for class_request in classes:
        scenes_file = Path(class_request.scenes_file)
        annotation_file = Path(class_request.annotation_file)
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
            rows=[],
            scene_to_image={},
            train_names=set(),
            val_names=set(),
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    image_index = _index_images_or_collect_error(images_dir, errors)
    if image_index is None:
        report = _build_report(
            scenes=scenes,
            rows=[],
            scene_to_image={},
            train_names=set(),
            val_names=set(),
            missing_files=[],
            errors=errors,
        )
        return DatasetPreparationResult(dataset=None, report=report)

    filtered = filter_existing_scenes(scenes, image_index)
    missing_files = list(filtered.missing_scenes)
    scene_to_image = {
        scene: path.resolve()
        for scene, path in filtered.scene_to_image.items()
    }
    if missing_files:
        errors.append(f"Не найдены снимки для сцен: {', '.join(missing_files)}")
    for scene, paths in filtered.ambiguous_scenes.items():
        joined = "; ".join(path.resolve().as_posix() for path in paths)
        errors.append(f"Сцена неоднозначно сопоставлена со снимками: {scene}: {joined}")

    rows = _count_multiclass_objects_or_collect_errors(
        classes,
        scenes,
        scene_to_image,
        annotation_by_slug,
        errors,
    )
    found_rows = [row for row in rows if row.scene_name in scene_to_image]
    selection = _select_scene_rows(
        found_rows,
        negative_scene_limit=request.negative_scene_limit,
        seed=SPLIT_SEED,
    )
    split = (
        split_train_val_by_object_counts(
            selection.rows,
            target_val_fraction=request.val_fraction,
            seed=SPLIT_SEED,
        )
        if request.split_granularity == "scene"
        else None
    )
    train_scene_ids = [row.scene_name for row in split.train] if split is not None else []
    val_scene_ids = [row.scene_name for row in split.val] if split is not None else []
    train_names = set(train_scene_ids)
    val_names = set(val_scene_ids)
    pool_scene_ids = [row.scene_name for row in selection.rows]
    pool_names = set(pool_scene_ids) if request.split_granularity == "tile" else set()

    if not found_rows:
        errors.append("Не найдено ни одного снимка из списка сцен.")
    elif not selection.rows:
        errors.append("После отбора сцен датасет пуст.")
    elif request.split_granularity == "scene" and split is not None and (
        not split.train or not split.val
    ):
        errors.append("Недостаточно найденных сцен для построения train и val VRT.")

    selected_scene_to_image = {
        scene: scene_to_image[scene]
        for scene in pool_scene_ids
        if scene in scene_to_image
    }
    validation = validate_rasters(selected_scene_to_image) if selected_scene_to_image else None
    if validation is not None:
        errors.extend(validation.errors)

    dataset: PreparedDataset | None = None
    if not errors and validation is not None:
        raster_by_scene = {raster.scene_id: raster for raster in validation.rasters}
        try:
            if request.split_granularity == "tile":
                pool_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in pool_scene_ids])
                train_vrt_xml = pool_vrt_xml
                val_vrt_xml = pool_vrt_xml
            else:
                pool_vrt_xml = None
                train_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in train_scene_ids])
                val_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in val_scene_ids])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Не удалось построить VRT: {exc}")
        else:
            dataset = PreparedDataset(
                train_vrt_xml=train_vrt_xml,
                val_vrt_xml=val_vrt_xml,
                pool_vrt_xml=pool_vrt_xml,
                annotation_file=None,
                class_annotations=[
                    DatasetClassAnnotation(
                        class_id=class_id,
                        slug=class_request.slug,
                        name=class_request.name,
                        annotation_file=Path(class_request.annotation_file).resolve().as_posix(),
                        priority=class_request.priority,
                    )
                    for class_id, class_request in enumerate(classes, start=1)
                ],
            )

    report = _build_report(
        scenes=scenes,
        rows=rows,
        scene_to_image=scene_to_image,
        train_names=train_names,
        val_names=val_names,
        pool_names=pool_names,
        missing_files=missing_files,
        errors=errors,
        split_granularity=request.split_granularity,
        negative_scene_limit=request.negative_scene_limit,
        selected_positive_scenes_count=selection.positive_count,
        selected_negative_scenes_count=selection.negative_count,
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


def _unique_preserving_order(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _select_scene_rows(
    rows: list[SceneObjectCount],
    *,
    negative_scene_limit: int | None,
    seed: int,
) -> _SceneSelection:
    if negative_scene_limit is None:
        selected_rows = list(rows)
    else:
        positives = [row for row in rows if row.object_count > 0]
        negatives = [row for row in rows if row.object_count <= 0]
        rng = random.Random(seed)
        tie_break = {row.scene_name: rng.random() for row in negatives}
        selected_negative_names = {
            row.scene_name
            for row in sorted(
                negatives,
                key=lambda item: (tie_break[item.scene_name], item.scene_name),
            )[:negative_scene_limit]
        }
        selected_names = {
            row.scene_name for row in positives
        } | selected_negative_names
        selected_rows = [row for row in rows if row.scene_name in selected_names]

    return _SceneSelection(
        rows=selected_rows,
        positive_count=sum(1 for row in selected_rows if row.object_count > 0),
        negative_count=sum(1 for row in selected_rows if row.object_count <= 0),
    )


def _build_report(
    *,
    scenes: list[str],
    rows: list[SceneObjectCount],
    scene_to_image: dict[str, Path],
    train_names: set[str],
    val_names: set[str],
    missing_files: list[str],
    errors: list[str],
    pool_names: set[str] | None = None,
    split_granularity: str = "scene",
    negative_scene_limit: int | None = None,
    selected_positive_scenes_count: int = 0,
    selected_negative_scenes_count: int = 0,
) -> DatasetPreparationReport:
    count_by_scene = {row.scene_name: row.object_count for row in rows}
    pool_names = pool_names or set()
    scene_reports = [
        DatasetSceneReport(
            scene_id=scene,
            image_path=scene_to_image[scene].as_posix() if scene in scene_to_image else None,
            object_count=max(0, int(count_by_scene.get(scene, 0))),
            split=_scene_split(scene, scene_to_image, train_names, val_names, pool_names),
        )
        for scene in scenes
    ]
    train_objects = sum(item.object_count for item in scene_reports if item.split == "train")
    val_objects = sum(item.object_count for item in scene_reports if item.split == "val")
    return DatasetPreparationReport(
        status="error" if errors else "ok",
        split_granularity=split_granularity,
        negative_scene_limit=negative_scene_limit,
        selected_positive_scenes_count=selected_positive_scenes_count,
        selected_negative_scenes_count=selected_negative_scenes_count,
        scenes_total=len(scenes),
        scenes_found=len(scene_to_image),
        objects_total=sum(item.object_count for item in scene_reports),
        train_scenes_count=sum(1 for item in scene_reports if item.split == "train"),
        train_objects_count=train_objects,
        val_scenes_count=sum(1 for item in scene_reports if item.split == "val"),
        val_objects_count=val_objects,
        scenes=scene_reports,
        missing_files=missing_files,
        errors=errors,
    )


def _scene_split(
    scene: str,
    scene_to_image: dict[str, Path],
    train_names: set[str],
    val_names: set[str],
    pool_names: set[str],
) -> str:
    if scene not in scene_to_image:
        return "missing"
    if scene in train_names:
        return "train"
    if scene in val_names:
        return "val"
    if scene in pool_names:
        return "pool"
    return "excluded"
