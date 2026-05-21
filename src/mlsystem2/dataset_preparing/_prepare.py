"""Реализация подготовки датасета."""

from __future__ import annotations

from pathlib import Path

from ._object_counts import SceneObjectCount, count_objects_per_scene
from ._raster_validation import validate_rasters
from ._scene_matching import filter_existing_scenes, index_image_files, read_scene_list
from ._split import split_train_val_by_object_counts
from ._vrt import build_vrt_xml
from .contracts import (
    DatasetPreparationReport,
    DatasetPreparationRequest,
    DatasetPreparationResult,
    DatasetSceneReport,
    PreparedDataset,
)

SPLIT_SEED = 42


def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    images_dir = Path(request.images_dir)
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
    split = split_train_val_by_object_counts(
        found_rows,
        target_val_fraction=request.val_fraction,
        seed=SPLIT_SEED,
    )
    train_scene_ids = [row.scene_name for row in split.train]
    val_scene_ids = [row.scene_name for row in split.val]
    train_names = set(train_scene_ids)
    val_names = set(val_scene_ids)

    if not found_rows:
        errors.append("Не найдено ни одного снимка из списка сцен.")
    elif not split.train or not split.val:
        errors.append("Недостаточно найденных сцен для построения train и val VRT.")

    validation = validate_rasters(scene_to_image) if scene_to_image else None
    if validation is not None:
        errors.extend(validation.errors)

    dataset: PreparedDataset | None = None
    if not errors and validation is not None:
        raster_by_scene = {raster.scene_id: raster for raster in validation.rasters}
        try:
            train_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in train_scene_ids])
            val_vrt_xml = build_vrt_xml([raster_by_scene[scene] for scene in val_scene_ids])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Не удалось построить VRT: {exc}")
        else:
            dataset = PreparedDataset(
                train_vrt_xml=train_vrt_xml,
                val_vrt_xml=val_vrt_xml,
                annotation_file=annotation_file.resolve().as_posix(),
            )

    report = _build_report(
        scenes=scenes,
        rows=rows,
        scene_to_image=scene_to_image,
        train_names=train_names,
        val_names=val_names,
        missing_files=missing_files,
        errors=errors,
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


def _build_report(
    *,
    scenes: list[str],
    rows: list[SceneObjectCount],
    scene_to_image: dict[str, Path],
    train_names: set[str],
    val_names: set[str],
    missing_files: list[str],
    errors: list[str],
) -> DatasetPreparationReport:
    count_by_scene = {row.scene_name: row.object_count for row in rows}
    scene_reports = [
        DatasetSceneReport(
            scene_id=scene,
            image_path=scene_to_image[scene].as_posix() if scene in scene_to_image else None,
            object_count=max(0, int(count_by_scene.get(scene, 0))),
            split=_scene_split(scene, scene_to_image, train_names, val_names),
        )
        for scene in scenes
    ]
    train_objects = sum(item.object_count for item in scene_reports if item.split == "train")
    val_objects = sum(item.object_count for item in scene_reports if item.split == "val")
    return DatasetPreparationReport(
        status="error" if errors else "ok",
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
) -> str:
    if scene not in scene_to_image:
        return "missing"
    if scene in train_names:
        return "train"
    if scene in val_names:
        return "val"
    return "missing"
