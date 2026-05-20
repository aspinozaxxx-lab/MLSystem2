"""Разбиение сцен на train и val по количеству объектов."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ._object_counts import SceneObjectCount


@dataclass
class TrainValSplit:
    train: list[SceneObjectCount]
    val: list[SceneObjectCount]
    summary: dict[str, Any]


def split_train_val_by_object_counts(
    counts: list[SceneObjectCount],
    *,
    target_val_fraction: float = 0.2,
    seed: int = 42,
    target_val_objects: int | None = None,
    min_val_scenes: int | None = None,
    include_zero_object_scenes: bool = True,
) -> TrainValSplit:
    if not counts:
        return TrainValSplit([], [], _split_summary([], [], seed, target_val_fraction))

    rng = random.Random(seed)
    positives = [item for item in counts if item.object_count > 0]
    zeros = [item for item in counts if item.object_count <= 0]
    tie_break = {id(item): rng.random() for item in counts}
    positives.sort(key=lambda item: (-item.object_count, tie_break[id(item)], item.scene_name))
    zeros.sort(key=lambda item: (tie_break[id(item)], item.scene_name))

    total_objects = sum(item.object_count for item in positives)
    target_objects = (
        target_val_objects
        if target_val_objects is not None
        else int(round(total_objects * target_val_fraction))
    )
    if total_objects > 0:
        target_objects = max(1, target_objects)
    target_scene_count = (
        max(1, int(min_val_scenes))
        if min_val_scenes is not None and len(counts) > 1
        else 1
    )

    val_ids: set[int] = set()
    val_objects = 0
    for index, item in enumerate(positives):
        if not val_ids:
            remaining = positives[index + 1 :]
            if (
                target_objects > 0
                and item.object_count > target_objects
                and any(candidate.object_count <= target_objects for candidate in remaining)
            ):
                continue
            val_ids.add(id(item))
            val_objects += item.object_count
            continue
        if val_objects >= target_objects and len(val_ids) >= target_scene_count:
            break
        current_gap = abs(target_objects - val_objects)
        next_gap = abs(target_objects - (val_objects + item.object_count))
        if (
            (val_objects < target_objects and val_objects + item.object_count <= target_objects)
            or next_gap <= current_gap
            or len(val_ids) < target_scene_count
        ):
            val_ids.add(id(item))
            val_objects += item.object_count

    if total_objects > 0 and positives and not val_ids:
        fallback = min(
            positives,
            key=lambda item: (
                abs(target_objects - item.object_count),
                item.object_count,
                tie_break[id(item)],
                item.scene_name,
            ),
        )
        val_ids.add(id(fallback))

    if include_zero_object_scenes:
        desired_zero_val = int(round(len(zeros) * target_val_fraction))
        if zeros and len(counts) > 1 and desired_zero_val == 0:
            desired_zero_val = 1
        for item in zeros[:desired_zero_val]:
            val_ids.add(id(item))

    if len(counts) > 1 and not val_ids:
        val_ids.add(id(counts[-1]))
    if len(val_ids) == len(counts) and len(counts) > 1:
        removable = next((item for item in zeros if id(item) in val_ids), None)
        if removable is None:
            removable = next(item for item in positives if id(item) in val_ids)
        val_ids.remove(id(removable))

    train = [item for item in counts if id(item) not in val_ids]
    val = [item for item in counts if id(item) in val_ids]
    train_names = {item.scene_name for item in train}
    val_names = {item.scene_name for item in val}
    if train_names & val_names:
        raise RuntimeError(f"Разбиение train/val содержит пересечение: {sorted(train_names & val_names)}")

    return TrainValSplit(
        train=train,
        val=val,
        summary=_split_summary(train, val, seed, target_val_fraction),
    )


def _split_summary(
    train: list[SceneObjectCount],
    val: list[SceneObjectCount],
    seed: int,
    target_val_fraction: float,
) -> dict[str, Any]:
    train_objects = sum(item.object_count for item in train)
    val_objects = sum(item.object_count for item in val)
    total_files = len(train) + len(val)
    total_objects = train_objects + val_objects
    rows = train + val
    return {
        "train_files": len(train),
        "train_objects": train_objects,
        "val_files": len(val),
        "val_objects": val_objects,
        "total_files": total_files,
        "total_objects": total_objects,
        "scenes_with_objects": sum(1 for item in rows if item.object_count > 0),
        "scenes_without_objects": sum(1 for item in rows if item.object_count <= 0),
        "val_fraction_by_files": (len(val) / total_files) if total_files else 0.0,
        "val_fraction_by_objects": (val_objects / total_objects) if total_objects else 0.0,
        "seed": seed,
        "target_val_fraction": target_val_fraction,
    }
