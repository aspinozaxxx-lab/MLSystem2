"""Вспомогательные функции разбиения датасета."""

from __future__ import annotations

from .contracts import DatasetSplit, SceneRecord


def object_count_balanced_split(records: list[SceneRecord], val_fraction: float) -> DatasetSplit:
    if not records:
        return DatasetSplit(train_scenes=[], val_scenes=[])

    total_objects = sum(record.object_count for record in records)
    target_val_objects = max(1, round(total_objects * val_fraction)) if total_objects > 0 else 0
    target_val_scenes = max(1, round(len(records) * val_fraction))

    sorted_records = sorted(records, key=lambda item: (-item.object_count, item.scene_id))
    val: list[SceneRecord] = []
    train: list[SceneRecord] = []
    val_objects = 0

    for record in sorted_records:
        need_more_objects = total_objects > 0 and val_objects < target_val_objects
        need_more_scenes = len(val) < target_val_scenes
        if need_more_objects or need_more_scenes:
            val.append(record)
            val_objects += record.object_count
        else:
            train.append(record)

    if not train and len(val) > 1:
        train.append(val.pop())

    return DatasetSplit(
        train_scenes=sorted(record.scene_id for record in train),
        val_scenes=sorted(record.scene_id for record in val),
    )
