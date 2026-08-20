"""Добавить технические имена классов и канонизировать multiclass-идентификаторы.

Revision ID: 20260820_0024
Revises: 20260819_0023
Create Date: 2026-08-20
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import context, op


revision = "20260820_0024"
down_revision = "20260819_0023"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def _table(name: str, *columns: sa.Column[Any]) -> sa.Table:
    return sa.table(name, *columns, schema=_schema())


def _technical_name(value: object, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", str(value or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_-")
    if not normalized or not re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?", normalized):
        normalized = fallback
    return normalized[:160].rstrip("_-")


def _canonical_schema(
    old_schema: object,
    target_schema: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    old_items = old_schema if isinstance(old_schema, list) else []
    old_by_id = {
        int(item.get("id", item.get("class_id", index))): item
        for index, item in enumerate(old_items, start=1)
        if isinstance(item, dict)
    }
    mapping: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    for target in target_schema:
        class_id = int(target["id"])
        old = old_by_id.get(class_id, {})
        old_slug = str(old.get("slug") or "").strip()
        new_slug = str(target["slug"])
        if old_slug and old_slug != new_slug:
            mapping[old_slug] = new_slug
        merged = dict(old)
        merged.update(target)
        result.append(merged)
    return result, mapping


def _remap_json(value: Any, mapping: dict[str, str]) -> Any:
    if not mapping:
        return value
    if isinstance(value, dict):
        return {
            mapping.get(str(key), str(key)): _remap_json(item, mapping)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_remap_json(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _rewrite_geojson(path_value: object, mapping: dict[str, str]) -> int | None:
    if not mapping or not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_file() or path.suffix.casefold() != ".geojson":
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    remapped = _remap_json(payload, mapping)
    if remapped == payload:
        return path.stat().st_size
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(remapped, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def upgrade() -> None:
    if context.is_offline_mode():
        op.execute(
            "DO $$ BEGIN RAISE EXCEPTION "
            "'Миграция 20260820_0024 требует online-режим Alembic: она читает данные "
            "и атомарно обновляет сохранённые GeoJSON'; END $$"
        )
        return
    schema = _schema()
    op.add_column(
        "dataset_classes",
        sa.Column("technical_name", sa.String(length=160), nullable=True),
        schema=schema,
    )
    connection = op.get_bind()
    classes = _table(
        "dataset_classes",
        sa.column("id", sa.Uuid()),
        sa.column("name", sa.String()),
        sa.column("primary_dataset_id", sa.Uuid()),
        sa.column("technical_name", sa.String()),
    )
    datasets = _table(
        "datasets",
        sa.column("id", sa.Uuid()),
        sa.column("key", sa.String()),
        sa.column("class_id", sa.Uuid()),
        sa.column("name", sa.String()),
        sa.column("model_name_stem", sa.String()),
    )
    class_rows = connection.execute(sa.select(classes)).mappings().all()
    dataset_rows = connection.execute(sa.select(datasets)).mappings().all()
    datasets_by_class: dict[Any, list[dict[str, Any]]] = {}
    dataset_by_id: dict[Any, dict[str, Any]] = {}
    dataset_by_key: dict[str, dict[str, Any]] = {}
    for row in dataset_rows:
        item = dict(row)
        datasets_by_class.setdefault(row["class_id"], []).append(item)
        dataset_by_id[row["id"]] = item
        dataset_by_key[str(row["key"])] = item

    used: set[str] = set()
    technical_by_class: dict[Any, str] = {}
    class_name_by_id: dict[Any, str] = {}
    for row in class_rows:
        class_id = row["id"]
        class_name_by_id[class_id] = str(row["name"])
        candidates = datasets_by_class.get(class_id, [])
        primary = dataset_by_id.get(row["primary_dataset_id"])
        main = next(
            (item for item in candidates if str(item["name"]).casefold() == "main"),
            None,
        )
        stem = next(
            (
                item.get("model_name_stem")
                for item in (main, primary, *candidates)
                if item is not None and item.get("model_name_stem")
            ),
            None,
        )
        raw_id = str(class_id).replace("-", "")
        base = _technical_name(stem, f"class_{raw_id[:12]}")
        value = base
        suffix = 2
        while value in used:
            value = f"{base[:156]}_{suffix}"
            suffix += 1
        used.add(value)
        technical_by_class[class_id] = value
        connection.execute(
            sa.update(classes).where(classes.c.id == class_id).values(technical_name=value)
        )

    op.alter_column(
        "dataset_classes",
        "technical_name",
        existing_type=sa.String(length=160),
        nullable=False,
        schema=schema,
    )
    op.create_index(
        "ix_dataset_classes_technical_name",
        "dataset_classes",
        ["technical_name"],
        unique=True,
        schema=schema,
    )

    relations = _table(
        "managed_dataset_sources",
        sa.column("id", sa.Uuid()),
        sa.column("managed_dataset_id", sa.Uuid()),
        sa.column("source_dataset_id", sa.Uuid()),
        sa.column("priority", sa.Integer()),
        sa.column("object_type_id", sa.Integer()),
        sa.column("object_type_slug", sa.String()),
        sa.column("object_type_name", sa.String()),
        sa.column("color", sa.String()),
    )
    relation_rows = connection.execute(sa.select(relations)).mappings().all()
    target_schema_by_id: dict[Any, list[dict[str, Any]]] = {}
    for row in relation_rows:
        source = dataset_by_id.get(row["source_dataset_id"])
        if source is None:
            continue
        source_class_id = source["class_id"]
        slug = technical_by_class[source_class_id]
        name = class_name_by_id[source_class_id]
        connection.execute(
            sa.update(relations)
            .where(relations.c.id == row["id"])
            .values(object_type_slug=slug, object_type_name=name)
        )
        target_schema_by_id.setdefault(row["managed_dataset_id"], []).append(
            {
                "id": int(row["object_type_id"]),
                "slug": slug,
                "name": name,
                "color": str(row["color"]).upper(),
                "priority": int(row["priority"] or 0),
            }
        )
    for value in target_schema_by_id.values():
        value.sort(key=lambda item: int(item["id"]))
    target_schema_by_key = {
        str(dataset_by_id[target_id]["key"]): value
        for target_id, value in target_schema_by_id.items()
        if target_id in dataset_by_id
    }

    training_results = _table(
        "training_results",
        sa.column("id", sa.Uuid()),
        sa.column("dataset_key", sa.String()),
        sa.column("class_key", sa.String()),
        sa.column("class_schema", sa.JSON()),
        sa.column("training_metrics", sa.JSON()),
    )
    result_mapping: dict[Any, dict[str, str]] = {}
    for row in connection.execute(sa.select(training_results)).mappings().all():
        target_schema = target_schema_by_key.get(str(row["dataset_key"] or ""))
        target_schema = target_schema or target_schema_by_key.get(str(row["class_key"] or ""))
        if not target_schema:
            continue
        canonical, mapping = _canonical_schema(row["class_schema"], target_schema)
        result_mapping[row["id"]] = mapping
        connection.execute(
            sa.update(training_results)
            .where(training_results.c.id == row["id"])
            .values(
                class_schema=canonical,
                training_metrics=_remap_json(row["training_metrics"] or {}, mapping),
            )
        )

    jobs = _table(
        "jobs",
        sa.column("id", sa.Uuid()),
        sa.column("dataset_key", sa.String()),
        sa.column("config", sa.JSON()),
    )
    for row in connection.execute(sa.select(jobs)).mappings().all():
        target_schema = target_schema_by_key.get(str(row["dataset_key"] or ""))
        if not target_schema:
            continue
        config = row["config"] if isinstance(row["config"], dict) else {}
        old_schema = config.get("class_schema") or config.get("object_types") or []
        canonical, mapping = _canonical_schema(old_schema, target_schema)
        remapped = _remap_json(config, mapping)
        if isinstance(remapped, dict):
            if "class_schema" in remapped:
                remapped["class_schema"] = canonical
            if "object_types" in remapped:
                remapped["object_types"] = canonical
        connection.execute(
            sa.update(jobs).where(jobs.c.id == row["id"]).values(config=remapped)
        )

    test_samples = _table(
        "test_samples",
        sa.column("id", sa.Uuid()),
        sa.column("dataset_key", sa.String()),
        sa.column("class_schema", sa.JSON()),
        sa.column("evaluation_metrics", sa.JSON()),
    )
    sample_mapping: dict[Any, dict[str, str]] = {}
    for row in connection.execute(sa.select(test_samples)).mappings().all():
        target_schema = target_schema_by_key.get(str(row["dataset_key"] or ""))
        if not target_schema:
            continue
        canonical, mapping = _canonical_schema(row["class_schema"], target_schema)
        sample_mapping[row["id"]] = mapping
        connection.execute(
            sa.update(test_samples)
            .where(test_samples.c.id == row["id"])
            .values(
                class_schema=canonical,
                evaluation_metrics=_remap_json(row["evaluation_metrics"] or {}, mapping),
            )
        )

    test_tiles = _table(
        "test_sample_tiles",
        sa.column("id", sa.Uuid()),
        sa.column("test_sample_id", sa.Uuid()),
        sa.column("class_object_counts", sa.JSON()),
        sa.column("evaluation_metrics", sa.JSON()),
    )
    for row in connection.execute(sa.select(test_tiles)).mappings().all():
        mapping = sample_mapping.get(row["test_sample_id"])
        if not mapping:
            continue
        connection.execute(
            sa.update(test_tiles)
            .where(test_tiles.c.id == row["id"])
            .values(
                class_object_counts=_remap_json(row["class_object_counts"] or {}, mapping),
                evaluation_metrics=_remap_json(row["evaluation_metrics"] or {}, mapping),
            )
        )

    result_metrics = _table(
        "training_result_test_metrics",
        sa.column("training_result_id", sa.Uuid()),
        sa.column("metrics", sa.JSON()),
    )
    for row in connection.execute(sa.select(result_metrics)).mappings().all():
        mapping = result_mapping.get(row["training_result_id"])
        if mapping:
            connection.execute(
                sa.update(result_metrics)
                .where(result_metrics.c.training_result_id == row["training_result_id"])
                .values(metrics=_remap_json(row["metrics"] or {}, mapping))
            )

    stored_files = _table(
        "stored_files",
        sa.column("id", sa.Uuid()),
        sa.column("path", sa.Text()),
        sa.column("size_bytes", sa.BigInteger()),
    )
    pseudo_results = _table(
        "pseudo_markup_results",
        sa.column("id", sa.Uuid()),
        sa.column("training_result_id", sa.Uuid()),
        sa.column("geojson_file_id", sa.Uuid()),
    )
    pseudo_files = connection.execute(
        sa.select(
            pseudo_results.c.training_result_id,
            stored_files.c.id,
            stored_files.c.path,
        ).join(stored_files, stored_files.c.id == pseudo_results.c.geojson_file_id)
    ).mappings().all()
    for row in pseudo_files:
        size_bytes = _rewrite_geojson(
            row["path"],
            result_mapping.get(row["training_result_id"], {}),
        )
        if size_bytes is not None:
            connection.execute(
                sa.update(stored_files)
                .where(stored_files.c.id == row["id"])
                .values(size_bytes=size_bytes)
            )

    sample_root = Path(
        os.getenv(
            "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT",
            "/data/mlsystem2/training-ui/files",
        )
    ) / "test-samples"
    for sample_id, mapping in sample_mapping.items():
        if not mapping:
            continue
        root = sample_root / str(sample_id)
        if root.is_dir():
            for path in root.glob("tile_*.geojson"):
                _rewrite_geojson(path, mapping)


def downgrade() -> None:
    op.drop_index(
        "ix_dataset_classes_technical_name",
        table_name="dataset_classes",
        schema=_schema(),
    )
    op.drop_column("dataset_classes", "technical_name", schema=_schema())
