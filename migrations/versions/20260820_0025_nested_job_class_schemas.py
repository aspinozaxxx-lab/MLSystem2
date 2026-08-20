"""Канонизировать вложенные схемы классов в исторических заданиях.

Revision ID: 20260820_0025
Revises: 20260820_0024
Create Date: 2026-08-20
"""

from __future__ import annotations

import os
from typing import Any

import sqlalchemy as sa
from alembic import context, op


revision = "20260820_0025"
down_revision = "20260820_0024"
branch_labels = None
depends_on = None


_CLASS_SCHEMA_FIELDS = frozenset({"class_schema", "object_types"})


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def _table(name: str, *columns: sa.Column[Any]) -> sa.Table:
    return sa.table(name, *columns, schema=_schema())


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
        old = dict(old_by_id.get(class_id) or {})
        old_slug = str(old.get("slug") or "")
        new_slug = str(target["slug"])
        if old_slug and old_slug != new_slug:
            mapping[old_slug] = new_slug
        old.update(target)
        result.append(old)
    return result, mapping


def _nested_schema_mapping(
    value: object,
    target_schema: list[dict[str, Any]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _CLASS_SCHEMA_FIELDS and isinstance(item, list):
                _schema_value, item_mapping = _canonical_schema(item, target_schema)
                mapping.update(item_mapping)
            else:
                mapping.update(_nested_schema_mapping(item, target_schema))
    elif isinstance(value, list):
        for item in value:
            mapping.update(_nested_schema_mapping(item, target_schema))
    return mapping


def _remap(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            mapping.get(str(key), str(key)): _remap(item, mapping)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_remap(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _canonicalize_nested_schemas(
    value: Any,
    target_schema: list[dict[str, Any]],
) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in _CLASS_SCHEMA_FIELDS and isinstance(item, list):
                result[key], _mapping = _canonical_schema(item, target_schema)
            else:
                result[key] = _canonicalize_nested_schemas(item, target_schema)
        return result
    if isinstance(value, list):
        return [_canonicalize_nested_schemas(item, target_schema) for item in value]
    return value


def upgrade() -> None:
    if context.is_offline_mode():
        op.execute(
            "DO $$ BEGIN RAISE EXCEPTION "
            "'Миграция 20260820_0025 требует online-режим Alembic: она читает JSON заданий'; "
            "END $$"
        )
        return

    connection = op.get_bind()
    classes = _table(
        "dataset_classes",
        sa.column("id", sa.Uuid()),
        sa.column("name", sa.String()),
        sa.column("technical_name", sa.String()),
    )
    datasets = _table(
        "datasets",
        sa.column("id", sa.Uuid()),
        sa.column("key", sa.String()),
        sa.column("class_id", sa.Uuid()),
    )
    relations = _table(
        "managed_dataset_sources",
        sa.column("managed_dataset_id", sa.Uuid()),
        sa.column("source_dataset_id", sa.Uuid()),
        sa.column("priority", sa.Integer()),
        sa.column("object_type_id", sa.Integer()),
        sa.column("object_type_slug", sa.String()),
        sa.column("object_type_name", sa.String()),
        sa.column("color", sa.String()),
    )
    jobs = _table(
        "jobs",
        sa.column("id", sa.Uuid()),
        sa.column("dataset_key", sa.String()),
        sa.column("config", sa.JSON()),
    )

    class_rows = {
        row["id"]: dict(row)
        for row in connection.execute(sa.select(classes)).mappings().all()
    }
    dataset_rows = {
        row["id"]: dict(row)
        for row in connection.execute(sa.select(datasets)).mappings().all()
    }
    schemas_by_dataset_id: dict[Any, list[dict[str, Any]]] = {}
    for relation in connection.execute(sa.select(relations)).mappings().all():
        source = dataset_rows.get(relation["source_dataset_id"])
        source_class = class_rows.get(source["class_id"]) if source is not None else None
        if source_class is None:
            continue
        schemas_by_dataset_id.setdefault(relation["managed_dataset_id"], []).append(
            {
                "id": int(relation["object_type_id"]),
                "slug": str(source_class["technical_name"]),
                "name": str(source_class["name"]),
                "color": str(relation["color"]).upper(),
                "priority": int(relation["priority"] or 0),
            }
        )
    for schema_value in schemas_by_dataset_id.values():
        schema_value.sort(key=lambda item: int(item["id"]))
    schemas_by_dataset_key = {
        str(dataset_rows[dataset_id]["key"]): schema_value
        for dataset_id, schema_value in schemas_by_dataset_id.items()
        if dataset_id in dataset_rows
    }

    for row in connection.execute(sa.select(jobs)).mappings().all():
        target_schema = schemas_by_dataset_key.get(str(row["dataset_key"] or ""))
        if not target_schema:
            continue
        config = row["config"] if isinstance(row["config"], dict) else {}
        mapping = _nested_schema_mapping(config, target_schema)
        canonical = _canonicalize_nested_schemas(_remap(config, mapping), target_schema)
        if canonical != config:
            connection.execute(
                sa.update(jobs).where(jobs.c.id == row["id"]).values(config=canonical)
            )


def downgrade() -> None:
    # Старые технические идентификаторы неоднозначны и намеренно не восстанавливаются.
    pass
