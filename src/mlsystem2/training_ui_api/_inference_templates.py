"""Однократный перенос исторических настроек в общие именованные шаблоны."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ._dataset_catalog import primary_training_result
from ._models import DatasetClassRow, DatasetRow, InferenceTemplateRow
from ._template_selection import _dataset_template_row
from ._templates import INFERENCE_CONFIG_SCHEMA, sanitize_inference_template_config


def _signature(values: dict) -> str:
    normalized = {key: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                  else value for key, value in values.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def migrate_inference_templates(session: Session) -> None:
    """Сохранить эффективные параметры классов, объединяя одинаковые наборы."""

    if session.scalar(select(InferenceTemplateRow.id).where(
        InferenceTemplateRow.architecture.is_(None),
        InferenceTemplateRow.archived_at.is_(None),
    ).limit(1)) is not None:
        return
    legacy = list(session.scalars(select(InferenceTemplateRow).where(
        InferenceTemplateRow.architecture.is_not(None),
    )).all())
    now = datetime.now(timezone.utc)
    for row in legacy:
        row.archived_at = row.archived_at or now
    session.flush()
    datasets = list(session.scalars(select(DatasetRow)).all())
    grouped: dict[str, InferenceTemplateRow] = {}
    for class_row in session.scalars(select(DatasetClassRow).order_by(DatasetClassRow.name)).all():
        class_datasets = [row for row in datasets if row.class_id == class_row.id]
        keys = {row.key for row in class_datasets}
        names = {f"{class_row.name}\\{row.name}" for row in class_datasets}
        candidates = [row for row in legacy if row.is_active and row.dataset_key is not None
                      and (row.dataset_key in keys or row.dataset_name in names)]
        result = primary_training_result(session, class_row.key)
        selected = None
        if result is not None:
            selected = _dataset_template_row(
                session, InferenceTemplateRow, result.architecture,
                result.dataset_key or result.class_key,
            )
            if selected is None or not selected.is_active:
                selected = next((row for row in legacy if row.architecture == result.architecture
                                 and row.dataset_key is None), None)
        elif candidates:
            signatures = {row.id: _signature(sanitize_inference_template_config(row.default_config))
                          for row in candidates}
            counts = Counter(signatures.values())
            selected = max(candidates, key=lambda row: (
                counts[signatures[row.id]],
                row.updated_at.replace(tzinfo=timezone.utc) if row.updated_at.tzinfo is None
                else row.updated_at.astimezone(timezone.utc),
                str(row.id),
            ))
        copied = sanitize_inference_template_config(selected.default_config if selected else None)
        signature = _signature(copied)
        template = grouped.get(signature)
        if template is None:
            source = selected.source if selected else "manual"
            source_run = selected.source_mlflow_run_id if selected else None
            template = InferenceTemplateRow(
                display_name=f"Перенесённые настройки {len(grouped) + 1}",
                description="Параметры сохранены при переходе на общие шаблоны инференса.",
                config_schema=deepcopy(INFERENCE_CONFIG_SCHEMA),
                default_config=copied,
                baseline_default_config=deepcopy(copied),
                source=source, baseline_source=source,
                source_mlflow_run_id=source_run, baseline_source_mlflow_run_id=source_run,
                is_active=True, version=1,
            )
            session.add(template)
            session.flush()
            grouped[signature] = template
        class_row.inference_template_id = template.id
    session.flush()
