"""Выбор и восстановление датасетных шаблонов после смены ключа датасета."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ._models import (
    DatasetClassRow,
    DatasetRow,
    InferenceTemplateRow,
    TrainingTemplateRow,
)


def reconcile_dataset_template_keys(session: Session) -> int:
    """Перепривязать осиротевшие шаблоны к однозначному активному датасету."""

    names_by_key, keys_by_name = _active_dataset_identities(session)
    rebound = 0
    for model in (TrainingTemplateRow, InferenceTemplateRow):
        rows = list(
            session.scalars(select(model).where(model.dataset_key.is_not(None))).all()
        )
        occupied = {
            (row.architecture, row.dataset_key)
            for row in rows
            if row.dataset_key in names_by_key
        }
        for row in rows:
            current_name = names_by_key.get(row.dataset_key)
            if current_name is not None:
                row.dataset_name = current_name
                continue
            target_keys = keys_by_name.get(str(row.dataset_name or ""), [])
            if len(target_keys) != 1:
                continue
            target_key = target_keys[0]
            identity = (row.architecture, target_key)
            if identity in occupied:
                continue
            row.dataset_key = target_key
            row.dataset_name = names_by_key[target_key]
            occupied.add(identity)
            rebound += 1
    return rebound


def dataset_training_template_row(
    session: Session,
    architecture: str,
    dataset_key: str,
) -> TrainingTemplateRow | None:
    return _dataset_template_row(session, TrainingTemplateRow, architecture, dataset_key)


def dataset_inference_template_row(
    session: Session,
    architecture: str,
    dataset_key: str,
) -> InferenceTemplateRow | None:
    return _dataset_template_row(session, InferenceTemplateRow, architecture, dataset_key)


def effective_training_template_row(
    session: Session,
    architecture: str,
    dataset_key: str | None,
) -> TrainingTemplateRow | None:
    if dataset_key:
        row = dataset_training_template_row(session, architecture, dataset_key)
        if row is not None and row.is_active:
            return row
    return session.scalar(
        select(TrainingTemplateRow).where(
            TrainingTemplateRow.architecture == architecture,
            TrainingTemplateRow.dataset_key.is_(None),
        )
    )


def effective_inference_template_row(
    session: Session,
    architecture: str,
    dataset_key: str | None,
) -> InferenceTemplateRow | None:
    if dataset_key:
        row = dataset_inference_template_row(session, architecture, dataset_key)
        if row is not None and row.is_active:
            return row
    return session.scalar(
        select(InferenceTemplateRow).where(
            InferenceTemplateRow.architecture == architecture,
            InferenceTemplateRow.dataset_key.is_(None),
        )
    )


def _dataset_template_row(
    session: Session,
    model: Any,
    architecture: str,
    dataset_key: str,
):
    exact = session.scalar(
        select(model).where(
            model.architecture == architecture,
            model.dataset_key == dataset_key,
        )
    )
    if exact is not None:
        return exact

    names_by_key, _ = _active_dataset_identities(session)
    dataset_name = names_by_key.get(dataset_key)
    if dataset_name is None:
        return None
    candidates = list(
        session.scalars(
            select(model).where(
                model.architecture == architecture,
                model.dataset_name == dataset_name,
                model.dataset_key.is_not(None),
            )
        ).all()
    )
    return candidates[0] if len(candidates) == 1 else None


def _active_dataset_identities(
    session: Session,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    names_by_key: dict[str, str] = {}
    keys_by_name: defaultdict[str, list[str]] = defaultdict(list)
    rows = session.execute(
        select(DatasetRow.key, DatasetRow.name, DatasetClassRow.name)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(DatasetRow.deleted_at.is_(None))
    ).all()
    for dataset_key, dataset_name, class_name in rows:
        display_name = f"{class_name}\\{dataset_name}"
        names_by_key[dataset_key] = display_name
        keys_by_name[display_name].append(dataset_key)
    return names_by_key, dict(keys_by_name)


__all__ = [
    "dataset_inference_template_row",
    "dataset_training_template_row",
    "effective_inference_template_row",
    "effective_training_template_row",
    "reconcile_dataset_template_keys",
]
