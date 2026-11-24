"""Route registration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._annotation_versions import AnnotationVersionConflict
from mlsystem2.training_ui_api._config import TrainingUIAPIConfig
from mlsystem2.training_ui_api._dataset_editor import (
    DatasetEditorConflict,
    DatasetEditorGitError,
    recover_annotation_publications,
)
from mlsystem2.training_ui_api._markup_lock import markup_write_lock


@dataclass(frozen=True)
class RouteContext:
    config: TrainingUIAPIConfig
    get_db: Callable[[Request], Iterator[Session]]
    authenticated: Callable[[Request], str]
    pseudolabel_authenticated: Callable[[Request], str]


@contextmanager
def editor_errors() -> Iterator[None]:
    try:
        yield
    except (DatasetEditorConflict, AnnotationVersionConflict) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except DatasetEditorGitError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@contextmanager
def editor_transaction(session: Session, config: TrainingUIAPIConfig) -> Iterator[None]:
    with editor_errors(), markup_write_lock(config):
        recover_annotation_publications(session, config)
        yield
        session.commit()


__all__ = ["RouteContext", "editor_errors", "editor_transaction"]
