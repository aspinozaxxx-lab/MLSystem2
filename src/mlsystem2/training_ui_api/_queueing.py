"""Внутренние правила приоритета очереди training UI API."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ._models import JobRow
from .contracts import JobSource, JobStatus, JobType


DATASET_EDITOR_PSEUDO_OPERATION = "dataset_editor_scene_pseudo"


class _QueueRow(Protocol):
    type: str
    source: str
    status: str
    queue_position: int
    created_at: datetime


_JOB_PRIORITIES = {
    (JobType.INFERENCE.value, JobSource.MANUAL.value): 40,
    (JobType.TRAINING.value, JobSource.MANUAL.value): 30,
    (JobType.INFERENCE.value, JobSource.AUTOMATION.value): 20,
    (JobType.TRAINING.value, JobSource.AUTOMATION.value): 10,
}
_QUEUE_POSITION_BLOCK = 10_000
_QUEUE_POSITION_BASES = {
    key: index * _QUEUE_POSITION_BLOCK
    for index, key in enumerate(
        sorted(_JOB_PRIORITIES, key=lambda item: _JOB_PRIORITIES[item], reverse=True),
        start=1,
    )
}
_MIN_MANAGED_QUEUE_POSITION = min(_QUEUE_POSITION_BASES.values())
_ACTIVE_JOB_STATUSES = {
    JobStatus.QUEUED.value,
    JobStatus.RUNNING.value,
    JobStatus.PAUSED.value,
}


def job_priority(row: _QueueRow) -> int:
    return _JOB_PRIORITIES.get((row.type, row.source), 0)


def queue_sort_key(row: _QueueRow) -> tuple[int, int, int, datetime]:
    status_rank = 0 if row.status in {JobStatus.RUNNING.value, JobStatus.PAUSED.value} else 1
    return status_rank, row.queue_position, -job_priority(row), row.created_at


def dispatch_sort_key(row: _QueueRow) -> tuple[int, int, datetime]:
    return row.queue_position, -job_priority(row), row.created_at


def next_queue_position(session: Session, job_type: JobType, source: JobSource) -> int:
    ensure_queue_positions(session)
    base = _queue_position_base(job_type.value, source.value)
    upper_bound = base + _QUEUE_POSITION_BLOCK
    positions = session.scalars(
        select(JobRow.queue_position).where(
            JobRow.status.in_(_ACTIVE_JOB_STATUSES),
            JobRow.queue_position >= base,
            JobRow.queue_position < upper_bound,
        )
    ).all()
    return (max(positions) if positions else base) + 1


def ensure_queue_positions(session: Session) -> None:
    rows = session.scalars(
        select(JobRow).where(JobRow.status.in_(_ACTIVE_JOB_STATUSES))
    ).all()
    legacy_rows = [row for row in rows if row.queue_position < _MIN_MANAGED_QUEUE_POSITION]
    if not legacy_rows:
        return
    next_positions = {
        base: max(
            [row.queue_position for row in rows if base <= row.queue_position < base + _QUEUE_POSITION_BLOCK],
            default=base,
        )
        for base in _QUEUE_POSITION_BASES.values()
    }
    legacy_rows.sort(
        key=lambda row: (
            _queue_position_base(row.type, row.source),
            row.queue_position,
            row.created_at,
        )
    )
    for row in legacy_rows:
        base = _queue_position_base(row.type, row.source)
        next_positions[base] += 1
        row.queue_position = next_positions[base]
    session.flush()


def _queue_position_base(job_type: str, source: str) -> int:
    return _QUEUE_POSITION_BASES.get((job_type, source), _QUEUE_POSITION_BLOCK * 99)
