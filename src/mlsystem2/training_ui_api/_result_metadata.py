"""Номера и пользовательские метаданные результатов; веса и метрики не изменяются."""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ._models import DatasetClassRow, JobRow, PseudoMarkupResultRow, TestSampleBatchItemRow, TrainingResultCounterRow, TrainingResultRow
from .contracts import TrainingResultCommentInfo, TrainingResultCommentUpdate


def next_result_number(session: Session, dataset_key: str) -> int:
    """Атомарный счётчик не переиспользует номера удалённых и отменённых результатов."""
    insert = sqlite_insert if session.get_bind().dialect.name == "sqlite" else postgres_insert
    counter = TrainingResultCounterRow
    statement = insert(counter).values(dataset_key=dataset_key, last_number=1)
    return session.execute(statement.on_conflict_do_update(
        index_elements=[counter.dataset_key], set_={"last_number": counter.last_number + 1},
    ).returning(counter.last_number)).scalar_one()


def result_comment(session: Session, result_id: UUID) -> TrainingResultCommentInfo:
    row = session.get(TrainingResultRow, result_id)
    if row is None or row.archived_at is not None:
        raise HTTPException(404, "Результат обучения не найден.")
    return TrainingResultCommentInfo(id=row.id, comment=row.comment)


def update_result_comment(session: Session, result_id: UUID, request: TrainingResultCommentUpdate) -> TrainingResultCommentInfo:
    result_comment(session, result_id)
    comment = request.comment.replace("\r\n", "\n").strip()
    updated = session.execute(update(TrainingResultRow).where(
        TrainingResultRow.id == result_id, TrainingResultRow.archived_at.is_(None),
        TrainingResultRow.comment == request.expected_comment,
    ).values(comment=comment, updated_at=TrainingResultRow.updated_at).returning(TrainingResultRow.id)).scalar_one_or_none()
    if updated is None:
        raise HTTPException(409, "Комментарий уже изменён в другой вкладке или результат удалён. Ваш текст не сохранён.")
    return TrainingResultCommentInfo(id=result_id, comment=comment)


def _references(value: object, result_id: str) -> bool:
    if isinstance(value, dict):
        return any(_references(item, result_id) for item in value.values())
    if isinstance(value, list):
        return any(_references(item, result_id) for item in value)
    return value == result_id


def archive_training_result(session: Session, result_id: UUID, username: str) -> None:
    row = session.scalar(select(TrainingResultRow).where(TrainingResultRow.id == result_id).with_for_update())
    if row is None:
        raise HTTPException(404, "Результат обучения не найден.")
    if row.archived_at is not None:
        return
    pseudo_job_ids = set(session.scalars(select(PseudoMarkupResultRow.job_id).where(
        PseudoMarkupResultRow.training_result_id == row.id,
    )))
    for job in session.scalars(select(JobRow).where(JobRow.status.in_({"queued", "running", "paused"}))):
        if job.id == row.job_id or job.id in pseudo_job_ids or _references(job.config, str(row.id)):
            raise HTTPException(409, "Сеть используется активным заданием. Дождитесь его завершения или остановите его штатно.")
    if session.scalar(select(TestSampleBatchItemRow.id).where(
        TestSampleBatchItemRow.training_result_id == row.id,
        TestSampleBatchItemRow.status.in_({"queued", "running"}),
    ).limit(1)) is not None:
        raise HTTPException(409, "Сеть используется при создании тестовой разметки. Дождитесь завершения.")
    if row.status not in {"ok", "error", "cancelled"}:
        raise HTTPException(409, "Сначала завершите или остановите обучение.")
    session.execute(update(DatasetClassRow).where(
        DatasetClassRow.primary_training_result_id == row.id,
    ).values(primary_training_result_id=None))
    row.archived_at = datetime.now(timezone.utc)
    row.archived_by = username
    session.flush()
