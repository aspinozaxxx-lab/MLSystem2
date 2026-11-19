"""Текущее состояние личной заметки и файлы в постоянном хранилище UI."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ._config import TrainingUIAPIConfig
from ._models import PersonalNoteAttachmentRow, PersonalNoteRow
from .contracts import PersonalNoteAttachment, PersonalNoteInfo, PersonalNoteUpdate, TrainingUIAPIError


class NoteConflict(TrainingUIAPIError):
    """Заметка изменена в другой вкладке."""


def note_info(db: Session) -> PersonalNoteInfo:
    row = db.get(PersonalNoteRow, 1)
    attachments = db.scalars(
        select(PersonalNoteAttachmentRow).order_by(
            PersonalNoteAttachmentRow.original_name, PersonalNoteAttachmentRow.id,
        )
    ).all()
    return PersonalNoteInfo(
        content=row.content if row else "",
        revision=row.revision if row else 0,
        attachments=[attachment_info(item) for item in attachments],
    )


def save_note(db: Session, request: PersonalNoteUpdate) -> PersonalNoteInfo:
    if request.expected_revision == 0:
        try:
            with db.begin_nested():
                db.add(PersonalNoteRow(id=1, content=request.content, revision=1))
                db.flush()
        except IntegrityError as exc:
            raise NoteConflict("Заметка уже изменена в другой вкладке. Обновите страницу.") from exc
    else:
        result = db.execute(
            update(PersonalNoteRow)
            .where(PersonalNoteRow.id == 1, PersonalNoteRow.revision == request.expected_revision)
            .values(content=request.content, revision=request.expected_revision + 1)
        )
        if result.rowcount != 1:
            raise NoteConflict("Заметка уже изменена в другой вкладке. Обновите страницу.")
    db.flush()
    db.expire_all()
    return note_info(db)


def attachment_info(row: PersonalNoteAttachmentRow) -> PersonalNoteAttachment:
    return PersonalNoteAttachment(
        id=row.id,
        original_name=row.original_name,
        size_bytes=row.size_bytes,
        download_url=f"/api/v1/personal-note/attachments/{row.id}/download",
    )


def attachment_path(config: TrainingUIAPIConfig, file_id: uuid.UUID) -> Path:
    return config.stored_files_root / "personal-note" / str(file_id)


def add_attachment(
    db: Session, config: TrainingUIAPIConfig, stream: BinaryIO, filename: str,
) -> PersonalNoteAttachment:
    # Имя используется только при скачивании, физический путь определяется серверным UUID.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in {".", ".."} or len(name) > 255 or any(ord(c) < 32 for c in name):
        raise ValueError("Недопустимое имя файла")
    file_id = uuid.uuid4()
    path = attachment_path(config, file_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".part")
    try:
        with temporary.open("xb") as destination:
            shutil.copyfileobj(stream, destination, length=1024 * 1024)
        temporary.chmod(0o600)
        temporary.replace(path)
        row = PersonalNoteAttachmentRow(
            id=file_id, original_name=name, size_bytes=path.stat().st_size,
        )
        db.add(row)
        db.commit()
        return attachment_info(row)
    except BaseException:
        db.rollback()
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise


def remove_attachment(
    db: Session, config: TrainingUIAPIConfig, row: PersonalNoteAttachmentRow,
) -> None:
    db.delete(row)
    db.flush()
    attachment_path(config, row.id).unlink(missing_ok=True)
    db.commit()
