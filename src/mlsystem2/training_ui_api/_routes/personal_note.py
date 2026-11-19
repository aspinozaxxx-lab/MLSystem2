"""Личная заметка: проверка владельца при каждой операции, включая скачивание."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._models import PersonalNoteAttachmentRow
from mlsystem2.training_ui_api._personal_note import (
    NoteConflict,
    add_attachment,
    attachment_path,
    note_info,
    remove_attachment,
    save_note,
)
from mlsystem2.training_ui_api.contracts import PersonalNoteAttachment, PersonalNoteInfo, PersonalNoteUpdate
from .common import RouteContext


def register_personal_note_routes(app: FastAPI, ctx: RouteContext) -> None:
    def owner(username: str = Depends(ctx.authenticated)) -> None:
        if username.casefold() != "aspinoza":
            raise HTTPException(status_code=403, detail="Нет доступа к личной заметке")

    dependencies = [Depends(owner)]
    prefix = "/api/v1/personal-note"

    @app.get(prefix, response_model=PersonalNoteInfo, dependencies=dependencies)
    def get_note(response: Response, db: Session = Depends(ctx.get_db)) -> PersonalNoteInfo:
        response.headers["Cache-Control"] = "no-store"
        return note_info(db)

    @app.put(prefix, response_model=PersonalNoteInfo, dependencies=dependencies)
    def put_note(
        request: PersonalNoteUpdate, response: Response, db: Session = Depends(ctx.get_db),
    ) -> PersonalNoteInfo:
        response.headers["Cache-Control"] = "no-store"
        try:
            return save_note(db, request)
        except NoteConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        prefix + "/attachments", response_model=PersonalNoteAttachment,
        status_code=201, dependencies=dependencies,
    )
    def upload_attachment(
        file: UploadFile, response: Response, db: Session = Depends(ctx.get_db),
    ) -> PersonalNoteAttachment:
        response.headers["Cache-Control"] = "no-store"
        try:
            return add_attachment(db, ctx.config, file.file, file.filename or "")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def attachment(db: Session, file_id: uuid.UUID) -> PersonalNoteAttachmentRow:
        row = db.get(PersonalNoteAttachmentRow, file_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Вложение не найдено")
        return row

    @app.get(prefix + "/attachments/{file_id}/download", dependencies=dependencies)
    def download_attachment(file_id: uuid.UUID, db: Session = Depends(ctx.get_db)) -> FileResponse:
        row = attachment(db, file_id)
        path = attachment_path(ctx.config, row.id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Файл вложения не найден")
        return FileResponse(
            path, filename=row.original_name, media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.delete(prefix + "/attachments/{file_id}", status_code=204, dependencies=dependencies)
    def delete_attachment(file_id: uuid.UUID, db: Session = Depends(ctx.get_db)) -> Response:
        remove_attachment(db, ctx.config, attachment(db, file_id))
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
