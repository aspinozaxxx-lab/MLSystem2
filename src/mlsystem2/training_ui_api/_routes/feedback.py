"""Приём обращений и управляемое продвижение по этапам."""

import hmac
import re
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._auth import current_principal
from mlsystem2.training_ui_api._models import FeedbackRow
from ..contracts import FeedbackCreate, FeedbackInfo, FeedbackListResponse, FeedbackUpdate
from .common import RouteContext


def register_feedback_routes(app: FastAPI, ctx: RouteContext) -> None:
    def operator(request: Request) -> str:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if ctx.config.feedback_api_token and scheme.lower() == "bearer" and hmac.compare_digest(token.encode("utf-8"), ctx.config.feedback_api_token.encode("utf-8")):
            return "codex"
        principal = current_principal(request, ctx.config)
        if principal is None:
            raise HTTPException(401, "Требуется авторизация")
        if principal.role != "admin":
            raise HTTPException(403, "Изменять этапы может только владелец или исполнитель")
        return principal.username

    def reader(request: Request) -> str:
        if request.headers.get("Authorization"):
            return operator(request)
        return ctx.authenticated(request)

    @app.post("/api/v1/feedback", response_model=FeedbackInfo)
    def create(payload: FeedbackCreate, author: str = Depends(ctx.authenticated), db: Session = Depends(ctx.get_db)):
        existing = select(FeedbackRow).where(FeedbackRow.author == author, FeedbackRow.submission_id == payload.submission_id)
        row = db.scalar(existing)
        if row is not None:
            return row
        marker = ctx.config.project_root / "DEPLOYED_COMMIT"
        version = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        row = FeedbackRow(**payload.model_dump(), author=author, app_version=version if re.fullmatch(r"[a-f0-9]{40}", version) else None)
        try:
            with db.begin_nested():
                db.add(row)
                db.flush()
        except IntegrityError:
            row = db.scalar(existing)
            if row is None:
                raise
        return row

    @app.get("/api/v1/feedback", response_model=FeedbackListResponse)
    def listing(before_id: int | None = Query(default=None, ge=1), limit: int = Query(default=20, ge=1, le=100), active: bool = False, _: str = Depends(reader), db: Session = Depends(ctx.get_db)):
        query = select(FeedbackRow).order_by(FeedbackRow.id.desc())
        if before_id is not None:
            query = query.where(FeedbackRow.id < before_id)
        if active:
            query = query.where(FeedbackRow.status != "implemented")
        rows = list(db.scalars(query.limit(limit + 1)))
        return FeedbackListResponse(items=[FeedbackInfo.model_validate(row) for row in rows[:limit]], has_more=len(rows) > limit)

    @app.get("/api/v1/feedback/{feedback_id}", response_model=FeedbackInfo)
    def detail(feedback_id: int, _: str = Depends(reader), db: Session = Depends(ctx.get_db)):
        row = db.get(FeedbackRow, feedback_id)
        if row is None:
            raise HTTPException(404, "Обращение не найдено")
        return row

    @app.patch("/api/v1/feedback/{feedback_id}", response_model=FeedbackInfo)
    def change(feedback_id: int, payload: FeedbackUpdate, actor: str = Depends(operator), db: Session = Depends(ctx.get_db)):
        row = db.get(FeedbackRow, feedback_id)
        if row is None:
            raise HTTPException(404, "Обращение не найдено")
        if row.revision != payload.expected_revision:
            raise HTTPException(409, "Обращение изменилось: перечитайте актуальную версию")
        if row.status == "implemented":
            raise HTTPException(409, "Реализованное обращение уже завершено")
        values = payload.model_dump(exclude_unset=True, exclude_none=True)
        values.pop("expected_revision")
        new_status = values.get("status", row.status)
        allowed = {"waiting": {"waiting", "preparing"}, "preparing": {"preparing", "implementing"}, "implementing": {"implementing", "implemented"}}
        if new_status not in allowed[row.status]:
            raise HTTPException(409, "Нельзя пропускать этапы обращения")
        if "preparation" in values and (row.status != "preparing" or new_status != "preparing"):
            raise HTTPException(409, "Решение можно уточнять только на этапе подготовки")
        if "approval_note" in values and not (row.status == "preparing" and new_status == "implementing"):
            raise HTTPException(409, "Подтверждение фиксируется при начале реализации")
        if row.status == "preparing" and new_status == "implementing":
            if not row.preparation.strip() or not payload.approval_note:
                raise HTTPException(409, "Нужны подготовленное решение и явное подтверждение владельца")
            values.update(approved_at=datetime.now(timezone.utc), approved_by=actor)
        if "news_slug" in values or "commit_sha" in values:
            if new_status != "implemented":
                raise HTTPException(409, "Результат сохраняется после успешной публикации")
        if new_status == "implemented" and (not payload.news_slug or not payload.commit_sha):
            raise HTTPException(409, "Для завершения нужны новость и коммит установленного обновления")
        if not values:
            raise HTTPException(422, "Нужно указать изменение обращения")
        values.update(revision=row.revision + 1, updated_at=datetime.now(timezone.utc))
        changed = db.execute(update(FeedbackRow).where(FeedbackRow.id == feedback_id, FeedbackRow.revision == payload.expected_revision).values(**values))
        if changed.rowcount != 1:
            raise HTTPException(409, "Обращение изменилось: перечитайте актуальную версию")
        db.expire(row)
        return row


__all__ = ["register_feedback_routes"]
