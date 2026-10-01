"""Контракты обращений пользователей Гровики."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

FeedbackKind = Literal["remark", "improvement", "feature"]
FeedbackStatus = Literal["waiting", "preparing", "implementing", "implemented"]


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    submission_id: UUID
    kind: FeedbackKind
    title: str = Field(min_length=3, max_length=160)
    message: str = Field(min_length=5, max_length=6000)
    page_path: str = Field(max_length=1000, pattern=r"^#/[^\s]*$")
    page_title: str = Field(min_length=1, max_length=160)
    credit_name: str | None = Field(default=None, min_length=1, max_length=80)

    @field_validator("page_path")
    @classmethod
    def safe_page_path(cls, value: str) -> str:
        if any(char in value for char in ("?", "\\", "<", ">", '"', "'")):
            raise ValueError("Нужен внутренний адрес страницы без параметров запроса")
        return value


class FeedbackInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: FeedbackKind
    title: str
    message: str
    author: str
    page_path: str
    page_title: str
    app_version: str | None
    credit_name: str | None
    status: FeedbackStatus
    revision: int
    preparation: str
    progress: str
    approved_at: datetime | None
    approved_by: str | None
    approval_note: str | None
    news_slug: str | None
    commit_sha: str | None
    created_at: datetime
    updated_at: datetime


class FeedbackListResponse(BaseModel):
    items: list[FeedbackInfo]
    has_more: bool


class FeedbackUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=1)
    status: FeedbackStatus | None = None
    preparation: str | None = Field(default=None, max_length=12000)
    progress: str | None = Field(default=None, max_length=4000)
    approval_note: str | None = Field(default=None, min_length=5, max_length=4000)
    news_slug: str | None = Field(default=None, max_length=160, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    commit_sha: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")


__all__ = ["FeedbackKind", "FeedbackStatus", "FeedbackCreate", "FeedbackInfo", "FeedbackListResponse", "FeedbackUpdate"]
