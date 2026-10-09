"""Настройки аналитики для текущей авторизованной сессии."""

from pydantic import BaseModel


class UsageConfig(BaseModel):
    metrica_counter_id: int | None
    metrica_user_id: str


__all__ = ["UsageConfig"]
