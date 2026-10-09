"""Настройки счётчика и идентификатор текущего пользователя."""

from fastapi import Depends, FastAPI, Response

from .._usage import metrica_user_id
from ..contracts import UsageConfig
from .common import RouteContext


def register_usage_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/usage/config", response_model=UsageConfig, summary="Настройки Яндекс Метрики")
    def configuration(response: Response, username: str = Depends(ctx.authenticated)):
        response.headers["Cache-Control"] = "no-store"
        return UsageConfig(metrica_counter_id=ctx.config.usage_metrica_counter_id,
                           metrica_user_id=metrica_user_id(username, ctx.config))
