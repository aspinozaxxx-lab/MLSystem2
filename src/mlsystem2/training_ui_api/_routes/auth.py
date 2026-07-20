"""Authentication routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict

from mlsystem2.training_ui_api._auth import (
    current_user,
    login_response,
    logout_response,
    verify_credentials,
)

from .common import RouteContext


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    password: str


def register_auth_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.post("/api/v1/auth/login")
    def login(request: LoginRequest, response: Response) -> dict[str, str]:
        if not verify_credentials(request.username, request.password, ctx.config):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный логин или пароль",
            )
        login_response(response, request.username, ctx.config)
        return {"status": "ok"}

    @app.post("/api/v1/auth/logout")
    def logout(response: Response) -> dict[str, str]:
        logout_response(response, ctx.config)
        return {"status": "ok"}

    @app.get("/api/v1/auth/me")
    def me(request: Request) -> dict[str, str | bool | None]:
        user = current_user(request, ctx.config)
        return {"authenticated": user is not None, "username": user}

    @app.get("/auth/proxy-check", include_in_schema=False)
    def proxy_check(request: Request) -> Response:
        user = current_user(request, ctx.config)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Нужна авторизация")
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"X-Remote-User": user})


__all__ = ["register_auth_routes"]
