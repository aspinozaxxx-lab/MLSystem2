"""Сессионная авторизация UI по тем же env vars, что и старый frontend."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import HTTPException, Request, Response, status

from ._config import TrainingUIAPIConfig


def verify_credentials(username: str, password: str, config: TrainingUIAPIConfig) -> bool:
    if not config.frontend_password:
        return False
    username_matches = any(
        hmac.compare_digest(username, allowed_username)
        for allowed_username in (
            config.frontend_username,
            *config.frontend_username_aliases,
        )
    )
    password_matches = hmac.compare_digest(password, config.frontend_password)
    return username_matches and password_matches


def login_response(response: Response, username: str, config: TrainingUIAPIConfig) -> None:
    if not config.session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Не задан MLSYSTEM2_TRAINING_UI_SESSION_SECRET",
        )
    payload = {"user": username, "ts": int(time.time())}
    response.set_cookie(
        config.session_cookie_name,
        _encode_cookie(payload, config.session_secret),
        max_age=config.session_ttl_seconds,
        httponly=True,
        secure=config.secure_cookies,
        samesite="lax",
        path="/",
    )


def logout_response(response: Response, config: TrainingUIAPIConfig) -> None:
    response.delete_cookie(config.session_cookie_name, path="/")


def current_user(request: Request, config: TrainingUIAPIConfig) -> str | None:
    value = request.cookies.get(config.session_cookie_name)
    if not value or not config.session_secret:
        return None
    payload = _decode_cookie(value, config.session_secret)
    if payload is None:
        return None
    user = payload.get("user")
    ts = payload.get("ts")
    if not isinstance(user, str) or not isinstance(ts, int):
        return None
    if int(time.time()) - ts > config.session_ttl_seconds:
        return None
    return user


def require_user(request: Request, config: TrainingUIAPIConfig) -> str:
    user = current_user(request, config)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Требуется вход")
    return user


def require_pseudolabel_user(request: Request, config: TrainingUIAPIConfig) -> str:
    """Proverit session cookie ili otdelnyi bearer token QGIS."""

    user = current_user(request, config)
    if user is not None:
        return user
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if (
        config.pseudolabel_api_token
        and scheme.casefold() == "bearer"
        and hmac.compare_digest(token, config.pseudolabel_api_token)
    ):
        return "qgis"
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Требуется авторизация")


def _encode_cookie(payload: dict[str, Any], secret: str) -> str:
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    signature = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def _decode_cookie(value: str, secret: str) -> dict[str, Any] | None:
    try:
        body, signature = value.rsplit(".", 1)
    except ValueError:
        return None
    expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        decoded = base64.urlsafe_b64decode(body.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None

