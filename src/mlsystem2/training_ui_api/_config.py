"""Конфигурация training UI API из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TrainingUIAPIConfig:
    host: str
    port: int
    database_url: str
    database_schema: str | None
    mlmarkup_root: Path
    images_root: Path
    stored_files_root: Path
    scratch_root: Path
    frontend_dist: Path
    mlflow_tracking_uri: str
    frontend_username: str
    frontend_password: str
    session_secret: str
    session_cookie_name: str
    session_ttl_seconds: int
    secure_cookies: bool
    grafana_url: str
    mlflow_ui_url: str
    minio_ui_url: str
    cors_origin: str | None


def get_config() -> TrainingUIAPIConfig:
    return TrainingUIAPIConfig(
        host=os.getenv("MLSYSTEM2_TRAINING_UI_API_HOST", "0.0.0.0"),
        port=_int_env("MLSYSTEM2_TRAINING_UI_API_PORT", 8091),
        database_url=os.getenv(
            "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
            os.getenv(
                "TRAINING_UI_DATABASE_URL",
                "postgresql+psycopg://mlsystem2_training_ui@localhost:5432/mlsystem2_training_ui",
            ),
        ),
        database_schema=os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui"),
        mlmarkup_root=Path(
            os.getenv(
                "MLSYSTEM2_MLMARKUP_ROOT",
                os.getenv("MLSYSTEM_MLMARKUP_REPO_PATH", "/data/MLMarkup"),
            )
        ),
        images_root=Path(os.getenv("MLSYSTEM2_IMAGES_ROOT", "/data/mlsystem2/prepared_images")),
        stored_files_root=Path(
            os.getenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", "/data/mlsystem2/training-ui/files")
        ),
        scratch_root=Path(
            os.getenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", "/data/mlsystem2/training-ui/tmp")
        ),
        frontend_dist=Path(os.getenv("MLSYSTEM2_TRAINING_UI_FRONTEND_DIST", "/opt/mlsystem2/frontend")),
        mlflow_tracking_uri=os.getenv(
            "MLSYSTEM2_MLFLOW_TRACKING_URI",
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        ).rstrip("/"),
        frontend_username=os.getenv(
            "MLSYSTEM2_TRAINING_UI_USER",
            os.getenv("MLSYSTEM_FRONTEND_USER", "mluser"),
        ),
        frontend_password=os.getenv(
            "MLSYSTEM2_TRAINING_UI_PASSWORD",
            os.getenv("MLSYSTEM_FRONTEND_PASSWORD", ""),
        ),
        session_secret=os.getenv(
            "MLSYSTEM2_TRAINING_UI_SESSION_SECRET",
            os.getenv("MLSYSTEM_FRONTEND_SESSION_SECRET", ""),
        ),
        session_cookie_name=os.getenv(
            "MLSYSTEM2_TRAINING_UI_SESSION_COOKIE_NAME",
            "mlsystem2_training_ui_session",
        ),
        session_ttl_seconds=_int_env("MLSYSTEM2_TRAINING_UI_SESSION_TTL_SECONDS", 28800),
        secure_cookies=_bool_env("MLSYSTEM2_TRAINING_UI_COOKIE_SECURE", False),
        grafana_url=os.getenv("MLSYSTEM2_GRAFANA_URL", os.getenv("FRONTEND_GRAFANA_URL", "/grafana/")),
        mlflow_ui_url=os.getenv(
            "MLSYSTEM2_MLFLOW_UI_URL",
            os.getenv("FRONTEND_MLFLOW_UI_URL", "/mlflow/"),
        ),
        minio_ui_url=os.getenv(
            "MLSYSTEM2_MINIO_UI_URL",
            os.getenv("FRONTEND_MINIO_UI_URL", "/minio/browser/mlsystems/images/"),
        ),
        cors_origin=os.getenv("MLSYSTEM2_TRAINING_UI_CORS_ORIGIN"),
    )
