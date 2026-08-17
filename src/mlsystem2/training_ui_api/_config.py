"""Конфигурация training UI API из переменных окружения."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


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


def _float_env(name: str, default: float) -> float:
    """Прочитать число с плавающей точкой из окружения."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _string_tuple_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Прочитать непустые уникальные значения через запятую."""

    value = os.getenv(name)
    if value is None:
        return default
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


@dataclass(frozen=True, slots=True)
class FrontendUser:
    username: str
    password: str
    role: Literal["admin", "user"]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingUIAPIConfig:
    host: str
    port: int
    project_root: Path
    database_url: str
    database_schema: str | None
    mlmarkup_root: Path
    images_root: Path
    stored_files_root: Path
    scratch_root: Path
    training_settings_path: Path
    frontend_dist: Path
    mlflow_tracking_uri: str
    frontend_username: str
    frontend_username_aliases: tuple[str, ...]
    frontend_password: str
    frontend_users: tuple[FrontendUser, ...]
    session_secret: str
    session_cookie_name: str
    session_ttl_seconds: int
    secure_cookies: bool
    grafana_url: str
    mlflow_ui_url: str
    minio_ui_url: str
    open_webui_url: str
    journal_unit: str
    cors_origin: str | None
    worker_enabled: bool
    worker_interval_seconds: int
    pseudolabel_api_token: str
    pseudolabel_max_aoi_area_m2: float | None
    pseudolabel_max_vertices: int
    pseudolabel_job_timeout_seconds: int
    pseudolabel_imagery_providers_path: Path | None = None
    pseudolabel_image_scan_workers: int = 8
    pseudolabel_tile_read_workers: int = 4
    pseudolabel_prefetch_batches: int = 2
    pseudolabel_external_http_workers: int = 8
    mlmarkup_editor_root: Path = Path("/data/mlsystem2/mlmarkup-editor")
    mlmarkup_release_marker: Path = Path("/data/MLMarkup/.mlsystem2-release")
    mlmarkup_editor_branch: str = "main"
    training_torch_num_threads: int = 4
    training_torch_num_interop_threads: int = 2
    training_process_nice: int = 10
    training_process_io_priority: int = 7
    geoalert_python_path: Path = Path("/opt/geoalert/inference/.venv/bin/python")
    geoalert_inference_root: Path = Path("/opt/geoalert/inference")
    geoalert_model_repository: Path = Path("/opt/geoalert/triton_models")
    geoalert_pipeline_root: Path = Path("/opt/geoalert/pipelines/mlsystem2-runtime")
    geoalert_triton_http_url: str = "http://127.0.0.1:8000"


def get_config() -> TrainingUIAPIConfig:
    frontend_username = os.getenv(
        "MLSYSTEM2_TRAINING_UI_USER",
        os.getenv("MLSYSTEM_FRONTEND_USER", "mlsystem"),
    )
    frontend_username_aliases = _string_tuple_env(
        "MLSYSTEM2_TRAINING_UI_USER_ALIASES",
        ("mluser",),
    )
    frontend_password = os.getenv(
        "MLSYSTEM2_TRAINING_UI_PASSWORD",
        os.getenv("MLSYSTEM_FRONTEND_PASSWORD", ""),
    )
    return TrainingUIAPIConfig(
        host=os.getenv("MLSYSTEM2_TRAINING_UI_API_HOST", "0.0.0.0"),
        port=_int_env("MLSYSTEM2_TRAINING_UI_API_PORT", 8091),
        project_root=Path(os.getenv("MLSYSTEM2_PROJECT_ROOT", os.getcwd())),
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
        training_settings_path=Path(os.getenv("MLSYSTEM2_TRAINING_SETTINGS_PATH", "configs/settings.server.yaml")),
        frontend_dist=Path(os.getenv("MLSYSTEM2_TRAINING_UI_FRONTEND_DIST", "/opt/mlsystem2/frontend")),
        mlflow_tracking_uri=os.getenv(
            "MLSYSTEM2_MLFLOW_TRACKING_URI",
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        ).rstrip("/"),
        frontend_username=frontend_username,
        frontend_username_aliases=frontend_username_aliases,
        frontend_password=frontend_password,
        frontend_users=_frontend_users_env(
            FrontendUser(
                username=frontend_username,
                password=frontend_password,
                role="admin",
                aliases=frontend_username_aliases,
            )
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
        open_webui_url=os.getenv("MLSYSTEM2_OPEN_WEBUI_URL", os.getenv("FRONTEND_OPEN_WEBUI_URL", "/open-webui/")),
        journal_unit=os.getenv("MLSYSTEM2_TRAINING_UI_JOURNAL_UNIT", "mlsystem2-training-ui-api.service"),
        cors_origin=os.getenv("MLSYSTEM2_TRAINING_UI_CORS_ORIGIN"),
        worker_enabled=_bool_env("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", True),
        worker_interval_seconds=_int_env("MLSYSTEM2_TRAINING_UI_WORKER_INTERVAL_SECONDS", 5),
        pseudolabel_api_token=os.getenv("MLSYSTEM2_PSEUDOLABEL_API_TOKEN", ""),
        pseudolabel_max_aoi_area_m2=_optional_positive_float_env(
            "MLSYSTEM2_PSEUDOLABEL_MAX_AOI_AREA_M2",
        ),
        pseudolabel_max_vertices=max(
            4,
            _int_env("MLSYSTEM2_PSEUDOLABEL_MAX_VERTICES", 10_000),
        ),
        pseudolabel_job_timeout_seconds=max(
            1,
            _int_env("MLSYSTEM2_PSEUDOLABEL_JOB_TIMEOUT_SECONDS", 3600),
        ),
        pseudolabel_imagery_providers_path=_optional_path_env(
            "MLSYSTEM2_PSEUDOLABEL_IMAGERY_PROVIDERS_PATH"
        ),
        pseudolabel_image_scan_workers=max(
            1,
            _int_env("MLSYSTEM2_PSEUDOLABEL_IMAGE_SCAN_WORKERS", 8),
        ),
        pseudolabel_tile_read_workers=max(
            1,
            _int_env("MLSYSTEM2_PSEUDOLABEL_TILE_READ_WORKERS", 4),
        ),
        pseudolabel_prefetch_batches=max(
            1,
            _int_env("MLSYSTEM2_PSEUDOLABEL_PREFETCH_BATCHES", 2),
        ),
        pseudolabel_external_http_workers=max(
            1,
            _int_env("MLSYSTEM2_PSEUDOLABEL_EXTERNAL_HTTP_WORKERS", 8),
        ),
        mlmarkup_editor_root=Path(
            os.getenv(
                "MLSYSTEM2_MLMARKUP_EDITOR_ROOT",
                "/data/mlsystem2/mlmarkup-editor",
            )
        ),
        mlmarkup_release_marker=Path(
            os.getenv(
                "MLSYSTEM2_MLMARKUP_RELEASE_MARKER",
                "/data/MLMarkup/.mlsystem2-release",
            )
        ),
        mlmarkup_editor_branch=os.getenv(
            "MLSYSTEM2_MLMARKUP_EDITOR_BRANCH",
            "main",
        ),
        training_torch_num_threads=max(
            1,
            _int_env("MLSYSTEM2_TRAINING_TORCH_NUM_THREADS", 4),
        ),
        training_torch_num_interop_threads=max(
            1,
            _int_env("MLSYSTEM2_TRAINING_TORCH_NUM_INTEROP_THREADS", 2),
        ),
        training_process_nice=min(
            19,
            max(0, _int_env("MLSYSTEM2_TRAINING_PROCESS_NICE", 10)),
        ),
        training_process_io_priority=min(
            7,
            max(0, _int_env("MLSYSTEM2_TRAINING_PROCESS_IO_PRIORITY", 7)),
        ),
        geoalert_python_path=Path(
            os.getenv(
                "MLSYSTEM2_GEOALERT_PYTHON_PATH",
                "/opt/geoalert/inference/.venv/bin/python",
            )
        ),
        geoalert_inference_root=Path(
            os.getenv(
                "MLSYSTEM2_GEOALERT_INFERENCE_ROOT",
                "/opt/geoalert/inference",
            )
        ),
        geoalert_model_repository=Path(
            os.getenv(
                "MLSYSTEM2_GEOALERT_MODEL_REPOSITORY",
                "/opt/geoalert/triton_models",
            )
        ),
        geoalert_pipeline_root=Path(
            os.getenv(
                "MLSYSTEM2_GEOALERT_PIPELINE_ROOT",
                "/opt/geoalert/pipelines/mlsystem2-runtime",
            )
        ),
        geoalert_triton_http_url=os.getenv(
            "MLSYSTEM2_GEOALERT_TRITON_HTTP_URL",
            "http://127.0.0.1:8000",
        ).rstrip("/"),
    )


def _optional_positive_float_env(name: str) -> float | None:
    """Вернуть положительный лимит либо отключить его нулём."""

    value = _float_env(name, 0.0)
    return value if value > 0 else None


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def _frontend_users_env(default: FrontendUser) -> tuple[FrontendUser, ...]:
    raw = os.getenv("MLSYSTEM2_TRAINING_UI_USERS_JSON")
    if raw is None or not raw.strip():
        return (default,)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MLSYSTEM2_TRAINING_UI_USERS_JSON содержит некорректный JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("MLSYSTEM2_TRAINING_UI_USERS_JSON должен быть непустым списком")
    users: list[FrontendUser] = []
    identities: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Пользователь #{index} должен быть объектом")
        username = str(item.get("username") or "").strip()
        password = str(item.get("password") or "")
        role = str(item.get("role") or "user").strip().casefold()
        raw_aliases = item.get("aliases", [])
        if not username or not password:
            raise ValueError(f"У пользователя #{index} обязательны username и password")
        if role not in {"admin", "user"}:
            raise ValueError(f"У пользователя {username} неизвестная роль: {role}")
        if not isinstance(raw_aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in raw_aliases
        ):
            raise ValueError(f"aliases пользователя {username} должен быть списком строк")
        aliases = tuple(dict.fromkeys(alias.strip() for alias in raw_aliases))
        for identity in (username, *aliases):
            key = identity.casefold()
            if key in identities:
                raise ValueError(f"Логин или alias повторяется: {identity}")
            identities.add(key)
        users.append(
            FrontendUser(
                username=username,
                password=password,
                role=cast(Literal["admin", "user"], role),
                aliases=aliases,
            )
        )
    return tuple(users)
