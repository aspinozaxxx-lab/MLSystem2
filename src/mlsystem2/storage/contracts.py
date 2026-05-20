"""Публичные контракты хранилища."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class StorageError(RuntimeError):
    """Ошибка операции хранилища."""


class ResolvedUri(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    scheme: Literal["local", "s3"]
    path: str | None = None
    bucket: str | None = None
    key: str | None = None


class StorageObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    name: str
    size_bytes: int | None = None
    modified_at: str | None = None


@runtime_checkable
class StorageBackend(Protocol):
    def exists(self, uri: str) -> bool: ...

    def list_files(
        self,
        uri: str,
        *,
        suffixes: tuple[str, ...] | None = None,
    ) -> list[StorageObject]: ...

    def read_json(self, uri: str) -> dict[str, object]: ...

    def write_json(self, uri: str, payload: dict[str, object]) -> None: ...


__all__ = [
    "ResolvedUri",
    "StorageBackend",
    "StorageError",
    "StorageObject",
]
