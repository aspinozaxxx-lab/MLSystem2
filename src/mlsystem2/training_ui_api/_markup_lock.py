"""Согласованное чтение рабочего клона и сериализация его Git-публикаций."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ._config import TrainingUIAPIConfig


_LOCAL = threading.local()
_WINDOWS_LOCK = threading.RLock()


@contextmanager
def _clone_lock(config: TrainingUIAPIConfig, *, write: bool) -> Iterator[None]:
    path = config.mlmarkup_editor_root.parent / ".mlmarkup-editor.lock"
    key = path.resolve()
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = _LOCAL.held = {}
    if key in held:
        if write and not held[key]:
            raise RuntimeError("Нельзя начать публикацию внутри чтения рабочего клона")
        yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # Windows не поддерживает shared flock; RLock также защищает потоки процесса.
        with _WINDOWS_LOCK, path.open("a+b") as stream:
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            held[key] = write
            try:
                yield
            finally:
                del held[key]
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        with path.open("a+b") as stream:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            held[key] = write
            try:
                yield
            finally:
                del held[key]
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def markup_read_lock(
    config: TrainingUIAPIConfig,
    *,
    source_root: Path | None = None,
) -> Iterator[None]:
    root = source_root if source_root is not None else config.mlmarkup_root
    if root.resolve() != config.mlmarkup_editor_root.resolve():
        # Архивный атомарный релиз остаётся пригодным для ручного восстановления.
        yield
        return
    with _clone_lock(config, write=False):
        yield


@contextmanager
def markup_write_lock(config: TrainingUIAPIConfig) -> Iterator[None]:
    with _clone_lock(config, write=True):
        yield
