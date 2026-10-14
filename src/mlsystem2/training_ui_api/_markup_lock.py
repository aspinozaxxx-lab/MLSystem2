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


class TestSampleStorageBusy(BlockingIOError):
    """Публикация заняла хранилище, подготовку уже запрошенной оценки нужно отложить."""


@contextmanager
def _file_lock(path: Path, *, write: bool, wait: bool = True) -> Iterator[None]:
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
        if not _WINDOWS_LOCK.acquire(blocking=wait):
            raise TestSampleStorageBusy("Публикуется тестовая разметка. Подготовка расчёта отложена.")
        try:
            with path.open("a+b") as stream:
                import msvcrt

                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if not wait:
                        raise TestSampleStorageBusy("Публикуется тестовая разметка. Подготовка расчёта отложена.") from exc
                    raise
                held[key] = write
                try:
                    yield
                finally:
                    del held[key]
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            _WINDOWS_LOCK.release()
    else:
        with path.open("a+b") as stream:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), (fcntl.LOCK_EX if write else fcntl.LOCK_SH) | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise TestSampleStorageBusy("Публикуется тестовая разметка. Подготовка расчёта отложена.") from exc
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
    if not root.is_dir() or root.resolve() != config.mlmarkup_editor_root.resolve():
        # Архивный атомарный релиз остаётся пригодным для ручного восстановления.
        yield
        return
    with _file_lock(config.mlmarkup_editor_root.parent / ".mlmarkup-editor.lock", write=False):
        yield


@contextmanager
def markup_write_lock(config: TrainingUIAPIConfig) -> Iterator[None]:
    with _file_lock(config.mlmarkup_editor_root.parent / ".mlmarkup-editor.lock", write=True):
        yield


@contextmanager
def test_sample_storage_lock(config: TrainingUIAPIConfig, sample_id: object, *, write: bool = False, wait: bool = True) -> Iterator[None]:
    """Согласовать чтение TIFF и атомарную публикацию тестовой разметки."""
    path = config.stored_files_root / "test-samples" / f".sample-{sample_id}.lock"
    with _file_lock(path, write=write, wait=wait):
        yield
