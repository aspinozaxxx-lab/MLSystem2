"""Работа с процессами jobs training UI API."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import time

from ._models import JobRow
from .contracts import TrainingUIAPIError


_TERMINATION_GRACE_SECONDS = 5.0
_KILL_WAIT_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.05


def terminate_job_process(row: JobRow) -> None:
    if row.process_pid is None:
        return
    pid = row.process_pid
    if pid <= 1:
        raise TrainingUIAPIError("Некорректный PID задания; остановка отменена.")
    process_group = hasattr(os, "killpg")
    try:
        if process_group:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                process_group = False
        _send_signal(pid, signal.SIGTERM, process_group)
        if not _wait_for_exit(pid, process_group, _TERMINATION_GRACE_SECONDS):
            _send_signal(pid, getattr(signal, "SIGKILL", signal.SIGTERM), process_group)
            if not _wait_for_exit(pid, process_group, _KILL_WAIT_SECONDS):
                raise TrainingUIAPIError(
                    "Процессы задания не завершились после принудительной остановки. "
                    "Задание и его рабочая папка сохранены."
                )
    except OSError as exc:
        raise TrainingUIAPIError("Не удалось остановить процессы задания.") from exc
    row.process_pid = None


def job_process_is_alive(pid: int) -> bool:
    """Лидер может уже завершиться, пока его дочерние процессы продолжают работу."""
    return _target_is_alive(pid, False) or (
        hasattr(os, "killpg") and _target_is_alive(pid, True)
    )


def _send_signal(pid: int, sig: int, process_group: bool) -> None:
    try:
        if process_group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _wait_for_exit(pid: int, process_group: bool, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _target_is_alive(pid, process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))
    return True


def _target_is_alive(pid: int, process_group: bool) -> bool:
    try:
        (os.killpg if process_group else os.kill)(pid, 0)
    except ProcessLookupError:
        return False
    proc = Path("/proc")
    if not proc.is_dir():
        return True
    paths = proc.glob("[0-9]*/stat") if process_group else [proc / str(pid) / "stat"]
    for path in paths:
        try:
            # Имя процесса в скобках может содержать пробелы и закрывающие скобки.
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (not process_group or int(fields[2]) == pid) and fields[0] not in {"Z", "X"}:
            return True
    return False
