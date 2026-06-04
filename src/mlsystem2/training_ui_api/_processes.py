"""Работа с процессами jobs training UI API."""

from __future__ import annotations

import os
import signal

from ._models import JobRow


def terminate_job_process(row: JobRow) -> None:
    if row.process_pid is None:
        return
    pid = row.process_pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (AttributeError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    row.process_pid = None
