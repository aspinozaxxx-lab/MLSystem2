"""Остановка всей группы задания до удаления его данных."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mlsystem2.training_ui_api import _processes, _service
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError


def test_termination_failure_preserves_job_and_directory(tmp_path, monkeypatch):
    row = SimpleNamespace(process_pid=12345, tmp_path=str(tmp_path))
    marker = tmp_path / "checkpoint.pt"
    marker.write_bytes(b"checkpoint")
    monkeypatch.setattr(_processes, "_send_signal", lambda *args: None)
    monkeypatch.setattr(_processes, "_wait_for_exit", lambda *args: False)
    if hasattr(os, "killpg"):
        monkeypatch.setattr(_processes.os, "killpg", lambda *args: None)
    with pytest.raises(TrainingUIAPIError, match="не завершились"):
        _service._stop_process_and_cleanup(row)
    assert row.process_pid == 12345
    assert row.tmp_path == str(tmp_path)
    assert marker.read_bytes() == b"checkpoint"


@pytest.mark.skipif(not Path('/proc').is_dir() or not hasattr(os, 'killpg'), reason="Проверка Linux-группы процессов")
@pytest.mark.parametrize("parent_exits_first", [False, True])
def test_cancel_kills_ignoring_child_even_after_group_leader_exits(tmp_path, monkeypatch, parent_exits_first):
    monkeypatch.setattr(_processes, '_TERMINATION_GRACE_SECONDS', 0.2)
    child_file = tmp_path / 'child.pid'
    child_source = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.05)\n"
    )
    parent_source = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
        "while True: time.sleep(0.05)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, '-c', parent_source, child_source, str(child_file)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not child_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_file.is_file(), 'Дочерний процесс не запустился'
        child_pid = int(child_file.read_text())
        assert os.getpgid(child_pid) == parent.pid
        if parent_exits_first:
            parent.terminate()
            parent.wait(timeout=5)
        assert _processes.job_process_is_alive(parent.pid)
        row = SimpleNamespace(process_pid=parent.pid, tmp_path=str(tmp_path))
        _service._stop_process_and_cleanup(row)
        assert not _processes._target_is_alive(child_pid, False)
        assert not _processes._target_is_alive(parent.pid, True)
        assert not _processes.job_process_is_alive(parent.pid)
        assert row.process_pid is None and row.tmp_path is None
        assert not tmp_path.exists()
    finally:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=5)
