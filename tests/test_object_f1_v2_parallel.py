"""Параллельная проверка сохраняет порядок, пороги и управление из главного потока."""

from threading import Barrier, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from mlsystem2.inference.contracts import ObjectSeparationConfig
from mlsystem2.train import _object_parallel


def _scene():
    truth = np.zeros((32, 32), np.int32)
    truth[4:28, 4:16] = 1
    truth[4:28, 16:28] = 2
    predicted = truth.copy()
    result = SimpleNamespace(instances=predicted, valid_pixels=np.ones_like(truth, bool),
                             probabilities=np.stack([truth > 0, np.zeros_like(truth)]).astype(np.float32))

    def resegment(config):
        predicted[:] = (truth > 0) if config.boundary_threshold == .35 else truth
        return result

    return SimpleNamespace(finish=lambda: result, resegment=resegment), truth


def _inputs():
    scenes = {name: _scene() for name in ("второй", "первый", "третий")}
    boundaries = {name: (np.zeros((32, 32), bool), np.ones((32, 32), bool)) for name in scenes}
    return scenes, boundaries


def test_parallel_matches_serial_and_keeps_report_order(monkeypatch):
    separation = ObjectSeparationConfig(profile="object_f1_v2")
    expected = {threshold: [] for threshold in (.35, .5, .65)}
    monkeypatch.setattr(_object_parallel, "_scene_workers", lambda *args: 1)
    _object_parallel.evaluate_scenes(*_inputs(), separation, expected, 0, None)
    barrier = Barrier(2)
    main_thread = get_ident()
    worker_threads = set()
    original = _object_parallel.evaluate_candidates

    def observed(scene_id, *args, **kwargs):
        worker_threads.add(get_ident())
        if scene_id != "третий":
            barrier.wait(timeout=10)
        return original(scene_id, *args, **kwargs)

    def pause():
        assert get_ident() == main_thread

    monkeypatch.setattr(_object_parallel, "_scene_workers", lambda *args: 2)
    monkeypatch.setattr(_object_parallel, "evaluate_candidates", observed)
    actual = {threshold: [] for threshold in (.35, .5, .65)}
    _object_parallel.evaluate_scenes(*_inputs(), separation, actual, 8, SimpleNamespace(pause_if_requested=pause))
    assert actual == expected
    assert len(worker_threads) == 2 and main_thread not in worker_threads
    assert [row["scene_id"] for row in actual[.5]] == ["второй", "первый", "третий"]


def test_scene_failure_waits_for_workers_before_closing_maps(monkeypatch):
    monkeypatch.setattr(_object_parallel, "_scene_workers", lambda *args: 2)
    scenes, boundaries = _inputs()
    called = []

    def fail(*args, **kwargs):
        called.append(1)
        raise RuntimeError("Ошибка проверки сцены")

    monkeypatch.setattr(_object_parallel, "evaluate_candidates", fail)
    with pytest.raises(RuntimeError, match="Ошибка проверки сцены"):
        _object_parallel.evaluate_scenes(scenes, boundaries, ObjectSeparationConfig(profile="object_f1_v2"),
                                         {.5: []}, 8, None)
    assert called


def test_stop_from_main_thread_is_propagated(monkeypatch):
    monkeypatch.setattr(_object_parallel, "_scene_workers", lambda *args: 2)

    def stop():
        raise RuntimeError("Остановка пользователем")

    with pytest.raises(RuntimeError, match="Остановка пользователем"):
        _object_parallel.evaluate_scenes(*_inputs(), ObjectSeparationConfig(profile="object_f1_v2"),
                                         {.5: []}, 8, SimpleNamespace(pause_if_requested=stop))


def test_workers_respect_memory_loader_and_scene_limits(monkeypatch):
    scenes = {str(i): (None, SimpleNamespace(size=200_000_000)) for i in range(6)}
    monkeypatch.setattr(_object_parallel.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(_object_parallel.Path, "read_text", lambda self: "MemAvailable: 100000000 kB\n")
    assert _object_parallel._scene_workers(scenes, 8) == 2
    assert _object_parallel._scene_workers(scenes, 0) == 1
    monkeypatch.setattr(_object_parallel.Path, "read_text", lambda self: "MemAvailable: 100 kB\n")
    assert _object_parallel._scene_workers(scenes, 8) == 1
    monkeypatch.setattr(_object_parallel.Path, "read_text", lambda self: "")
    assert _object_parallel._scene_workers(scenes, 8) == 1
