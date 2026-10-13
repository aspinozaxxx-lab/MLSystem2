"""Ограниченная параллельная проверка независимых полных сцен object f1 v2."""

import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event
from time import perf_counter

from ._object_v2 import evaluate_candidates
from .contracts import TrainError


_LOGGER = logging.getLogger(__name__)


def _scene_workers(scenes, requested):
    """Оставить запас RAM для модели, дисковых карт и временных массивов watershed."""
    workers = min(4, max(1, requested), max(1, os.cpu_count() or 1), len(scenes))
    try:
        available = next(int(line.split()[1]) * 1024 for line in
                         Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    except (OSError, ValueError, StopIteration):
        return 1
    # Оценка включает худший случай резервных EDT-маркеров и очереди watershed.
    largest = max(truth.size for _, truth in scenes.values())
    per_scene = max(512 * 1024**2, largest * 160)
    return max(1, min(workers, int(available * 0.65) // per_scene))


class _PauseGate:
    """Рабочие потоки ждут между порогами; только главный поток управляет CUDA."""

    def __init__(self):
        self.ready = Event()
        self.ready.set()
        self.cancelled = Event()

    def pause_if_requested(self):
        self.ready.wait()
        if self.cancelled.is_set():
            raise TrainError("Проверка сцен прервана")

    def check(self, controller):
        if controller is None:
            return
        self.ready.clear()
        try:
            controller.pause_if_requested()
        finally:
            self.ready.set()


def evaluate_scenes(scenes, boundary_truths, separation, candidates, requested_workers, pause_controller):
    workers = _scene_workers(scenes, requested_workers)
    started = perf_counter()
    _LOGGER.info("Проверка object_f1_v2 на CPU: снимков=%s, потоков=%s, порогов=%s",
                 len(scenes), workers, len(candidates))
    gate = _PauseGate()

    def evaluate(item):
        scene_id, (accumulator, truth) = item
        gate.pause_if_requested()
        scene_started = perf_counter()
        _LOGGER.info("Проверка снимка %s: начало, пикселей=%s", scene_id, truth.size)
        local = {threshold: [] for threshold in candidates}
        result = accumulator.finish()
        evaluate_candidates(scene_id, accumulator, truth, *boundary_truths[scene_id][:2],
                            separation, local, gate, initial_result=result)
        _LOGGER.info("Проверка снимка %s: завершена за %.1f с", scene_id, perf_counter() - scene_started)
        return local

    completed = {}
    iterator = iter(scenes.items())
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="object-f1-v2") as executor:
        pending = {}
        try:
            gate.check(pause_controller)
            for _ in range(workers):
                item = next(iterator)
                pending[executor.submit(evaluate, item)] = item[0]
            while pending:
                gate.check(pause_controller)
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    scene_id = pending.pop(future)
                    completed[scene_id] = future.result()
                    item = next(iterator, None)
                    if item is not None:
                        pending[executor.submit(evaluate, item)] = item[0]
        except BaseException:
            gate.cancelled.set()
            gate.ready.set()
            for future in pending:
                future.cancel()
            raise
    # Порядок отчёта не зависит от скорости потоков; карты одной сцены не разделяются.
    for scene_id in scenes:
        for threshold in candidates:
            candidates[threshold].extend(completed[scene_id][threshold])
    _LOGGER.info("Проверка object_f1_v2 на CPU завершена за %.1f с", perf_counter() - started)
