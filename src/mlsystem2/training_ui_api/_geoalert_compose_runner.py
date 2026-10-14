"""Изолированный запуск Geoalert Workflow Engine в его собственном окружении."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


PAUSE_REQUEST_FILE = "pause.request"
PAUSED_MARKER_FILE = "paused"
_SCENE_MEMORY_CAP_BYTES = 32 * 1024**3
_SCENE_MEMORY_MIN_BYTES = 512 * 1024**2
_SCENE_BYTES_PER_PIXEL = 66


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-geoalert-compose-runner")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    scene_workers = _requested_scene_workers(spec)

    try:
        from urban import Compose
        if __package__:
            from ._geoalert_notebook import register_notebook_bricks
        else:
            from _geoalert_notebook import register_notebook_bricks

        register_notebook_bricks()
        # Geoalert требует словарь даже для конвейера только с обязательными блоками.
        pipeline = (
            Compose.load(str(spec["pipeline_path"]), enable_blocks={})
            if scene_workers == 1 else None
        )
    except Exception as exc:
        _write_json_atomic(Path(spec["result_path"]), {
            "status": "error", "reports": [], "failures": [],
            "error": f"Не удалось загрузить конвейер Geoalert: {exc}",
        })
        return 1
    compose_root = Path(spec["compose_root"]).resolve()
    compose_root.mkdir(parents=True, exist_ok=True)
    progress_path = Path(spec["progress_path"])
    result_path = Path(spec["result_path"])
    scenes = list(spec.get("scenes") or [])
    initial_reports = list(spec.get("initial_reports") or [])
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    scene_execution = None

    if scene_workers > 1:
        reports, failures, scene_execution = _run_parallel_scenes(
            spec, Compose, compose_root, scenes, initial_reports, progress_path, started,
        )
    else:
        # Исторические задания сохраняют один Compose и прежний порядок вызовов паузы.
        for number, scene in enumerate(scenes, start=1):
            _pause_if_requested(spec)
            report = _process_scene(
                spec, pipeline, compose_root, scene, number, len(initial_reports),
                tolerant_telemetry=spec.get("pipeline_variant") == "object_f1_v2",
            )
            reports.append(report)
            if report["status"] == "failed":
                failures.append(_scene_failure(report))
            _write_progress(spec, progress_path, scenes, initial_reports, reports, failures, started)

    result = {
        "status": "ok" if not failures else "partial" if reports else "error",
        "reports": reports,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if scene_execution is not None:
        result["scene_execution"] = scene_execution
    _write_json_atomic(result_path, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 1


def _requested_scene_workers(spec: dict[str, Any]) -> int:
    if spec.get("pipeline_variant") != "object_f1_v2":
        return 1
    try:
        return min(2, max(1, int(spec.get("scene_workers") or 1)))
    except (TypeError, ValueError):
        return 1


def _scene_memory_estimate(scene: dict[str, Any]) -> int | None:
    """Читать только размер растра: пиксели TIFF здесь не декодируются."""
    try:
        import rasterio

        with rasterio.open(scene["image_path"]) as image:
            pixels = int(image.width) * int(image.height)
        if pixels <= 0:
            return None
        return max(_SCENE_MEMORY_MIN_BYTES, pixels * _SCENE_BYTES_PER_PIXEL)
    except (OSError, ValueError, KeyError, ImportError):
        return None


def _scene_memory_budget() -> int | None:
    """Оставить 35% свободной RAM и не занять параллельными сценами больше 32 ГиБ."""
    try:
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("MemAvailable:")
        )
        if available <= 0:
            return None
        return min(_SCENE_MEMORY_CAP_BYTES, available * 65 // 100)
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _pause_requested(spec: dict[str, Any]) -> bool:
    raw_control_dir = str(spec.get("control_dir") or "").strip()
    if not raw_control_dir:
        return False
    try:
        return bool((Path(raw_control_dir) / PAUSE_REQUEST_FILE).read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return False


def _run_parallel_scenes(
    spec, compose_type, compose_root, scenes, initial_reports, progress_path, started,
):
    """Две независимые Compose/HTTP-сессии; управление Triton остаётся в главном потоке."""
    # ComposeLoader меняет глобальный yaml.SafeLoader: сериализуем только загрузку.
    load_lock = threading.Lock()
    completed: dict[int, dict[str, Any]] = {}
    pending: dict[Any, tuple[int, int | None]] = {}
    next_index = 0
    execution_lock = threading.Lock()
    active_scenes = 0
    active_pipelines = 0
    overlap_started = None
    overlap_elapsed = 0.0
    timeline_start = time.perf_counter()
    pipeline_timeline = []
    execution = {
        "requested_workers": _requested_scene_workers(spec),
        "max_in_flight": 0,
        "max_active_pipelines": 0,
        "overlapped_scene_pairs": 0,
        "memory_cap_bytes": _SCENE_MEMORY_CAP_BYTES,
    }

    def run_scene(number, scene):
        nonlocal active_scenes

        def pipeline(directory):
            nonlocal active_pipelines, overlap_started, overlap_elapsed
            # Клиент gevent создаётся, вызывается и освобождается в одном OS-потоке.
            with load_lock:
                local_pipeline = compose_type.load(str(spec["pipeline_path"]), enable_blocks={})
            pipeline_started = time.perf_counter()
            with execution_lock:
                active_pipelines += 1
                execution["max_active_pipelines"] = max(
                    execution["max_active_pipelines"], active_pipelines,
                )
                if active_pipelines == 2:
                    execution["overlapped_scene_pairs"] += 1
                    overlap_started = time.perf_counter()
            try:
                local_pipeline(directory)
            finally:
                pipeline_finished = time.perf_counter()
                with execution_lock:
                    if active_pipelines == 2:
                        overlap_elapsed += time.perf_counter() - overlap_started
                        overlap_started = None
                    active_pipelines -= 1
                    pipeline_timeline.append({
                        "number": number + len(initial_reports), "scene_id": str(scene["scene_id"]),
                        "worker_thread": threading.current_thread().name,
                        "started_sec": round(pipeline_started - timeline_start, 6),
                        "finished_sec": round(pipeline_finished - timeline_start, 6),
                        "elapsed_sec": round(pipeline_finished - pipeline_started, 6),
                    })

        with execution_lock:
            active_scenes += 1
            execution["max_in_flight"] = max(execution["max_in_flight"], active_scenes)
        try:
            return _process_scene(
                spec, pipeline, compose_root, scene, number, len(initial_reports),
                tolerant_telemetry=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed_scene_report(scene, number, len(initial_reports), exc)
        finally:
            with execution_lock:
                active_scenes -= 1

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="object-v2-scene") as executor:
        while next_index < len(scenes) or pending:
            pause_requested = _pause_requested(spec)
            if pause_requested and not pending:
                _pause_if_requested(spec)
                continue
            while next_index < len(scenes) and len(pending) < 2 and not pause_requested:
                estimate = _scene_memory_estimate(scenes[next_index])
                if pending:
                    budget = _scene_memory_budget()
                    estimates = [item[1] for item in pending.values()]
                    if (
                        budget is None or estimate is None or None in estimates
                        or sum(estimates) + estimate > budget
                    ):
                        break
                # Одна сцена запускается и при неизвестной/большой оценке: прежний fallback.
                if _pause_requested(spec):
                    break
                number = next_index + 1
                future = executor.submit(run_scene, number, scenes[next_index])
                pending[future] = (next_index, estimate)
                next_index += 1
                pause_requested = _pause_requested(spec)
            if not pending:
                continue
            done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                index, _estimate = pending.pop(future)
                try:
                    completed[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    completed[index] = _failed_scene_report(
                        scenes[index], index + 1, len(initial_reports), exc,
                    )
            if done:
                reports = [completed[index] for index in sorted(completed)]
                failures = [_scene_failure(item) for item in reports if item["status"] == "failed"]
                _write_progress(spec, progress_path, scenes, initial_reports, reports, failures, started)
        if _pause_requested(spec):
            _pause_if_requested(spec)
    reports = [completed[index] for index in range(len(scenes))]
    execution["pipeline_overlap_sec"] = round(overlap_elapsed, 6)
    execution["effective_workers"] = execution["max_in_flight"]
    execution["pipeline_timeline"] = sorted(pipeline_timeline, key=lambda item: item["number"])
    return reports, [_scene_failure(item) for item in reports if item["status"] == "failed"], execution


def _process_scene(
    spec, pipeline, compose_root, scene, number, initial_count, *, tolerant_telemetry=False,
):
    scene_id = str(scene["scene_id"])
    image_path = Path(scene["image_path"]).resolve()
    workdir = compose_root / _safe_workdir_name(scene_id, number)
    _reset_owned_directory(compose_root, workdir)
    (workdir / "input.tif").symlink_to(image_path)
    scene_started = time.time()
    report = {
        "scene_id": scene_id,
        "request_scene": str(scene.get("request_scene") or scene_id),
        "request_scenes": list(scene.get("request_scenes") or [scene_id]),
        "request_scene_count": int(scene.get("request_scene_count") or 1),
        "number": number + initial_count,
        "image": str(image_path),
    }
    try:
        pipeline(str(workdir))
        outputs: dict[str, str] = {}
        feature_count = 0
        for output in spec.get("outputs") or []:
            key = str(output["key"])
            path = workdir / str(output["filename"])
            if not path.is_file():
                raise RuntimeError(f"Geoalert не создал обязательный файл {path.name}.")
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            features = payload.get("features") if isinstance(payload, dict) else None
            if not isinstance(features, list):
                raise RuntimeError(f"Geoalert вернул некорректный FeatureCollection {path.name}.")
            outputs[key] = str(path)
            feature_count += len(features)
        report.update({"status": "ok", "outputs": outputs, "feature_count": feature_count})
    except Exception as exc:  # noqa: BLE001
        report.update({"status": "failed", "outputs": {}, "feature_count": 0, "error": repr(exc)})
    report["elapsed_sec"] = round(time.time() - scene_started, 3)
    performance_path = workdir / "object_inference_performance.json"
    if performance_path.is_file():
        if tolerant_telemetry:
            try:
                report["performance"] = json.loads(performance_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                report["performance_warning"] = f"Не удалось прочитать замер скорости Geoalert: {exc}"
        else:
            report["performance"] = json.loads(performance_path.read_text(encoding="utf-8"))
    return report


def _failed_scene_report(scene, number, initial_count, exc):
    scene_id = str(scene.get("scene_id") or "")
    return {
        "scene_id": scene_id,
        "request_scene": str(scene.get("request_scene") or scene_id),
        "request_scenes": list(scene.get("request_scenes") or [scene_id]),
        "request_scene_count": int(scene.get("request_scene_count") or 1),
        "number": number + initial_count,
        "image": str(Path(scene.get("image_path") or "").resolve()),
        "status": "failed", "outputs": {}, "feature_count": 0,
        "error": f"Не удалось обработать снимок Geoalert: {exc!r}", "elapsed_sec": 0.0,
    }


def _scene_failure(report):
    return {key: report[key] for key in ("scene_id", "image", "error")}


def _write_progress(spec, progress_path, scenes, initial_reports, reports, failures, started):
    _write_json_atomic(progress_path, {
        "current": len(initial_reports) + len(reports),
        "total": len(initial_reports) + len(scenes),
        "processed": sum(item.get("status") == "ok" for item in reports),
        "failed": len(failures),
        "missing": sum(item.get("status") == "missing_image" for item in initial_reports),
        "elapsed_sec": round(time.time() - started, 3),
        "stage": "inference",
        "source_image_ids": list(spec.get("source_image_ids") or []),
        "coverage_percent": spec.get("coverage_percent"),
        "warnings": list(spec.get("warnings") or []),
    })


def _pause_if_requested(spec: dict[str, Any]) -> None:
    raw_control_dir = str(spec.get("control_dir") or "").strip()
    if not raw_control_dir:
        return
    control_dir = Path(raw_control_dir)
    request_path = control_dir / PAUSE_REQUEST_FILE
    if not request_path.is_file():
        return
    pause_token = request_path.read_text(encoding="utf-8").strip()
    if not pause_token:
        return
    model_name = str(spec.get("triton_model_name") or "").strip()
    triton_http_url = str(spec.get("triton_http_url") or "").strip()
    if not model_name or not triton_http_url:
        raise RuntimeError("Для паузы Geoalert не заданы модель Triton и HTTP-адрес.")
    marker_path = control_dir / PAUSED_MARKER_FILE
    control_dir.mkdir(parents=True, exist_ok=True)
    try:
        _triton_repository_action(triton_http_url, model_name, "unload")
        temporary = marker_path.with_suffix(".tmp")
        temporary.write_text(f"{pause_token}\n", encoding="utf-8")
        os.replace(temporary, marker_path)
        while request_path.is_file():
            time.sleep(0.2)
        _triton_repository_action(triton_http_url, model_name, "load")
        _wait_for_triton_model(triton_http_url, model_name)
    finally:
        marker_path.unlink(missing_ok=True)


def _triton_repository_action(http_url: str, model_name: str, action: str) -> None:
    url = (
        f"{http_url.rstrip('/')}/v2/repository/models/"
        f"{urllib_parse.quote(model_name)}/{action}"
    )
    request = urllib_request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Triton вернул HTTP {response.status} для {action}.")
    except urllib_error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Triton не выполнил {action} модели {model_name}: {details}"
        ) from exc


def _wait_for_triton_model(http_url: str, model_name: str) -> None:
    ready_url = f"{http_url.rstrip('/')}/v2/models/{urllib_parse.quote(model_name)}/ready"
    deadline = time.monotonic() + 120.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib_request.urlopen(ready_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib_error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Triton не перевёл модель {model_name} в READY: {last_error!r}")


def _safe_workdir_name(scene_id: str, number: int) -> str:
    slug = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", scene_id).strip("_")[:80] or "scene"
    digest = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:10]
    return f"{number:05d}_{slug}_{digest}"


def _reset_owned_directory(root: Path, target: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise RuntimeError("Рабочая папка Geoalert вышла за пределы каталога задания.")
    if resolved_target.exists():
        shutil.rmtree(resolved_target)
    resolved_target.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
