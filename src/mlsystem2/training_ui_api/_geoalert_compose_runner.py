"""Изолированный запуск Geoalert Workflow Engine в его собственном окружении."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


PAUSE_REQUEST_FILE = "pause.request"
PAUSED_MARKER_FILE = "paused"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-geoalert-compose-runner")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    from urban import Compose

    pipeline = Compose.load(str(spec["pipeline_path"]))
    compose_root = Path(spec["compose_root"]).resolve()
    compose_root.mkdir(parents=True, exist_ok=True)
    progress_path = Path(spec["progress_path"])
    result_path = Path(spec["result_path"])
    scenes = list(spec.get("scenes") or [])
    initial_reports = list(spec.get("initial_reports") or [])
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()

    for number, scene in enumerate(scenes, start=1):
        _pause_if_requested(spec)
        scene_id = str(scene["scene_id"])
        image_path = Path(scene["image_path"]).resolve()
        workdir = compose_root / _safe_workdir_name(scene_id, number)
        _reset_owned_directory(compose_root, workdir)
        input_path = workdir / "input.tif"
        input_path.symlink_to(image_path)
        scene_started = time.time()
        report = {
            "scene_id": scene_id,
            "request_scene": str(scene.get("request_scene") or scene_id),
            "request_scenes": list(scene.get("request_scenes") or [scene_id]),
            "request_scene_count": int(scene.get("request_scene_count") or 1),
            "number": number + len(initial_reports),
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
            report.update(
                {
                    "status": "ok",
                    "outputs": outputs,
                    "feature_count": feature_count,
                    "elapsed_sec": round(time.time() - scene_started, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "scene_id": scene_id,
                "image": str(image_path),
                "error": repr(exc),
            }
            failures.append(failure)
            report.update(
                {
                    "status": "failed",
                    "outputs": {},
                    "feature_count": 0,
                    "error": repr(exc),
                    "elapsed_sec": round(time.time() - scene_started, 3),
                }
            )
        reports.append(report)
        _write_json_atomic(
            progress_path,
            {
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
            },
        )

    result = {
        "status": "ok" if not failures else "partial" if reports else "error",
        "reports": reports,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }
    _write_json_atomic(result_path, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 1


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
