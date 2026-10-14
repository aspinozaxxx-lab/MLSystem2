"""Порядок, память и безопасная пауза двух независимых сцен Geoalert v2."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from mlsystem2.training_ui_api import _geoalert_compose_runner as runner
from mlsystem2.training_ui_api import _geoalert_notebook


def _spec(tmp_path, count=3, **kwargs):
    return {
        "pipeline_path": str(tmp_path / "pipeline.yaml"),
        "pipeline_variant": "object_f1_v2",
        "scene_workers": 2,
        "compose_root": str(tmp_path / "compose"),
        "progress_path": str(tmp_path / "progress.json"),
        "result_path": str(tmp_path / "result.json"),
        "scenes": [
            {"scene_id": f"снимок{number}", "image_path": str(tmp_path / f"{number}.tif"),
             "request_scenes": [f"запрос{number}"]}
            for number in range(1, count + 1)
        ],
        "outputs": [{"key": "участки", "filename": "output.geojson"}],
        "initial_reports": [{"status": "missing_image"}, {"status": "missing_image"}],
        "source_image_ids": ["исходный снимок"],
        "coverage_percent": 90,
        "warnings": ["Не все снимки найдены"],
        **kwargs,
    }


def _number(directory):
    return int(Path(directory).name.split("_", 1)[0])


def _output(directory):
    number = _number(directory)
    (Path(directory) / "output.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"properties": {"номер": number}}]}),
        encoding="utf-8",
    )


def _install(monkeypatch, load):
    urban = ModuleType("urban")
    urban.Compose = SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, "urban", urban)
    monkeypatch.setattr(_geoalert_notebook, "register_notebook_bricks", lambda: None)
    monkeypatch.setattr(Path, "symlink_to", lambda path, _target: path.touch())
    monkeypatch.setattr(runner, "_scene_memory_estimate", lambda _scene: 100)
    monkeypatch.setattr(runner, "_scene_memory_budget", lambda: 1000)


def _run(tmp_path, spec):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    code = runner.main(["--spec", str(spec_path)])
    return code, json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))


def test_out_of_order_scenes_keep_original_reports_and_distinct_thread_local_pipelines(
    tmp_path, monkeypatch,
):
    barrier = threading.Barrier(2)
    third_done = threading.Event()
    loaded_threads, finished = [], []
    instances = []
    main_thread = threading.get_ident()

    def load(_path, *, enable_blocks):
        assert enable_blocks == {}
        creator_thread = threading.get_ident()
        loaded_threads.append(creator_thread)
        identity = object()
        instances.append(identity)

        def pipeline(directory):
            assert creator_thread == threading.get_ident()
            number = _number(directory)
            if number < 3:
                barrier.wait(timeout=3)
            if number == 1:
                assert third_done.wait(timeout=3)
            _output(directory)
            finished.append(number)
            if number == 3:
                third_done.set()

        return pipeline

    _install(monkeypatch, load)
    spec = _spec(tmp_path)
    code, result = _run(tmp_path, spec)
    assert code == 0
    assert finished == [2, 3, 1]
    assert len(set(loaded_threads)) == 2
    assert main_thread not in loaded_threads
    assert len(set(instances)) == 3
    reports = result["reports"]
    assert [item["scene_id"] for item in reports] == ["снимок1", "снимок2", "снимок3"]
    assert [item["number"] for item in reports] == [3, 4, 5]
    assert [item["request_scenes"] for item in reports] == [["запрос1"], ["запрос2"], ["запрос3"]]
    assert len({item["outputs"]["участки"] for item in reports}) == 3
    for number, report in enumerate(reports, start=1):
        payload = json.loads(Path(report["outputs"]["участки"]).read_text(encoding="utf-8"))
        assert payload["features"][0]["properties"]["номер"] == number
    assert result["scene_execution"]["max_in_flight"] == 2
    assert result["scene_execution"]["max_active_pipelines"] == 2
    assert result["scene_execution"]["overlapped_scene_pairs"] >= 1
    assert result["scene_execution"]["pipeline_overlap_sec"] > 0
    assert result["scene_execution"]["effective_workers"] == 2
    timeline = result["scene_execution"]["pipeline_timeline"]
    assert [item["number"] for item in timeline] == [3, 4, 5]
    assert [item["scene_id"] for item in timeline] == ["снимок1", "снимок2", "снимок3"]
    assert all(item["elapsed_sec"] >= 0 for item in timeline)
    assert timeline[0]["started_sec"] < timeline[1]["finished_sec"]
    assert timeline[1]["started_sec"] < timeline[0]["finished_sec"]
    progress = json.loads(Path(spec["progress_path"]).read_text(encoding="utf-8"))
    assert (progress["current"], progress["total"], progress["processed"], progress["failed"]) == (5, 5, 3, 0)
    assert progress["missing"] == 2
    assert progress["source_image_ids"] == ["исходный снимок"]


@pytest.mark.parametrize("failure", ["pipeline", "load", "missing_output", "invalid_geojson", "setup"])
def test_failed_scene_does_not_drop_or_repeat_other_scenes(tmp_path, monkeypatch, failure):
    load_count = 0
    calls = []

    def load(*_args, **_kwargs):
        nonlocal load_count
        load_count += 1
        if failure == "load" and load_count == 1:
            raise RuntimeError("Неизвестный блок Geoalert")

        def pipeline(directory):
            number = _number(directory)
            calls.append(number)
            if number == 1 and failure == "pipeline":
                raise RuntimeError("Ошибка обработки снимка")
            if number == 1 and failure == "missing_output":
                return
            if number == 1 and failure == "invalid_geojson":
                (Path(directory) / "output.geojson").write_text("{}", encoding="utf-8")
                return
            _output(directory)

        return pipeline

    _install(monkeypatch, load)
    if failure == "setup":
        def symlink(path, _target):
            if _number(path.parent) == 1:
                raise OSError("Нет доступа к ссылке TIFF")
            path.touch()
        monkeypatch.setattr(Path, "symlink_to", symlink)
    code, result = _run(tmp_path, _spec(tmp_path))
    assert code == 1
    assert result["status"] == "partial"
    assert len(result["reports"]) == 3
    assert [item["number"] for item in result["reports"]] == [3, 4, 5]
    assert sum(item["status"] == "failed" for item in result["reports"]) == 1
    assert sum(item["status"] == "ok" for item in result["reports"]) == 2
    assert len(result["failures"]) == 1
    failed_report = next(item for item in result["reports"] if item["status"] == "failed")
    assert result["failures"][0]["scene_id"] == failed_report["scene_id"]
    assert len(set(calls)) == len(calls)
    assert all(_number(Path(item["outputs"]["участки"]).parent) in calls
               for item in result["reports"] if item["status"] == "ok")


def test_future_failure_is_converted_to_scene_report(tmp_path, monkeypatch):
    class FaultyExecutor(ThreadPoolExecutor):
        def submit(self, fn, number, scene):
            if number == 1:
                future = Future()
                future.set_exception(RuntimeError("Ошибка рабочего потока"))
                return future
            return super().submit(fn, number, scene)

    _install(monkeypatch, lambda *_args, **_kwargs: _output)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", FaultyExecutor)
    code, result = _run(tmp_path, _spec(tmp_path))
    assert code == 1
    assert [item["status"] for item in result["reports"]] == ["failed", "ok", "ok"]
    assert "Ошибка рабочего потока" in result["failures"][0]["error"]


def test_multiple_failures_keep_original_order_despite_completion_order(tmp_path, monkeypatch):
    second_failed = threading.Event()
    def pipeline(directory):
        number = _number(directory)
        if number == 1:
            assert second_failed.wait(timeout=3)
            raise RuntimeError("Ошибка первого снимка")
        if number == 2:
            second_failed.set()
            raise RuntimeError("Ошибка второго снимка")
        _output(directory)

    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    code, result = _run(tmp_path, _spec(tmp_path))
    assert code == 1
    assert [item["scene_id"] for item in result["failures"]] == ["снимок1", "снимок2"]
    assert [item["status"] for item in result["reports"]] == ["failed", "failed", "ok"]
    assert [item["number"] for item in result["reports"]] == [3, 4, 5]


def test_equal_scene_ids_still_have_distinct_numbered_outputs(tmp_path, monkeypatch):
    _install(monkeypatch, lambda *_args, **_kwargs: _output)
    spec = _spec(tmp_path)
    for item in spec["scenes"]:
        item["scene_id"] = "общий идентификатор"
    code, result = _run(tmp_path, spec)
    assert code == 0
    assert len(result["reports"]) == 3
    assert len({item["outputs"]["участки"] for item in result["reports"]}) == 3
    assert [item["number"] for item in result["reports"]] == [3, 4, 5]


@pytest.mark.parametrize("workers", [1, 2])
def test_corrupt_timing_json_is_nonfatal_for_v2(tmp_path, monkeypatch, workers):
    def pipeline(directory):
        _output(directory)
        (Path(directory) / "object_inference_performance.json").write_text("{", encoding="utf-8")

    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    code, result = _run(tmp_path, _spec(tmp_path, scene_workers=workers))
    assert code == 0
    assert all("performance_warning" in item for item in result["reports"])
    assert all(item["status"] == "ok" for item in result["reports"])


def test_legacy_timing_json_failure_retains_original_behavior(tmp_path, monkeypatch):
    def pipeline(directory):
        _output(directory)
        (Path(directory) / "object_inference_performance.json").write_text("{", encoding="utf-8")
    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    with pytest.raises(json.JSONDecodeError):
        _run(tmp_path, _spec(tmp_path, pipeline_variant="legacy"))


@pytest.mark.parametrize("budget,estimate", [(None, 100), (1000, None), (199, 100), (1000, 10000)])
def test_memory_unknown_or_too_large_pair_falls_back_to_one_scene(
    tmp_path, monkeypatch, budget, estimate,
):
    def pipeline(directory):
        time.sleep(.02)
        _output(directory)

    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    monkeypatch.setattr(runner, "_scene_memory_budget", lambda: budget)
    monkeypatch.setattr(runner, "_scene_memory_estimate", lambda _scene: estimate)
    code, result = _run(tmp_path, _spec(tmp_path))
    assert code == 0
    assert result["scene_execution"]["max_in_flight"] == 1
    assert result["scene_execution"]["max_active_pipelines"] == 1
    assert result["scene_execution"]["overlapped_scene_pairs"] == 0
    assert len(result["reports"]) == 3


def test_memory_is_admitted_for_actual_pair_not_global_largest_scene(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    def pipeline(directory):
        if _number(directory) > 1:
            barrier.wait(timeout=3)
        _output(directory)

    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    monkeypatch.setattr(runner, "_scene_memory_estimate", lambda scene: 10000 if scene["scene_id"] == "снимок1" else 100)
    code, result = _run(tmp_path, _spec(tmp_path))
    assert code == 0
    assert result["scene_execution"]["max_active_pipelines"] == 2


def test_pause_drains_both_scenes_before_unload_and_resume_does_not_repeat(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    release = threading.Event()
    control = tmp_path / "control"
    control.mkdir()
    request = control / "pause.request"
    lock = threading.Lock()
    active, calls, actions = [], [], []
    main_thread = threading.get_ident()

    def pipeline(directory):
        number = _number(directory)
        with lock:
            active.append(number)
            calls.append(number)
        if number <= 2:
            barrier.wait(timeout=3)
            if number == 1:
                request.write_text("приостановить\n", encoding="utf-8")
            assert release.wait(timeout=3)
        _output(directory)
        with lock:
            active.remove(number)

    def requested(spec):
        value = original_requested(spec)
        if value:
            release.set()
        return value

    def action(_url, _model, action):
        assert threading.get_ident() == main_thread
        assert active == []
        assert sorted(calls) == [1, 2]
        actions.append(action)
        if action == "unload":
            request.unlink()

    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    original_requested = runner._pause_requested
    monkeypatch.setattr(runner, "_pause_requested", requested)
    monkeypatch.setattr(runner, "_triton_repository_action", action)
    monkeypatch.setattr(runner, "_wait_for_triton_model", lambda *_args: None)
    code, result = _run(tmp_path, _spec(tmp_path, control_dir=str(control),
                                     triton_http_url="http://triton", triton_model_name="model"))
    assert code == 0
    assert actions == ["unload", "load"]
    assert sorted(calls) == [1, 2, 3]
    assert len(set(calls)) == 3
    assert [item["number"] for item in result["reports"]] == [3, 4, 5]
    assert not (control / "paused").exists()


def test_pause_before_start_does_not_create_client_until_resume(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    request = control / "pause.request"
    request.write_text("приостановить\n", encoding="utf-8")
    events = []
    def load(*_args, **_kwargs):
        assert not request.exists()
        events.append("load")
        return _output
    def pause(_spec):
        assert events == []
        events.append("pause")
        request.unlink()

    _install(monkeypatch, load)
    monkeypatch.setattr(runner, "_pause_if_requested", pause)
    code, result = _run(tmp_path, _spec(tmp_path, control_dir=str(control)))
    assert code == 0
    assert events[0] == "pause"
    assert events.count("load") == 3
    assert len(result["reports"]) == 3


def test_pause_during_metadata_admission_prevents_next_submit(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    request = control / "pause.request"
    calls = []
    def estimate(scene):
        if scene["scene_id"] == "снимок2" and not calls:
            request.write_text("приостановить\n", encoding="utf-8")
        return 100
    def pipeline(directory):
        time.sleep(.02)
        _output(directory)
    def pause(_spec):
        calls.append("pause")
        request.unlink()

    _install(monkeypatch, lambda *_args, **_kwargs: pipeline)
    monkeypatch.setattr(runner, "_scene_memory_estimate", estimate)
    monkeypatch.setattr(runner, "_pause_if_requested", pause)
    code, result = _run(tmp_path, _spec(tmp_path, control_dir=str(control)))
    assert code == 0
    assert calls == ["pause"]
    timeline = result["scene_execution"]["pipeline_timeline"]
    assert timeline[1]["started_sec"] >= timeline[0]["finished_sec"]


@pytest.mark.parametrize("variant,workers", [(None, 2), ("legacy", 2), ("object_f1", 2), ("next_gen2", 2), ("object_f1_v2", None), ("object_f1_v2", 1)])
def test_legacy_and_specs_without_scene_workers_keep_exact_sequential_path(
    tmp_path, monkeypatch, variant, workers,
):
    main_thread = threading.get_ident()
    events = []
    def load(*_args, **_kwargs):
        assert threading.get_ident() == main_thread
        events.append("load")
        def pipeline(directory):
            assert threading.get_ident() == main_thread
            events.append(_number(directory))
            _output(directory)
        return pipeline

    _install(monkeypatch, load)
    monkeypatch.setattr(runner, "_pause_if_requested", lambda _spec: events.append("pause"))
    monkeypatch.setattr(runner, "_scene_memory_estimate", lambda *_args: pytest.fail("Лишнее чтение растра в legacy"))
    monkeypatch.setattr(runner, "_scene_memory_budget", lambda: pytest.fail("Лишний admission в legacy"))
    spec = _spec(tmp_path, pipeline_variant=variant, scene_workers=workers)
    if workers is None:
        spec.pop("scene_workers")
    code, result = _run(tmp_path, spec)
    assert code == 0
    assert events == ["load", "pause", 1, "pause", 2, "pause", 3]
    assert "scene_execution" not in result


@pytest.mark.parametrize("requested,expected", [(100, 2), (2, 2), (1, 1), (0, 1), (-1, 1), ("не число", 1)])
def test_parallel_workers_are_capped_at_two(requested, expected):
    assert runner._requested_scene_workers({"pipeline_variant": "object_f1_v2", "scene_workers": requested}) == expected


def test_memory_estimate_reads_metadata_only(monkeypatch):
    image = SimpleNamespace(width=16000, height=12000)
    class Dataset:
        def __enter__(self):
            return image
        def __exit__(self, *_args):
            pass
    rasterio = ModuleType("rasterio")
    rasterio.open = lambda _path: Dataset()
    monkeypatch.setitem(sys.modules, "rasterio", rasterio)
    assert runner._scene_memory_estimate({"image_path": "снимок.tif"}) == 16000 * 12000 * 66
    image.width = image.height = 1
    assert runner._scene_memory_estimate({"image_path": "снимок.tif"}) == 512 * 1024**2


@pytest.mark.parametrize("payload,expected", [
    ("MemAvailable: 104857600 kB\n", 32 * 1024**3),
    ("MemAvailable: 1000 kB\n", 1000 * 1024 * 65 // 100),
    ("MemAvailable: 0 kB\n", None), ("MemAvailable: ошибочное kB\n", None), ("", None),
])
def test_memory_budget_reserves_ram_and_caps_at_32_gib(monkeypatch, payload, expected):
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: payload)
    assert runner._scene_memory_budget() == expected


def test_unavailable_metadata_and_memory_do_not_enable_parallelism(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("Нет данных")
    monkeypatch.setattr(Path, "read_text", unavailable)
    assert runner._scene_memory_budget() is None
    rasterio = ModuleType("rasterio")
    rasterio.open = unavailable
    monkeypatch.setitem(sys.modules, "rasterio", rasterio)
    assert runner._scene_memory_estimate({"image_path": "снимок.tif"}) is None
