from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from sqlalchemy import select
from shapely.geometry import box, shape

from mlsystem2.mlflow_adapter.contracts import MLflowBestCheckpoint
from mlsystem2.training_ui_api import _markup_export, _pseudolabel, _pseudo_runner, _worker
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import synchronize_dataset_catalog
from mlsystem2.training_ui_api._models import DatasetClassRow, DatasetRow, JobRow, TrainingResultRow
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api.contracts import (
    JobStatus,
    PseudolabelAPIError,
    PseudolabelJobCreate,
    ResultStatus,
)


# Proveriaet polnyi HTTP i worker flow AOI job.
def test_pseudolabel_http_flow_pins_model_and_stores_result(tmp_path: Path, monkeypatch) -> None:
    config, session_factory = _environment(tmp_path, monkeypatch)
    checkpoint = MLflowBestCheckpoint(
        tracking_uri=config.mlflow_tracking_uri,
        run_id="run-current",
        metric_name="val/best_threshold_pixel_f1",
        f1_score=0.81,
        epoch=4,
        artifact_path="checkpoints/best.pt",
        artifact_uri="s3://models/run-current/checkpoints/best.pt",
        threshold=0.63,
    )
    monkeypatch.setattr(
        _pseudolabel,
        "get_usable_training_checkpoint",
        lambda tracking_uri, run_id: checkpoint if run_id == "run-current" else None,
    )
    class_id, model_id = _seed_model(session_factory, config)
    headers = {"Authorization": "Bearer qgis-test-token"}
    body = {
        "class_id": class_id,
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[30.01, 59.99], [30.05, 59.99], [30.05, 59.95], [30.01, 59.95], [30.01, 59.99]]],
        },
        "aoi_crs": "EPSG:4326",
    }

    with TestClient(create_app()) as client:
        assert client.get("/api/v1/pseudolabel/classes").status_code == 401
        classes = client.get("/api/v1/pseudolabel/classes", headers=headers)
        assert classes.status_code == 200
        assert classes.json()["classes"][0]["model_id"] == str(model_id)

        unsupported = client.post(
            "/api/v1/pseudolabel/jobs",
            content=json.dumps(body),
            headers={**headers, "Content-Type": "text/plain"},
        )
        assert unsupported.status_code == 415
        assert unsupported.json()["error"]["code"] == "UNSUPPORTED_CONTENT_TYPE"

        invalid = client.post(
            "/api/v1/pseudolabel/jobs",
            json={**body, "images": ["client.tif"]},
            headers=headers,
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"

        created = client.post("/api/v1/pseudolabel/jobs", json=body, headers=headers)
        assert created.status_code == 202
        created_payload = created.json()
        assert created_payload["status"] == "queued"
        assert created_payload["model_id"] == str(model_id)
        assert created_payload["model_version"] == "run-current"
        job_id = created_payload["job_id"]
        job_uuid = UUID(job_id)

        with session_factory() as session:
            _worker.dispatch_inference_queue_once(
                session,
                config,
                popen_factory=lambda *args, **kwargs: SimpleNamespace(pid=4321),
            )
            session.flush()
            row = session.get(JobRow, job_uuid)
            assert row is not None
            assert row.status == JobStatus.RUNNING.value
            assert row.tmp_path is not None
            run_dir = Path(row.tmp_path)
            run_config = (run_dir / "pseudolabel_config.yaml").read_text(encoding="utf-8")
            assert "run-current" in run_config
            assert "client.tif" not in run_config
            scratch = run_dir / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            result = {
                "type": "FeatureCollection",
                "metadata": {
                    "job_id": job_id,
                    "class_id": class_id,
                    "model_id": str(model_id),
                    "model_version": "run-current",
                    "source_image_ids": ["region/scene-a"],
                    "coverage_percent": 100.0,
                    "warnings": [],
                },
                "features": [
                    {
                        "type": "Feature",
                        "geometry": body["aoi"],
                        "properties": {
                            "candidate_id": "candidate-1",
                            "class_id": class_id,
                            "model_id": str(model_id),
                            "model_version": "run-current",
                            "job_id": job_id,
                            "source_image_ids": ["region/scene-a"],
                            "area_m2": 1.0,
                        },
                    }
                ],
            }
            (scratch / "pseudo_markup.geojson").write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
            (scratch / "report.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "processed": 1,
                        "feature_count": 1,
                        "source_image_ids": ["region/scene-a"],
                        "coverage_percent": 100.0,
                        "warnings": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "exit_code").write_text("0\n", encoding="utf-8")
            _worker.dispatch_inference_queue_once(
                session,
                config,
                popen_factory=lambda *args, **kwargs: SimpleNamespace(pid=4322),
            )
            session.commit()

        status_response = client.get(f"/api/v1/pseudolabel/jobs/{job_id}", headers=headers)
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "succeeded"
        assert status_response.json()["source_image_ids"] == ["region/scene-a"]
        result_response = client.get(
            f"/api/v1/pseudolabel/jobs/{job_id}/result",
            headers=headers,
        )
        assert result_response.status_code == 200
        assert result_response.json()["features"][0]["properties"]["candidate_id"] == "candidate-1"

        second = client.post("/api/v1/pseudolabel/jobs", json=body, headers=headers)
        cancelled = client.delete(
            f"/api/v1/pseudolabel/jobs/{second.json()['job_id']}",
            headers=headers,
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        failed_created = client.post("/api/v1/pseudolabel/jobs", json=body, headers=headers)
        failed_job_id = UUID(failed_created.json()["job_id"])
        with session_factory() as session:
            _worker.dispatch_inference_queue_once(
                session,
                config,
                popen_factory=lambda *args, **kwargs: SimpleNamespace(pid=4323),
            )
            session.flush()
            failed_row = session.get(JobRow, failed_job_id)
            assert failed_row is not None and failed_row.tmp_path is not None
            failed_run_dir = Path(failed_row.tmp_path)
            failed_scratch = failed_run_dir / "scratch"
            failed_scratch.mkdir(parents=True, exist_ok=True)
            (failed_scratch / "report.json").write_text(
                json.dumps(
                    {
                        "status": "error",
                        "processed": 0,
                        "source_image_ids": ["region/scene-a"],
                        "coverage_percent": 100.0,
                        "warnings": [],
                        "error": {
                            "code": "MODEL_EXECUTION_FAILED",
                            "message": "Модель не смогла обработать снимок.",
                            "details": {},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (failed_run_dir / "exit_code").write_text("1\n", encoding="utf-8")
            _worker.dispatch_inference_queue_once(
                session,
                config,
                popen_factory=lambda *args, **kwargs: SimpleNamespace(pid=4324),
            )
            session.commit()
        failed_status = client.get(
            f"/api/v1/pseudolabel/jobs/{failed_job_id}",
            headers=headers,
        )
        assert failed_status.json()["status"] == "failed"
        assert failed_status.json()["error"]["code"] == "MODEL_EXECUTION_FAILED"


# Proveriaet otbor TIFF, clipping i stabilnye ID.
def test_pseudolabel_spatial_selection_clipping_and_stable_ids(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    _write_raster(images_root / "inside.tif", 30.0, 60.0)
    _write_raster(images_root / "outside.tif", 40.0, 60.0)
    aoi = box(30.01, 59.95, 30.05, 59.99)

    selected = _markup_export.find_intersecting_images(aoi, images_root)

    assert [item.source_id for item in selected.images] == ["inside"]
    assert 99.0 <= selected.coverage_percent <= 100.0
    config = {
        "job_id": "job-1",
        "class_key": "class-1",
        "model_id": "model-1",
        "model_version": "run-1",
    }
    features = [
        {
            "type": "Feature",
            "geometry": box(29.9, 59.9, 30.03, 60.1).__geo_interface__,
            "properties": {"scene_id": "inside", "confidence": 0.87654321},
        }
    ]
    first = _pseudo_runner._finalize_aoi_features(features, aoi, config, ["inside"])
    second = _pseudo_runner._finalize_aoi_features(features, aoi, config, ["inside"])

    assert first == second
    assert len(first) == 1
    assert shape(first[0]["geometry"]).within(aoi)
    assert first[0]["properties"]["candidate_id"]
    assert first[0]["properties"]["source_image_ids"] == ["inside"]
    assert first[0]["properties"]["area_m2"] > 0
    assert first[0]["properties"]["confidence"] == 0.876543
    partial = _markup_export.find_intersecting_images(
        box(30.01, 59.95, 30.08, 59.99),
        images_root,
    )
    assert 0 < partial.coverage_percent < 100
    assert partial.warnings


# Proveriaet domennuyu oshibku pri nulevom pokrytii.
def test_pseudolabel_runner_reports_missing_source_images(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = {
        "operation": "pseudolabel_aoi",
        "job_id": "job-1",
        "run_root": str(run_root),
        "output_geojson": str(run_root / "result.geojson"),
        "images_root": str(tmp_path / "missing-images"),
        "aoi": box(30.0, 59.9, 30.1, 60.0).__geo_interface__,
        "class_key": "class-1",
        "model_id": "model-1",
        "model_version": "run-1",
    }

    report = _pseudo_runner.run_pseudo_markup(config)

    assert report["status"] == "error"
    assert report["error"]["code"] == "SOURCE_IMAGES_NOT_FOUND"
    result = json.loads((run_root / "result.geojson").read_text(encoding="utf-8"))
    assert result["type"] == "FeatureCollection"
    assert result["metadata"]["output_crs"] == "EPSG:4326"
    assert result["metadata"]["object_count"] == 0


# Proveriaet limity, CRS i validnost geometrii.
def test_pseudolabel_rejects_area_and_vertex_limits(tmp_path: Path, monkeypatch) -> None:
    config, _ = _environment(tmp_path, monkeypatch)
    request = PseudolabelJobCreate(
        class_id="class-1",
        aoi={
            "type": "Polygon",
            "coordinates": [[[30.0, 60.0], [30.1, 60.0], [30.1, 59.9], [30.0, 59.9], [30.0, 60.0]]],
        },
        aoi_crs="EPSG:4326",
    )

    with pytest.raises(PseudolabelAPIError) as area_error:
        _pseudolabel._validated_aoi(
            request,
            replace(config, pseudolabel_max_aoi_area_m2=1.0),
        )
    assert area_error.value.code == "AOI_TOO_LARGE"
    unlimited_geometry, unlimited_area, _ = _pseudolabel._validated_aoi(
        request,
        replace(config, pseudolabel_max_aoi_area_m2=None),
    )
    assert not unlimited_geometry.is_empty
    assert unlimited_area > 1.0
    with pytest.raises(PseudolabelAPIError) as vertex_error:
        _pseudolabel._validated_aoi(
            request,
            replace(config, pseudolabel_max_vertices=4),
        )
    assert vertex_error.value.code == "AOI_TOO_MANY_VERTICES"
    unknown_crs = request.model_copy(update={"aoi_crs": "NOT-A-CRS"})
    with pytest.raises(PseudolabelAPIError) as crs_error:
        _pseudolabel._validated_aoi(unknown_crs, config)
    assert crs_error.value.code == "INVALID_CRS"
    invalid_geometry = request.model_copy(
        update={
            "aoi": request.aoi.model_copy(
                update={
                    "coordinates": [
                        [[30.0, 60.0], [30.1, 59.9], [30.1, 60.0], [30.0, 59.9], [30.0, 60.0]]
                    ]
                }
            )
        }
    )
    with pytest.raises(PseudolabelAPIError) as geometry_error:
        _pseudolabel._validated_aoi(invalid_geometry, config)
    assert geometry_error.value.code == "INVALID_GEOMETRY"


# Proveriaet propusk failed run i otsutstvuyushchego artifact.
def test_pseudolabel_does_not_offer_failed_or_unusable_newer_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, session_factory = _environment(tmp_path, monkeypatch)
    class_id, expected_model_id = _seed_model(session_factory, config)
    checkpoint = MLflowBestCheckpoint(
        tracking_uri=config.mlflow_tracking_uri,
        run_id="run-current",
        metric_name="val/best_threshold_pixel_f1",
        f1_score=0.81,
        epoch=4,
        artifact_path="checkpoints/best.pt",
        artifact_uri="s3://models/run-current/checkpoints/best.pt",
        threshold=0.63,
    )
    monkeypatch.setattr(
        _pseudolabel,
        "get_usable_training_checkpoint",
        lambda tracking_uri, run_id: checkpoint if run_id == "run-current" else None,
    )
    with session_factory() as session:
        dataset = session.scalar(select(DatasetRow))
        session.add_all(
            [
                TrainingResultRow(
                    source="manual",
                    dataset_key=dataset.key,
                    class_key=dataset.key,
                    class_display_name="Лес\\main",
                    architecture="smp_segformer_b2",
                    model_name="failed-newer",
                    trained_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
                    mlflow_run_id="run-failed",
                    status=ResultStatus.ERROR.value,
                ),
                TrainingResultRow(
                    source="manual",
                    dataset_key=dataset.key,
                    class_key=dataset.key,
                    class_display_name="Лес\\main",
                    architecture="smp_segformer_b2",
                    model_name="missing-artifact",
                    trained_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
                    mlflow_run_id="run-missing",
                    status=ResultStatus.OK.value,
                ),
            ]
        )
        session.commit()

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/v1/pseudolabel/classes",
            headers={"Authorization": "Bearer qgis-test-token"},
        )

    assert response.status_code == 200
    assert len(response.json()["classes"]) == 1
    selected = response.json()["classes"][0]
    assert selected["class_id"] == class_id
    assert selected["model_id"] == str(expected_model_id)
    assert selected["model_version"] == "run-current"
    assert selected["model_name"] == "segformer b2"


# Proveriaet oshibku klassa bez prigodnoi modeli.
def test_pseudolabel_returns_domain_error_when_class_has_no_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, session_factory = _environment(tmp_path, monkeypatch)
    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        class_id = session.scalar(select(DatasetClassRow.key))
        session.commit()
    body = {
        "class_id": class_id,
        "aoi": box(30.01, 59.95, 30.05, 59.99).__geo_interface__,
        "aoi_crs": "EPSG:4326",
    }
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/pseudolabel/jobs",
            json=body,
            headers={"Authorization": "Bearer qgis-test-token"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "USABLE_MODEL_NOT_FOUND"


# Gotovit izolirovannoe servernoe okruzhenie testa.
def _environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "stored"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    monkeypatch.setenv("MLSYSTEM2_PSEUDOLABEL_API_TOKEN", "qgis-test-token")
    source = tmp_path / "MLMarkup" / "Лес" / "main"
    source.mkdir(parents=True)
    (source / "scenes.txt").write_text("region/scene-a\n", encoding="utf-8")
    (source / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    image = tmp_path / "images" / "kanopus" / "region" / "scene-a.tif"
    image.parent.mkdir(parents=True)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(30.0, 60.0, 0.002, 0.002),
    ) as dataset:
        dataset.write(np.ones((4, 32, 32), dtype=np.uint8))
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    return config, session_factory


# Sozdaet georeferencirovannyi chetyrehkanalnyi TIFF.
def _write_raster(path: Path, west: float, north: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(west, north, 0.002, 0.002),
    ) as dataset:
        dataset.write(np.ones((4, 32, 32), dtype=np.uint8))


# Sozdaet kanonicheskii uspeshnyi training result.
def _seed_model(session_factory, config) -> tuple[str, object]:
    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        class_row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Лес"))
        dataset_row = session.scalar(select(DatasetRow).where(DatasetRow.class_id == class_row.id))
        result = TrainingResultRow(
            source="manual",
            dataset_key=dataset_row.key,
            class_key=dataset_row.key,
            class_display_name="Лес\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            trained_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            mlflow_run_id="run-current",
            status=ResultStatus.OK.value,
        )
        session.add(result)
        session.commit()
        return class_row.key, result.id
