"""Восстановление полной исторической разметки управляемых датасетов."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from shapely import normalize
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from sqlalchemy import func, select

from mlsystem2.dataset_preparing.api import footprint_name_for_annotation

from ._automation import ACTIVE_JOB_STATUSES, ensure_automation_control
from ._config import get_config
from ._database import create_session_factory
from ._dataset_editor import (
    _append_managed_source_feature,
    _blob_revision,
    _commit,
    _editor_lock,
    _geojson_crs,
    _git,
    _git_optional,
    _polygonal_geometry,
    _push_with_retry,
    _read_geojson,
    _repo_relative,
    _synchronize_editor_clone,
    _write_geojson_atomic,
)
from ._managed_datasets import (
    SOURCE_MANAGED,
    invalidate_managed_cache,
    managed_dataset_version,
    managed_sources,
)
from ._managed_migration import (
    DEFAULT_TARGETS,
    _git_geojson_payloads,
    _ordinary_editor_dataset_version,
    _promote_existing_results,
)
from ._models import DatasetClassRow, DatasetRow, JobRow
from .contracts import TrainingUIAPIError


_ROLE = "_mlsystem2_role"
_CLASS = "_mlsystem2_class"
_REPAIR_NAMESPACE = uuid.UUID("838ce603-b66d-55ba-b363-b3c9820a73ab")


@dataclass(frozen=True, slots=True)
class ManagedRepairStats:
    """Итог восстановления одного управляемого датасета."""

    target_path: str
    snapshot_commit: str
    scene_count: int
    changed_source_files: int
    created_source_files: int
    created_footprints: int
    positive_features_added: dict[str, int]
    positive_area_added: dict[str, float]
    hard_negative_features: int
    hard_negative_source_files_changed: int
    max_uncovered_positive_area: float
    max_uncovered_hard_negative_area: float


@dataclass(slots=True)
class _SceneRepair:
    payloads: dict[str, dict[str, Any]]
    positive_features_added: dict[str, int] = field(default_factory=dict)
    positive_area_added: dict[str, float] = field(default_factory=dict)
    hard_negative_features: int = 0
    hard_negative_changed_slugs: set[str] = field(default_factory=set)
    max_uncovered_positive_area: float = 0.0
    max_uncovered_hard_negative_area: float = 0.0


@dataclass(slots=True)
class _TargetRepairPlan:
    target: DatasetRow
    sources: list[Any]
    source_payloads: dict[Path, dict[str, Any]]
    auxiliary_payloads: dict[Path, dict[str, Any]]
    stats: ManagedRepairStats


@dataclass(frozen=True, slots=True)
class _CanonicalGeometry:
    geometry: BaseGeometry
    user_properties: dict[str, Any]


def repair_historical_managed_datasets(
    *,
    apply: bool,
    username: str,
    snapshot_commit: str | None = None,
) -> list[ManagedRepairStats]:
    """Проверить либо восстановить два исторических combined-датасета.

    Полный последний снимок удалённых combined-папок считается нижней границей данных:
    отсутствующая положительная геометрия добавляется, а существующие и более поздние
    правки базовых датасетов не удаляются. Hard negative объединяется и одинаково
    записывается во все источники управляемого датасета.
    """

    config = get_config()
    session_factory = create_session_factory(config)
    with session_factory() as session, _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        plans = [
            _prepare_target_repair(
                session,
                config,
                target_path,
                snapshot_commit=snapshot_commit,
            )
            for target_path in DEFAULT_TARGETS
        ]
        stats = [plan.stats for plan in plans]
        if not apply or not any(plan.source_payloads or plan.auxiliary_payloads for plan in plans):
            return stats

        active_jobs = session.scalar(
            select(func.count()).select_from(JobRow).where(JobRow.status.in_(ACTIVE_JOB_STATUSES))
        )
        if active_jobs:
            raise TrainingUIAPIError(
                f"Восстановление нельзя применять при активных заданиях: найдено {active_jobs}."
            )

        automation = ensure_automation_control(session)
        automation_was_enabled = automation.enabled
        automation.enabled = False
        session.commit()

        expected_revisions: dict[PurePosixPath, str | None] = {}
        changed_paths: set[PurePosixPath] = set()
        pushed = False
        try:
            for plan in plans:
                for path, payload in {
                    **plan.source_payloads,
                    **plan.auxiliary_payloads,
                }.items():
                    relative = _repo_relative(config, path)
                    expected_revisions[relative] = _blob_revision(config, "HEAD", relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_geojson_atomic(path, payload)
                    changed_paths.add(relative)
            _git(config, "add", "--", *(path.as_posix() for path in sorted(changed_paths)))
            _commit(
                config,
                "Восстановить разметку управляемых датасетов",
                username,
            )
            commit = _push_with_retry(
                config,
                expected_revisions=expected_revisions,
            )
            pushed = True
        except Exception:
            if not pushed:
                _restore_editor_paths(config, expected_revisions)
                session.rollback()
                automation = ensure_automation_control(session)
                automation.enabled = automation_was_enabled
                session.commit()
            raise

        try:
            promoted_sources: set[str] = set()
            for plan in plans:
                invalidate_managed_cache(config, plan.target.key)
                version, _updated_at = managed_dataset_version(
                    session,
                    plan.target,
                    config.mlmarkup_editor_root,
                )
                _promote_existing_results(
                    session,
                    plan.target.key,
                    version,
                    promote_pseudo=True,
                )
                for source in plan.sources:
                    if source.dataset.key in promoted_sources:
                        continue
                    promoted_sources.add(source.dataset.key)
                    _promote_existing_results(
                        session,
                        source.dataset.key,
                        _ordinary_editor_dataset_version(config, source.dataset),
                        promote_pseudo=False,
                    )
            # Live-каталог обновляется асинхронно. До проверки release-marker автоматизация
            # должна оставаться выключенной, иначе она увидит переходную версию датасета.
            automation = ensure_automation_control(session)
            automation.enabled = False
            session.commit()
        except Exception:
            session.rollback()
            automation = ensure_automation_control(session)
            automation.enabled = False
            session.commit()
            raise

        print(f"MLMarkup commit: {commit}")
        if automation_was_enabled:
            print(
                "Автоматизация оставлена выключенной до завершения деплоя MLMarkup и "
                "проверки release-marker."
            )
        return stats


def _prepare_target_repair(
    session,
    config,
    target_path: str,
    *,
    snapshot_commit: str | None,
) -> _TargetRepairPlan:
    target = _managed_target(session, target_path)
    sources = managed_sources(session, target.id)
    if len(sources) < 2:
        raise TrainingUIAPIError(
            f"У управляемого датасета {target_path} найдено меньше двух источников."
        )
    source_by_slug = {source.relation.object_type_slug: source for source in sources}
    if len(source_by_slug) != len(sources):
        raise TrainingUIAPIError(f"В {target_path} повторяются типы объектов источников.")

    snapshot = _resolve_snapshot_commit(config, target_path, snapshot_commit)
    historical_payloads = _git_geojson_payloads(config, snapshot, target_path)
    if not historical_payloads:
        raise TrainingUIAPIError(
            f"В историческом снимке {snapshot[:8]} нет разметки {target_path}."
        )

    source_payloads: dict[Path, dict[str, Any]] = {}
    auxiliary_payloads: dict[Path, dict[str, Any]] = {}
    positive_counts: dict[str, int] = {slug: 0 for slug in source_by_slug}
    positive_areas: dict[str, float] = {slug: 0.0 for slug in source_by_slug}
    created_source_files = 0
    created_footprints = 0
    hard_negative_features = 0
    hard_negative_changed_files = 0
    max_uncovered_positive = 0.0
    max_uncovered_hard_negative = 0.0

    for annotation_name in sorted(historical_payloads, key=str.casefold):
        historical = historical_payloads[annotation_name]
        current: dict[str, dict[str, Any] | None] = {}
        paths: dict[str, Path] = {}
        for slug, source in source_by_slug.items():
            path = config.mlmarkup_editor_root.joinpath(
                *PurePosixPath(source.dataset.source_path).parts,
                annotation_name,
            )
            paths[slug] = path
            current[slug] = _read_geojson(path) if path.is_file() else None

        repaired = _repair_scene_payloads(
            historical,
            current,
            target_key=target.key,
            annotation_name=annotation_name,
        )
        hard_negative_features += repaired.hard_negative_features
        max_uncovered_positive = max(
            max_uncovered_positive,
            repaired.max_uncovered_positive_area,
        )
        max_uncovered_hard_negative = max(
            max_uncovered_hard_negative,
            repaired.max_uncovered_hard_negative_area,
        )
        for slug, count in repaired.positive_features_added.items():
            positive_counts[slug] += count
        for slug, area in repaired.positive_area_added.items():
            positive_areas[slug] += area

        for slug, payload in repaired.payloads.items():
            previous = current[slug]
            if previous is None and not payload.get("features"):
                continue
            if previous is not None and _json_hash(previous) == _json_hash(payload):
                continue
            path = paths[slug]
            source_payloads[path] = payload
            if previous is None:
                created_source_files += 1
            if slug in repaired.hard_negative_changed_slugs:
                hard_negative_changed_files += 1
            footprint_path = path.parent / footprint_name_for_annotation(annotation_name)
            if not footprint_path.is_file() and footprint_path not in auxiliary_payloads:
                auxiliary_payloads[footprint_path] = _historical_or_source_footprint(
                    config,
                    snapshot,
                    target_path,
                    annotation_name,
                    paths.values(),
                )
                created_footprints += 1

    stats = ManagedRepairStats(
        target_path=target_path,
        snapshot_commit=snapshot,
        scene_count=len(historical_payloads),
        changed_source_files=len(source_payloads),
        created_source_files=created_source_files,
        created_footprints=created_footprints,
        positive_features_added={key: value for key, value in positive_counts.items() if value},
        positive_area_added={key: value for key, value in positive_areas.items() if value > 0},
        hard_negative_features=hard_negative_features,
        hard_negative_source_files_changed=hard_negative_changed_files,
        max_uncovered_positive_area=max_uncovered_positive,
        max_uncovered_hard_negative_area=max_uncovered_hard_negative,
    )
    return _TargetRepairPlan(
        target=target,
        sources=sources,
        source_payloads=source_payloads,
        auxiliary_payloads=auxiliary_payloads,
        stats=stats,
    )


def _repair_scene_payloads(
    historical: dict[str, Any],
    current: dict[str, dict[str, Any] | None],
    *,
    target_key: str,
    annotation_name: str,
) -> _SceneRepair:
    """Вернуть полное, неразрушающее восстановление одной сцены."""

    historical_crs = _geojson_crs(historical)
    working = {
        slug: payload
        if payload is not None
        else {
            "type": "FeatureCollection",
            "crs": historical.get("crs"),
            "_mlsystem2_managed_dataset_key": target_key,
            "features": [],
        }
        for slug, payload in current.items()
    }
    initial_positive_unions = {
        slug: _role_union_in_crs(payload, "positive", historical_crs)
        for slug, payload in working.items()
    }
    result = _SceneRepair(payloads=working)

    historical_positives: list[tuple[str, BaseGeometry]] = []
    for index, feature in enumerate(historical.get("features") or [], start=1):
        properties = feature.get("properties") or {}
        role = str(properties.get(_ROLE) or "positive")
        if role == "hard_negative":
            continue
        if role != "positive":
            raise TrainingUIAPIError(
                f"Неизвестная роль в {annotation_name}, объект {index}: {role}"
            )
        slug = properties.get(_CLASS)
        if not isinstance(slug, str) or slug not in working:
            raise TrainingUIAPIError(
                f"Не найден источник класса {slug!r} для {annotation_name}, объект {index}."
            )
        geometry = _feature_geometry_in_crs(feature, historical_crs, historical_crs)
        if geometry.is_empty or geometry.area <= 0:
            continue
        historical_positives.append((slug, geometry))
        residual = _polygonal_geometry(geometry.difference(initial_positive_unions[slug]))
        if residual.is_empty or residual.area <= _area_tolerance(geometry):
            continue
        origin = _repair_origin(
            "positive",
            target_key,
            annotation_name,
            slug,
            index,
            geometry,
        )
        repair_feature = {
            "type": "Feature",
            "id": str(uuid.uuid5(_REPAIR_NAMESPACE, origin)),
            "properties": {
                **_user_properties(properties),
                _ROLE: "positive",
            },
            "geometry": dict(mapping(residual)),
        }
        working[slug] = _append_managed_source_feature(
            working[slug],
            repair_feature,
            source_crs=historical_crs,
            origin_key=origin,
        )
        result.positive_features_added[slug] = result.positive_features_added.get(slug, 0) + 1
        result.positive_area_added[slug] = result.positive_area_added.get(slug, 0.0) + residual.area

    hard_negatives = _canonical_hard_negatives(
        historical,
        working,
        historical_crs=historical_crs,
    )
    result.hard_negative_features = len(hard_negatives)
    for slug, payload in list(working.items()):
        positive_features = [
            feature
            for feature in payload.get("features") or []
            if str((feature.get("properties") or {}).get(_ROLE) or "positive") != "hard_negative"
        ]
        replaced = {**payload, "features": positive_features}
        for item in hard_negatives:
            origin = _hard_negative_origin(item)
            feature = {
                "type": "Feature",
                "id": str(uuid.uuid5(_REPAIR_NAMESPACE, origin)),
                "properties": {
                    **item.user_properties,
                    _ROLE: "hard_negative",
                    "_mlsystem2_source_origin_key": origin,
                },
                "geometry": dict(mapping(item.geometry)),
            }
            replaced = _append_managed_source_feature(
                replaced,
                feature,
                source_crs=historical_crs,
                origin_key=origin,
            )
        if _hard_negative_hash(payload) != _hard_negative_hash(replaced):
            result.hard_negative_changed_slugs.add(slug)
        working[slug] = replaced

    result.max_uncovered_positive_area = _validate_positive_coverage(
        historical_positives,
        working,
        historical_crs,
        annotation_name,
    )
    result.max_uncovered_hard_negative_area = _validate_hard_negative_coverage(
        hard_negatives,
        working,
        historical_crs,
        annotation_name,
    )
    return result


def _canonical_hard_negatives(
    historical: dict[str, Any],
    working: dict[str, dict[str, Any]],
    *,
    historical_crs: PyprojCRS,
) -> list[_CanonicalGeometry]:
    candidates: list[_CanonicalGeometry] = []

    def append_candidates(payload: dict[str, Any]) -> None:
        source_crs = _geojson_crs(payload)
        for feature in payload.get("features") or []:
            properties = feature.get("properties") or {}
            if str(properties.get(_ROLE) or "positive") != "hard_negative":
                continue
            geometry = _feature_geometry_in_crs(feature, source_crs, historical_crs)
            if geometry.is_empty or geometry.area <= 0:
                continue
            candidate = _CanonicalGeometry(geometry, _user_properties(properties))
            if not any(
                _geometries_equal(candidate.geometry, existing.geometry) for existing in candidates
            ):
                candidates.append(candidate)

    append_candidates(historical)
    for slug in sorted(working, key=str.casefold):
        append_candidates(working[slug])
    return candidates


def _validate_positive_coverage(
    historical: list[tuple[str, BaseGeometry]],
    payloads: dict[str, dict[str, Any]],
    historical_crs: PyprojCRS,
    annotation_name: str,
) -> float:
    unions = {
        slug: _role_union_in_crs(payload, "positive", historical_crs)
        for slug, payload in payloads.items()
    }
    maximum = 0.0
    for slug, geometry in historical:
        uncovered = _polygonal_geometry(geometry.difference(unions[slug])).area
        maximum = max(maximum, uncovered)
        if uncovered > _area_tolerance(geometry):
            raise TrainingUIAPIError(
                f"После восстановления {annotation_name} не покрыта положительная "
                f"геометрия {slug}: {uncovered:.12g}."
            )
    return maximum


def _validate_hard_negative_coverage(
    historical: list[_CanonicalGeometry],
    payloads: dict[str, dict[str, Any]],
    historical_crs: PyprojCRS,
    annotation_name: str,
) -> float:
    if not historical:
        return 0.0
    maximum = 0.0
    for slug, payload in payloads.items():
        union = _role_union_in_crs(payload, "hard_negative", historical_crs)
        for item in historical:
            uncovered = _polygonal_geometry(item.geometry.difference(union)).area
            maximum = max(maximum, uncovered)
            if uncovered > _area_tolerance(item.geometry):
                raise TrainingUIAPIError(
                    f"После восстановления {annotation_name} hard negative не покрыт "
                    f"источником {slug}: {uncovered:.12g}."
                )
    return maximum


def _role_union_in_crs(
    payload: dict[str, Any],
    role: str,
    target_crs: PyprojCRS,
) -> BaseGeometry:
    source_crs = _geojson_crs(payload)
    geometries = []
    for feature in payload.get("features") or []:
        properties = feature.get("properties") or {}
        feature_role = str(properties.get(_ROLE) or "positive")
        if feature_role != role:
            continue
        geometry = _feature_geometry_in_crs(feature, source_crs, target_crs)
        if not geometry.is_empty and geometry.area > 0:
            geometries.append(geometry)
    return (
        _polygonal_geometry(unary_union(geometries))
        if geometries
        else _polygonal_geometry(shape({"type": "Polygon", "coordinates": []}))
    )


def _feature_geometry_in_crs(
    feature: dict[str, Any],
    source_crs: PyprojCRS,
    target_crs: PyprojCRS,
) -> BaseGeometry:
    geometry = _polygonal_geometry(shape(feature.get("geometry")))
    if geometry.is_empty or source_crs == target_crs:
        return geometry
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return _polygonal_geometry(transform_geometry(transformer.transform, geometry))


def _managed_target(session, target_path: str) -> DatasetRow:
    parts = PurePosixPath(target_path).parts
    if len(parts) != 2:
        raise TrainingUIAPIError(f"Ожидался путь класс/датасет: {target_path}")
    row = session.scalar(
        select(DatasetRow)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(
            DatasetClassRow.name == parts[0],
            DatasetRow.name == parts[1],
            DatasetRow.source_type == SOURCE_MANAGED,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if row is None:
        raise TrainingUIAPIError(f"Управляемый датасет не найден: {target_path}")
    return row


def _resolve_snapshot_commit(config, target_path: str, override: str | None) -> str:
    if override:
        resolved = _git(config, "rev-parse", f"{override}^{{commit}}").stdout.strip()
        if _git_optional(config, "cat-file", "-e", f"{resolved}:{target_path}").returncode:
            raise TrainingUIAPIError(f"В Git-снимке {resolved[:8]} отсутствует {target_path}.")
        return resolved
    deletion_commits = _git(
        config,
        "log",
        "--diff-filter=D",
        "--format=%H",
        "--",
        f"{target_path}/.mlsystem2-dataset.json",
    ).stdout.splitlines()
    for deletion_commit in deletion_commits:
        parent = _git(config, "rev-parse", f"{deletion_commit.strip()}^").stdout.strip()
        if _git_optional(config, "cat-file", "-e", f"{parent}:{target_path}").returncode == 0:
            return parent
    raise TrainingUIAPIError(
        f"Не найден последний Git-снимок удалённого combined-датасета {target_path}."
    )


def _historical_or_source_footprint(
    config,
    snapshot: str,
    target_path: str,
    annotation_name: str,
    source_annotation_paths,
) -> dict[str, Any]:
    footprint_name = footprint_name_for_annotation(annotation_name)
    historical = _git_json_optional(
        config,
        snapshot,
        PurePosixPath(target_path) / footprint_name,
    )
    if historical is not None:
        return historical
    for annotation_path in source_annotation_paths:
        footprint = Path(annotation_path).parent / footprint_name
        if footprint.is_file():
            return _read_geojson(footprint)
    raise TrainingUIAPIError(f"Не найден footprint для восстанавливаемой сцены {annotation_name}.")


def _git_json_optional(
    config,
    commit: str,
    relative_path: PurePosixPath,
) -> dict[str, Any] | None:
    result = _git_optional(config, "show", f"{commit}:{relative_path.as_posix()}")
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TrainingUIAPIError(
            f"Некорректный JSON в Git {commit[:8]}:{relative_path.as_posix()}."
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingUIAPIError(f"GeoJSON в Git должен быть объектом: {relative_path.as_posix()}.")
    return payload


def _restore_editor_paths(
    config,
    expected_revisions: dict[PurePosixPath, str | None],
) -> None:
    created = [path for path, revision in expected_revisions.items() if revision is None]
    existing = [path for path, revision in expected_revisions.items() if revision is not None]
    for relative in created:
        config.mlmarkup_editor_root.joinpath(*relative.parts).unlink(missing_ok=True)
    paths = [*created, *existing]
    if paths:
        _git_optional(
            config,
            "restore",
            "--staged",
            "--worktree",
            "--",
            *(path.as_posix() for path in paths),
        )


def _hard_negative_hash(payload: dict[str, Any]) -> str:
    features = [
        feature
        for feature in payload.get("features") or []
        if str((feature.get("properties") or {}).get(_ROLE) or "positive") == "hard_negative"
    ]
    return _json_hash(features)


def _repair_origin(
    kind: str,
    target_key: str,
    annotation_name: str,
    slug: str,
    index: int,
    geometry: BaseGeometry,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "kind": kind,
                "target_key": target_key,
                "annotation": annotation_name,
                "slug": slug,
                "index": index,
                "geometry": mapping(normalize(geometry)),
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"repair:historical:{digest}"


def _hard_negative_origin(item: _CanonicalGeometry) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "geometry": mapping(normalize(item.geometry)),
                "properties": item.user_properties,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"repair:hard-negative:{digest}"


def _geometries_equal(first: BaseGeometry, second: BaseGeometry) -> bool:
    first_bounds = first.bounds
    second_bounds = second.bounds
    scale = max(*(abs(value) for value in (*first_bounds, *second_bounds)), 1.0)
    if any(abs(left - right) > scale * 1e-10 for left, right in zip(first_bounds, second_bounds)):
        return False
    if first.equals(second):
        return True
    difference = _polygonal_geometry(first.symmetric_difference(second)).area
    return difference <= max(_area_tolerance(first), _area_tolerance(second))


def _area_tolerance(geometry: BaseGeometry) -> float:
    return max(geometry.area, 1e-9) * 1e-9


def _user_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in properties.items() if not key.startswith("_mlsystem2_")}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Восстановить потерянную при combined→managed миграции разметку.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Применить изменения; без флага выполняется только проверка.",
    )
    parser.add_argument("--username", default="Aspinoza")
    parser.add_argument("--snapshot-commit")
    args = parser.parse_args()
    stats = repair_historical_managed_datasets(
        apply=args.apply,
        username=args.username,
        snapshot_commit=args.snapshot_commit,
    )
    for item in stats:
        positive = (
            ", ".join(
                f"{slug}: +{count}, {item.positive_area_added.get(slug, 0.0):.3f}"
                for slug, count in item.positive_features_added.items()
            )
            or "нет"
        )
        print(
            f"{item.target_path}: snapshot={item.snapshot_commit[:8]} "
            f"scenes={item.scene_count} files={item.changed_source_files} "
            f"created={item.created_source_files} footprints={item.created_footprints} "
            f"positive=[{positive}] hard_negative={item.hard_negative_features} "
            f"hn_files={item.hard_negative_source_files_changed}"
        )
    if not args.apply:
        print("Dry-run завершён; для применения добавьте --apply.")


if __name__ == "__main__":
    main()
