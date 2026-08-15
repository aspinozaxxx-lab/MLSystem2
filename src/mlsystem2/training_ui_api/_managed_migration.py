"""Одноразовая безопасная миграция исторических combined-папок в управляемые датасеты."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import delete, select
from shapely.geometry import shape

from mlsystem2.dataset_preparing.api import (
    footprint_name_for_annotation,
    is_per_image_footprint_name,
    load_dataset_manifest,
)

from ._config import get_config
from ._automation import ensure_automation_control
from ._database import create_session_factory
from ._managed_datasets import (
    SOURCE_MANAGED,
    managed_dataset_version,
)
from ._models import (
    DatasetClassRow,
    DatasetRow,
    ManagedDatasetSceneRow,
    ManagedDatasetSourceRow,
    PseudoMarkupResultRow,
    TrainingResultRow,
)
from ._datasets import _path_metadata, build_per_image_index, imagery_images_dir
from ._dataset_editor import (
    _append_managed_source_feature,
    _blob_revision,
    _commit,
    _editor_lock,
    _features_by_origin,
    _geojson_crs,
    _git,
    _git_optional,
    _push_with_retry,
    _read_geojson,
    _remove_managed_source_feature,
    _repo_relative,
    _synchronize_editor_clone,
    _tree_object_revision,
    _write_geojson_atomic,
)
from .contracts import TrainingUIAPIError


DEFAULT_TARGETS = (
    "Переувлажнения и заболачивания/main",
    "Опустынивание и ветровая эрозия/main",
)


@dataclass(frozen=True, slots=True)
class MigrationStats:
    target_path: str
    baseline_commit: str
    added: int
    edited: int
    deleted: int
    changed_source_files: int


def migrate_combined_datasets(*, apply: bool, username: str) -> list[MigrationStats]:
    config = get_config()
    session_factory = create_session_factory(config)
    with session_factory() as session, _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        image_indexes: dict[Path, dict[str, list[Path]]] = {}
        plans = [
            _prepare_target_migration(
                session,
                config,
                target,
                image_indexes=image_indexes,
            )
            for target in DEFAULT_TARGETS
        ]
        stats = [item["stats"] for item in plans]
        if not apply:
            return stats

        automation = ensure_automation_control(session)
        automation_was_enabled = automation.enabled
        if automation_was_enabled:
            automation.enabled = False
            session.commit()

        expected_revisions: dict[PurePosixPath, str | None] = {}
        expected_trees: dict[PurePosixPath, str | None] = {}
        changed_paths: set[PurePosixPath] = set()
        target_paths: list[PurePosixPath] = []
        try:
            for plan in plans:
                payloads = {
                    **plan["source_payloads"],
                    **plan["auxiliary_payloads"],
                }
                for path, payload in payloads.items():
                    relative = _repo_relative(config, path)
                    expected_revisions[relative] = _blob_revision(config, "HEAD", relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_geojson_atomic(path, payload)
                    changed_paths.add(relative)
                target_relative = PurePosixPath(plan["target_path"])
                expected_trees[target_relative] = _tree_object_revision(
                    config,
                    "HEAD",
                    target_relative,
                )
                target_paths.append(target_relative)
            if changed_paths:
                _git(config, "add", "--", *(path.as_posix() for path in sorted(changed_paths)))
            _git(config, "rm", "-r", "--", *(path.as_posix() for path in target_paths))
            commit = _commit(
                config,
                "Перевести комбинированные датасеты в управляемые",
                username,
            )
            commit = _push_with_retry(
                config,
                expected_revisions=expected_revisions,
                expected_tree_revisions=expected_trees,
            )
        except Exception:
            for relative, revision in expected_revisions.items():
                if revision is None:
                    config.mlmarkup_editor_root.joinpath(*relative.parts).unlink(missing_ok=True)
            restore = [*changed_paths, *target_paths]
            if restore:
                _git_optional(
                    config,
                    "restore",
                    "--staged",
                    "--worktree",
                    "--",
                    *(path.as_posix() for path in restore),
                )
            session.rollback()
            automation = ensure_automation_control(session)
            automation.enabled = automation_was_enabled
            session.commit()
            raise

        try:
            for plan in plans:
                target: DatasetRow = plan["target_row"]
                session.execute(
                    delete(ManagedDatasetSourceRow).where(
                        ManagedDatasetSourceRow.managed_dataset_id == target.id
                    )
                )
                for source in plan["relations"]:
                    session.add(source)
                session.execute(
                    delete(ManagedDatasetSceneRow).where(
                        ManagedDatasetSceneRow.managed_dataset_id == target.id
                    )
                )
                for scene in plan["scenes"]:
                    session.add(scene)
                target.source_type = SOURCE_MANAGED
                target.source_path = f"managed/{target.key}"
                target.config_revision += 1
                target.legacy_version = False
            session.flush()
            promoted_sources: set[str] = set()
            for plan in plans:
                target = plan["target_row"]
                version, _updated_at = managed_dataset_version(
                    session,
                    target,
                    config.mlmarkup_editor_root,
                )
                _promote_existing_results(session, target.key, version, promote_pseudo=True)
                for source in plan["source_rows"].values():
                    if source.key in promoted_sources:
                        continue
                    promoted_sources.add(source.key)
                    _promote_existing_results(
                        session,
                        source.key,
                        _ordinary_editor_dataset_version(config, source),
                        promote_pseudo=False,
                    )
            automation = ensure_automation_control(session)
            automation.enabled = automation_was_enabled
            session.commit()
        except Exception:
            session.rollback()
            automation = ensure_automation_control(session)
            automation.enabled = automation_was_enabled
            session.commit()
            raise
        print(f"MLMarkup commit: {commit}")
        return stats


def _prepare_target_migration(
    session,
    config,
    target_path: str,
    *,
    image_indexes: dict[Path, dict[str, list[Path]]] | None = None,
) -> dict[str, Any]:
    target_dir = config.mlmarkup_editor_root.joinpath(*PurePosixPath(target_path).parts)
    if not target_dir.is_dir():
        raise TrainingUIAPIError(f"Combined-папка не найдена: {target_path}")
    manifest = load_dataset_manifest(target_dir)
    if manifest is None or not manifest.combined:
        raise TrainingUIAPIError(f"Combined-манифест не найден: {target_path}")
    target_row = session.scalar(
        select(DatasetRow).where(
            DatasetRow.source_type == "mlmarkup",
            DatasetRow.source_path == target_path,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if target_row is None:
        raise TrainingUIAPIError(f"Строка датасета не найдена: {target_path}")
    source_rows: dict[str, DatasetRow] = {}
    for source in manifest.sources:
        row = session.scalar(
            select(DatasetRow).where(
                DatasetRow.source_type == "mlmarkup",
                DatasetRow.source_path == source.path,
                DatasetRow.deleted_at.is_(None),
            )
        )
        if row is None:
            raise TrainingUIAPIError(f"Исходный датасет не найден: {source.path}")
        source_rows[source.path] = row

    baseline_commit = _combined_addition_commit(config, target_path)
    baseline_payloads = _git_geojson_payloads(config, baseline_commit, target_path)
    current_payloads = {
        path.name: _read_geojson(path)
        for path in target_dir.glob("*.geojson")
        if not is_per_image_footprint_name(path.name)
    }
    target_class = session.get(DatasetClassRow, target_row.class_id)
    if target_class is None:
        raise TrainingUIAPIError(f"Класс combined-датасета не найден: {target_path}")
    images_root = imagery_images_dir(config.images_root, target_class.imagery_type).resolve()
    if image_indexes is None:
        image_index = build_per_image_index(images_root)
    else:
        image_index = image_indexes.get(images_root)
        if image_index is None:
            image_index = build_per_image_index(images_root)
            image_indexes[images_root] = image_index
    scenes: list[ManagedDatasetSceneRow] = []
    for annotation_name in sorted(current_payloads, key=str.casefold):
        candidates = image_index.get(annotation_name.casefold(), [])
        if len(candidates) != 1:
            raise TrainingUIAPIError(
                f"Для {target_path}/{annotation_name} ожидался ровно один TIFF, найдено: {len(candidates)}"
            )
        scenes.append(
            ManagedDatasetSceneRow(
                managed_dataset_id=target_row.id,
                annotation_name=annotation_name,
                image_relative_path=candidates[0].resolve().relative_to(images_root).as_posix(),
            )
        )
    classes = {item.slug: item for item in manifest.classes}
    source_by_slug = {item.class_slug: item for item in manifest.sources}
    default_source = sorted(
        manifest.sources,
        key=lambda item: (-classes[item.class_slug].priority, item.path),
    )[0]
    source_payloads: dict[Path, dict[str, Any]] = {}
    source_initial_payloads: dict[Path, dict[str, Any] | None] = {}
    auxiliary_payloads: dict[Path, dict[str, Any]] = {}
    auxiliary_owners: dict[Path, Path] = {}
    added = edited = deleted_count = 0

    def payload_for(source_path: str, annotation_name: str, template: dict[str, Any]):
        path = config.mlmarkup_editor_root.joinpath(
            *PurePosixPath(source_path).parts,
            annotation_name,
        )
        if path not in source_payloads:
            if path.is_file():
                source_payloads[path] = _read_geojson(path)
                source_initial_payloads[path] = source_payloads[path]
            else:
                source_payloads[path] = {
                    "type": "FeatureCollection",
                    "crs": template.get("crs"),
                    "_mlsystem2_managed_dataset_key": target_row.key,
                    "features": [],
                }
                source_initial_payloads[path] = None
                footprint = target_dir / footprint_name_for_annotation(annotation_name)
                if footprint.is_file():
                    target_footprint = path.parent / footprint.name
                    if not target_footprint.is_file():
                        auxiliary_payloads[target_footprint] = _read_geojson(footprint)
                        auxiliary_owners[target_footprint] = path
        return path, source_payloads[path]

    for annotation_name in sorted(
        set(baseline_payloads) | set(current_payloads),
        key=str.casefold,
    ):
        baseline = baseline_payloads.get(annotation_name) or {
            "type": "FeatureCollection",
            "crs": (current_payloads.get(annotation_name) or {}).get("crs"),
            "features": [],
        }
        current = current_payloads.get(annotation_name) or {
            "type": "FeatureCollection",
            "crs": baseline.get("crs"),
            "features": [],
        }
        baseline_by_origin = _features_by_origin(baseline)
        current_by_origin = _features_by_origin(current)
        baseline_crs = _geojson_crs(baseline)
        current_crs = _geojson_crs(current)
        for origin in sorted(set(baseline_by_origin) | set(current_by_origin)):
            old = baseline_by_origin.get(origin)
            new = current_by_origin.get(origin)
            if old is not None and new is not None and _migration_features_equal(old, new):
                continue
            if old is None:
                added += 1
            elif new is None:
                deleted_count += 1
            else:
                edited += 1
            old_properties = old.get("properties") or {} if old is not None else {}
            old_role = old_properties.get("_mlsystem2_role", "positive")
            old_slug = old_properties.get("_mlsystem2_class")
            old_source = source_by_slug.get(str(old_slug))
            if old_source is None and old is not None:
                old_source = _source_from_provenance(manifest.sources, old_properties) or default_source
            old_sources = (
                list(manifest.sources)
                if old is not None and old_role == "hard_negative"
                else ([old_source] if old is not None and old_source is not None else [])
            )
            for source in old_sources:
                path, payload = payload_for(source.path, annotation_name, baseline)
                source_payloads[path] = _remove_managed_source_feature(
                    payload,
                    source_identity=old_properties.get("_mlsystem2_source_identity"),
                    source_origin_key=old_properties.get("_mlsystem2_source_origin_key"),
                    fallback_geometry=old.get("geometry"),
                    fallback_crs=baseline_crs,
                    fallback_role=str(old_role),
                )
            if new is not None:
                new_properties = new.get("properties") or {}
                if new_properties.get("_mlsystem2_role", "positive") == "hard_negative":
                    new_sources = list(manifest.sources)
                else:
                    new_slug = new_properties.get("_mlsystem2_class")
                    new_source = source_by_slug.get(str(new_slug))
                    if new_source is None:
                        new_source = (
                            _source_from_provenance(manifest.sources, new_properties)
                            or default_source
                        )
                    new_sources = [new_source]
                for source in new_sources:
                    path, payload = payload_for(source.path, annotation_name, current)
                    source_payloads[path] = _append_managed_source_feature(
                        payload,
                        new,
                        source_crs=current_crs,
                        origin_key=origin,
                    )

    source_payloads = {
        path: payload
        for path, payload in source_payloads.items()
        if (
            source_initial_payloads[path] is None
            and bool(payload.get("features"))
        )
        or (
            source_initial_payloads[path] is not None
            and _file_feature_hash(source_initial_payloads[path])
            != _file_feature_hash(payload)
        )
    }
    auxiliary_payloads = {
        path: payload
        for path, payload in auxiliary_payloads.items()
        if auxiliary_owners[path] in source_payloads
    }

    relations = []
    for definition in sorted(manifest.classes, key=lambda item: item.id):
        source = source_by_slug[definition.slug]
        relations.append(
            ManagedDatasetSourceRow(
                managed_dataset_id=target_row.id,
                source_dataset_id=source_rows[source.path].id,
                priority=definition.priority,
                object_type_id=definition.id,
                object_type_slug=definition.slug,
                object_type_name=definition.name,
                color=definition.color.upper(),
            )
        )
    return {
        "target_path": target_path,
        "target_row": target_row,
        "source_payloads": source_payloads,
        "auxiliary_payloads": auxiliary_payloads,
        "relations": relations,
        "scenes": scenes,
        "source_rows": source_rows,
        "stats": MigrationStats(
            target_path=target_path,
            baseline_commit=baseline_commit,
            added=added,
            edited=edited,
            deleted=deleted_count,
            changed_source_files=len(source_payloads),
        ),
    }


def _combined_addition_commit(config, target_path: str) -> str:
    result = _git(
        config,
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--",
        target_path,
    )
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not commits:
        raise TrainingUIAPIError(f"Не найдена исходная Git-версия {target_path}")
    return commits[-1]


def _git_geojson_payloads(config, commit: str, target_path: str) -> dict[str, dict[str, Any]]:
    result = _git(
        config,
        "-c",
        "core.quotePath=false",
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        target_path,
    )
    output: dict[str, dict[str, Any]] = {}
    for value in result.stdout.splitlines():
        path = PurePosixPath(value.strip())
        if path.suffix.casefold() != ".geojson" or is_per_image_footprint_name(path.name):
            continue
        payload = _git(config, "show", f"{commit}:{path.as_posix()}").stdout
        output[path.name] = json.loads(payload)
    return output


def _source_from_provenance(sources, properties: dict[str, Any]):
    raw = properties.get("_mlsystem2_source_path")
    if not isinstance(raw, str):
        return None
    return next(
        (
            source
            for source in sources
            if raw == source.path or raw.startswith(source.path.rstrip("/") + "/")
        ),
        None,
    )


def _file_feature_hash(feature: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(feature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _migration_features_equal(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    previous_properties = previous.get("properties") or {}
    current_properties = current.get("properties") or {}
    semantic_keys = ("_mlsystem2_role", "_mlsystem2_class")
    if any(previous_properties.get(key) != current_properties.get(key) for key in semantic_keys):
        return False
    previous_user_properties = {
        key: value
        for key, value in previous_properties.items()
        if not key.startswith("_mlsystem2_")
    }
    current_user_properties = {
        key: value
        for key, value in current_properties.items()
        if not key.startswith("_mlsystem2_")
    }
    if previous_user_properties != current_user_properties:
        return False
    try:
        previous_geometry = shape(previous.get("geometry"))
        current_geometry = shape(current.get("geometry"))
        if previous_geometry.equals(current_geometry):
            return True
        difference_area = previous_geometry.symmetric_difference(current_geometry).area
        tolerance = max(previous_geometry.area, current_geometry.area, 1.0) * 1e-12
        return difference_area <= tolerance
    except Exception:  # noqa: BLE001
        return False


def _ordinary_editor_dataset_version(config, dataset: DatasetRow) -> str:
    source_path = config.mlmarkup_editor_root.joinpath(
        *PurePosixPath(dataset.source_path).parts
    )
    _updated_at, source_version = _path_metadata(
        source_path,
        config.mlmarkup_editor_root,
    )
    if dataset.legacy_version:
        return source_version or "missing"
    return f"managed:{dataset.config_revision}:{source_version or 'missing'}"


def _promote_existing_results(
    session,
    dataset_key: str,
    version: str,
    *,
    promote_pseudo: bool,
) -> None:
    rows = session.scalars(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.dataset_key == dataset_key,
            TrainingResultRow.status == "ok",
        )
        .order_by(
            TrainingResultRow.trained_at.desc().nullslast(),
            TrainingResultRow.created_at.desc(),
            TrainingResultRow.id.desc(),
        )
    ).all()
    selected: dict[tuple[str, str, object], TrainingResultRow] = {}
    for row in rows:
        selected.setdefault(
            (row.architecture, row.source, row.automation_rule_id),
            row,
        )
    for row in selected.values():
        row.dataset_version = version
        if not promote_pseudo:
            continue
        pseudo = session.scalar(
            select(PseudoMarkupResultRow)
            .where(
                PseudoMarkupResultRow.training_result_id == row.id,
                PseudoMarkupResultRow.status == "ok",
            )
            .order_by(
                PseudoMarkupResultRow.updated_at.desc(),
                PseudoMarkupResultRow.created_at.desc(),
                PseudoMarkupResultRow.id.desc(),
            )
            .limit(1)
        )
        if pseudo is not None:
            pseudo.dataset_version = version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Миграция двух исторических combined-датасетов в виртуальные управляемые.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Применить изменения; без флага выполняется только dry-run.",
    )
    parser.add_argument("--username", default="Aspinoza")
    args = parser.parse_args()
    stats = migrate_combined_datasets(apply=args.apply, username=args.username)
    for item in stats:
        print(
            f"{item.target_path}: baseline={item.baseline_commit[:8]} "
            f"added={item.added} edited={item.edited} deleted={item.deleted} "
            f"source_files={item.changed_source_files}"
        )
    if not args.apply:
        print("Dry-run завершён; для применения добавьте --apply.")


if __name__ == "__main__":
    main()
