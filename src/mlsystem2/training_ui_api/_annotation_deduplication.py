"""Разовое объединение одинаковых вариантов через штатный журнал публикаций."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timezone
import json
from pathlib import Path, PurePosixPath
import uuid

from pyproj import CRS
from shapely import normalize, to_wkb
from shapely.geometry import shape
from sqlalchemy import select

from mlsystem2.dataset_preparing.api import is_per_image_footprint_name

from . import _dataset_editor as editor
from ._annotation_versions import AnnotationCatalog, merge_plans, payload_hash, source_path
from ._config import get_config
from ._database import create_session_factory
from ._datasets import build_per_image_index
from ._markup_lock import markup_read_lock
from ._models import (
    AnnotationImageRow, AnnotationPublicationRow, AnnotationVersionRow, DatasetAnnotationRow,
    DatasetClassRow, DatasetEditorDraftRow, DatasetRow, ManagedDatasetSourceRow,
)
from .contracts import AnnotationVersionDraft, TrainingUIAPIError


OPERATION = "deduplicate_annotation_versions_v1"


def geometry_signature(payload):
    """Точное покрытие отдельных полигонов и обучающие роли, без допуска и ремонта."""
    values = []

    def parts(geometry):
        if geometry.geom_type == "Polygon":
            if not geometry.is_empty:
                yield geometry
        elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
            for part in geometry.geoms:
                yield from parts(part)
        else:
            raise ValueError("Неполигональная геометрия")

    for feature in payload.get("features", []):
        if not feature.get("geometry"):
            raise ValueError("Объект без геометрии")
        geometry = shape(feature["geometry"])
        if not geometry.is_valid or geometry.has_z:
            raise ValueError("Невалидная или трёхмерная геометрия")
        properties = feature.get("properties") or {}
        role = properties.get("_mlsystem2_role", "positive")
        class_id = properties.get("_mlsystem2_class")
        for part in parts(geometry):
            reduced = part.simplify(0, preserve_topology=True)
            if reduced.geom_type == "Polygon" and part.equals(reduced):
                part = reduced
            values.append((to_wkb(normalize(part), byte_order=1).hex(), role, str(class_id)))
    crs = payload.get("crs")
    if isinstance(crs, dict):
        crs = crs.get("properties", {}).get("name", crs)
    crs = CRS.from_user_input(crs).to_wkt() if crs else None
    # Эти поля описывают задачу, а не произвольные атрибуты объектов.
    semantics = {"task": payload.get("_mlsystem2_task", "binary")}
    if semantics["task"] == "multiclass":
        semantics.update(schema=payload.get("_mlsystem2_schema_version"), classes=payload.get("_mlsystem2_classes"))
    return payload_hash({"crs": crs, "polygons": sorted(values), "semantics": semantics}), len(values)


def _stamp(value):
    return value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat() if value else "9999"


def inspect_duplicates(session, config):
    """Только чтение: не импортирует файлы в БД и не создаёт версии."""
    rows = {row.id: row for row in session.scalars(select(DatasetRow).where(DatasetRow.deleted_at.is_(None)))}
    classes = {row.id: row for row in session.scalars(select(DatasetClassRow))}
    images = {row.id: row for row in session.scalars(select(AnnotationImageRow))}
    bindings = {(row.dataset_id, row.annotation_name.casefold()): row
                for row in session.scalars(select(DatasetAnnotationRow))}
    records, warnings, files = {}, [], {}
    for row in session.scalars(select(AnnotationVersionRow).where(AnnotationVersionRow.deleted_at.is_(None))):
        image = images[row.image_id]
        records[str(row.id)] = {
            "id": str(row.id), "class_id": str(row.class_id), "class": classes[row.class_id].name,
            "image": image.relative_path, "kind": image.imagery_type, "name": row.name,
            "created_at": _stamp(row.created_at), "updated_at": _stamp(row.updated_at),
            "revision": row.revision, "payload": row.geojson, "datasets": [],
            "annotation_name": image.annotation_name,
        }
    indices = {kind: build_per_image_index(config.images_root / ("orto" if kind == "ortho" else "kanopus"))
               for kind in {row.imagery_type for row in classes.values()}}
    dataset_count = 0
    for row in sorted(rows.values(), key=lambda item: str(item.id)):
        if row.source_type == "managed":
            continue
        folder = source_path(config, row, "").resolve()
        if not folder.is_dir() or any(folder.glob("*.txt")):
            warnings.append({"dataset": row.key, "reason": "Нет папки или старый формат разметки"})
            continue
        dataset_count += 1
        kind = classes[row.class_id].imagery_type
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() != ".geojson" or is_per_image_footprint_name(path.name):
                continue
            candidates = indices[kind].get(path.name.casefold(), [])
            if len(candidates) != 1:
                warnings.append({"file": path.relative_to(config.mlmarkup_editor_root).as_posix(),
                                 "reason": "Неоднозначное сопоставление со снимком"})
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            digest = payload_hash(payload)
            files[path.relative_to(config.mlmarkup_editor_root).as_posix()] = digest
            binding = bindings.get((row.id, path.name.casefold()))
            key = str(binding.version_id) if binding else None
            if key not in records or binding.materialized_hash != digest:
                key = f"file:{row.id}:{path.name}"
                relative = candidates[0].resolve().relative_to(
                    (config.images_root / ("orto" if kind == "ortho" else "kanopus")).resolve()).as_posix()
                records[key] = {
                    "id": key, "class_id": str(row.class_id), "class": classes[row.class_id].name,
                    "image": relative, "kind": kind, "name": f"Разметка «{row.name}»",
                    "created_at": "9999", "updated_at": "9999", "revision": 1,
                    "payload": payload, "datasets": [], "annotation_name": path.name,
                }
            records[key]["datasets"].append({"key": row.key, "name": row.name, "id": str(row.id),
                                             "annotation_name": path.name})

    # Защищаем весь каталог снимка/класса: его ревизия общая для всех вариантов.
    protected = set()
    drafts = list(session.scalars(select(DatasetEditorDraftRow)))
    by_key = {row.key: row for row in rows.values()}
    dependencies = defaultdict(set)
    for link in session.scalars(select(ManagedDatasetSourceRow)):
        dependencies[link.managed_dataset_id].add(link.source_dataset_id)
    for draft in drafts:
        row = by_key.get(draft.dataset_key)
        if row is None:
            continue
        source_ids = dependencies[row.id] if row.source_type == "managed" else {row.id}
        for source_id in source_ids:
            source = rows.get(source_id)
            if source is None:
                continue
            kind = classes[source.class_id].imagery_type
            candidates = indices[kind].get(draft.annotation_name.casefold(), [])
            for path in candidates:
                relative = path.resolve().relative_to(
                    (config.images_root / ("orto" if kind == "ortho" else "kanopus")).resolve()).as_posix()
                protected.add((str(source.class_id), kind, relative))
            state = draft.annotation_versions or {}
            ids = [state.get("selected_id"), *state.get("replacements", {}),
                   *state.get("replacements", {}).values(), *[item.get("id") for item in state.get("edits", [])]]
            for key in ids:
                if key in records:
                    item = records[key]
                    protected.add((item["class_id"], item["kind"], item["image"]))

    grouped = defaultdict(list)
    for item in records.values():
        payload = item.pop("payload")
        item["payload_hash"] = payload_hash(payload)
        try:
            item["signature"], count = geometry_signature(payload)
        except (ValueError, TypeError) as exc:
            warnings.append({"version": item["id"], "reason": str(exc)})
            continue
        if count:
            grouped[(item["class_id"], item["kind"], item["image"], item["signature"])].append(item)
    groups = []
    for key, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda item: (
            not any(dataset["name"].casefold() == "main" for dataset in item["datasets"]),
            -len(item["datasets"]), item["created_at"], item["id"],
        ))
        reason = "Есть неопубликованный черновик" if key[:3] in protected else None
        if not any(item["datasets"] for item in members):
            reason = "У группы нет активного датасета для публикации"
        groups.append({"class": members[0]["class"], "image": key[2], "signature": key[3],
                       "keep": members[0]["id"], "members": members, "skip": reason})
    report = {
        "operation": OPERATION, "markup_commit": editor._git(config, "rev-parse", "HEAD").stdout.strip(),
        "dataset_count": dataset_count, "version_count": len(records), "groups": groups,
        "warnings": warnings, "draft_count": len(drafts),
        "protected_catalogs": sorted(protected),
        "draft_signatures": {str(row.id): payload_hash({"geojson": row.geojson,
            "versions": row.annotation_versions, "deleted": row.deleted, "base": row.base_revision,
            "updated_at": _stamp(row.updated_at)}) for row in drafts},
    }
    # Включаем все варианты/связи/черновики, а не только группы: применение устаревшего плана запрещено.
    report["fingerprint"] = payload_hash({
        "report": report, "records": records, "files": files,
        "datasets": sorted((str(row.id), row.name, row.source_path, row.config_revision) for row in rows.values()),
        "bindings": sorted((str(row.id), str(row.version_id), row.revision, row.materialized_hash)
                           for row in bindings.values()),
        "drafts": sorted((str(row.id), _stamp(row.updated_at), payload_hash(row.geojson),
                          row.annotation_versions, row.deleted, row.base_revision) for row in drafts),
    })
    return report


def automation_exclusions(session):
    """Незавершённая операция блокирует автозапуск; завершённая — только свои ревизии."""
    pending, exact = set(), set()
    # Не загружать тяжёлые GeoJSON журнала при каждом обходе автоматизации.
    for completed, keys, versions in session.execute(select(
        AnnotationPublicationRow.completed,
        AnnotationPublicationRow.payload["affected_dataset_keys"],
        AnnotationPublicationRow.payload["automation_exclusions"],
    ).where(AnnotationPublicationRow.payload["operation"].as_string() == OPERATION)):
        if not completed:
            pending.update(keys or [])
        else:
            exact.update((versions or {}).items())
    return pending, exact


def complete_maintenance(session, config, publication, commit):
    """Зафиксировать реальные версии, не подавляя более поздние изменения при восстановлении."""
    if publication.payload.get("operation") != OPERATION:
        return
    from ._datasets import _path_metadata
    from ._managed_datasets import managed_dataset_version

    payload = deepcopy(publication.payload)
    changed = editor._git(config, "diff", "--name-only", "-z", commit, "HEAD").stdout.split("\0")
    rows = {row.key: row for row in session.scalars(select(DatasetRow).where(DatasetRow.deleted_at.is_(None)))}
    exclusions = {}
    warnings = []
    for key, spec in payload["affected_datasets"].items():
        if any(path == folder or path.startswith(folder + "/") for path in changed for folder in spec["paths"]):
            continue
        if any(name not in rows or rows[name].config_revision != revision for name, revision in spec["revisions"].items()):
            continue
        if key not in rows:
            continue
        row = rows[key]
        if row.source_type == "managed":
            try:
                version, _ = managed_dataset_version(session, row, config.mlmarkup_editor_root)
            except TrainingUIAPIError as exc:
                # Недоступный набор не пригоден для автоматизации; не скрываем его будущую починку.
                warnings.append({"dataset": key, "reason": str(exc)})
                continue
        else:
            _, version = _path_metadata(source_path(config, row, ""), config.mlmarkup_editor_root)
            if not row.legacy_version:
                version = f"managed:{row.config_revision}:{version or 'missing'}"
        if version:
            exclusions[key] = version
    payload["automation_exclusions"] = exclusions
    payload["automation_warnings"] = warnings
    publication.payload = payload


def apply_duplicates(session, config, expected, *, username="Обслуживание разметок"):
    """Применить только заново проверенный отчёт; вызывается под блокировкой публикации."""
    if config.mlmarkup_root.resolve() != config.mlmarkup_editor_root.resolve():
        raise TrainingUIAPIError("Очистка требует рабочий Git-клон как опубликованный источник")
    if expected.get("operation") != OPERATION or not expected.get("fingerprint"):
        raise TrainingUIAPIError("Файл не является планом объединения разметок")
    editor.recover_annotation_publications(session, config)
    previous = session.scalar(select(AnnotationPublicationRow).where(
        AnnotationPublicationRow.completed.is_(True),
        AnnotationPublicationRow.payload["operation"].as_string() == OPERATION,
        AnnotationPublicationRow.payload["audit_fingerprint"].as_string() == expected["fingerprint"],
    ))
    if previous is not None:
        return {"commit": previous.commit_sha, "merged_versions": 0, "already_applied": True, "report": expected}
    editor._fetch_editor_clone(config)
    if editor._git(config, "status", "--porcelain").stdout.strip():
        raise TrainingUIAPIError("В репозитории разметок есть неопубликованные файловые изменения")
    remote = f"origin/{config.mlmarkup_editor_branch}"
    if editor._git(config, "rev-parse", "HEAD").stdout != editor._git(config, "rev-parse", remote).stdout:
        raise TrainingUIAPIError("Локальная и опубликованная ревизии различаются; повторите аудит после синхронизации")
    actual = inspect_duplicates(session, config)
    if actual["fingerprint"] != expected["fingerprint"]:
        raise TrainingUIAPIError("Разметки или черновики изменились после аудита; нужен новый план")
    plans, replacements, catalogs = [], {}, {}
    protected = {tuple(item) for item in actual["protected_catalogs"]}

    def make_catalog(dataset, annotation_name):
        def ensure_bindings(row):
            known = set(session.scalars(select(DatasetAnnotationRow.annotation_name)
                .where(DatasetAnnotationRow.dataset_id == row.id)))
            peer = dataset.model_copy(update={"key": row.key})
            for path in editor._direct_annotation_files(source_path(config, row, "")):
                if path.name in known:
                    continue
                image = editor._image_path_for_annotation(peer, path.name)
                kind = peer.imagery_type.value
                root = config.images_root / ("orto" if kind == "ortho" else "kanopus")
                if (str(row.class_id), kind, image.resolve().relative_to(root.resolve()).as_posix()) in protected:
                    continue
                AnnotationCatalog(session, config, peer, image, path.name)
        return AnnotationCatalog(session, config, dataset,
            editor._image_path_for_annotation(dataset, annotation_name), annotation_name,
            ensure_bindings=ensure_bindings)

    for group in actual["groups"]:
        if group["skip"]:
            continue
        seed = next(dataset for member in group["members"] for dataset in member["datasets"])
        key = (group["members"][0]["class_id"], group["image"])
        if key not in catalogs:
            dataset, _ = editor._editor_dataset_context(session, config, seed["key"])
            catalogs[key] = make_catalog(dataset, seed["annotation_name"])
        catalog = catalogs[key]
        ids = {}
        for member in group["members"]:
            if member["id"].startswith("file:"):
                binding = catalog.bindings[uuid.UUID(member["datasets"][0]["id"])]
                ids[member["id"]] = str(binding.version_id)
            else:
                ids[member["id"]] = member["id"]
            if geometry_signature(catalog.versions[ids[member["id"]]].geojson)[0] != group["signature"]:
                raise TrainingUIAPIError("Геометрия изменилась при импорте; очистка отменена")
        keep = ids[group["keep"]]
        replacements.update({value: keep for value in ids.values() if value != keep})
    for catalog in catalogs.values():
        mapping = {key: value for key, value in replacements.items() if key in catalog.versions}
        selected = str(catalog.catalog().selected_id)
        state = AnnotationVersionDraft(catalog_revision=catalog.catalog().revision,
            selected_id=mapping.get(selected, selected), replacements=mapping)
        plans.append(catalog.plan(state))
    if not plans:
        return {"commit": None, "merged_versions": 0, "report": actual}
    plan = merge_plans(plans)
    geometry_checks = {}
    for path, payload in plan["files"].items():
        if path.endswith(".geojson"):
            previous = json.loads((config.mlmarkup_editor_root / path).read_text(encoding="utf-8-sig"))
            before = geometry_signature(previous)
            if before != geometry_signature(payload):
                raise TrainingUIAPIError("Объединение изменяет геометрию или обучающие роли: " + path)
            geometry_checks[path] = before[0]
    paths = [PurePosixPath(path) for path in plan["files"]]
    expected_revisions = {path: editor._blob_revision(config, "HEAD", path) for path in paths}
    rows = {row.id: row for row in session.scalars(select(DatasetRow).where(DatasetRow.deleted_at.is_(None)))}
    changed_ids = {row.id for row in rows.values() if any(path.as_posix().startswith(row.source_path + "/") for path in paths)}
    affected = {rows[key].key: {"paths": [rows[key].source_path], "revisions": {rows[key].key: rows[key].config_revision}}
                for key in changed_ids}
    dependencies = defaultdict(set)
    for link in session.scalars(select(ManagedDatasetSourceRow)):
        dependencies[link.managed_dataset_id].add(link.source_dataset_id)
    for target, sources in dependencies.items():
        if target in rows and sources & changed_ids:
            source_rows = [rows[key] for key in sources if key in rows]
            affected[rows[target].key] = {"paths": [row.source_path for row in source_rows],
                "revisions": {row.key: row.config_revision for row in [rows[target], *source_rows]}}
    plan.update(operation=OPERATION, affected_dataset_keys=sorted(affected), affected_datasets=affected,
        expected_revisions={str(path): revision for path, revision in expected_revisions.items()},
        paths=[str(path) for path in paths], drafts=[], replacements=replacements,
        audit_fingerprint=actual["fingerprint"], geometry_checks=geometry_checks)
    publication = editor._prepare_annotation_publication(session, plan, username)
    try:
        for path, payload in plan["files"].items():
            editor._write_geojson_atomic(config.mlmarkup_editor_root / path, payload)
        for path, signature in geometry_checks.items():
            written = json.loads((config.mlmarkup_editor_root / path).read_text(encoding="utf-8-sig"))
            if geometry_signature(written)[0] != signature:
                raise TrainingUIAPIError("Проверка записанной геометрии не пройдена: " + path)
        editor._stage_annotation_paths(config, paths)
        commit = editor._commit(config, "Объединить одинаковые версии разметки без переобучения", username,
                                publication_id=publication.id)
        editor._record_annotation_commit(session, publication, commit)
        commit = editor._push_with_retry(config, expected_revisions=expected_revisions)
    except Exception:
        editor._restore_editor_paths(config, paths)
        editor._cancel_annotation_publication(session, publication)
        raise
    editor._complete_annotation_publication(session, publication, commit, config=config)
    session.commit()
    return {"commit": commit, "merged_versions": len(replacements), "replacements": replacements, "report": actual}


def main():
    parser = argparse.ArgumentParser(description="Проверить или объединить точные копии разметок")
    parser.add_argument("--report", required=True, type=Path, help="Новый файл отчёта")
    parser.add_argument("--apply-plan", type=Path, help="Применить ранее проверенный отчёт")
    args = parser.parse_args()
    config = get_config()
    # Не допускаем затереть единственную копию предварительного плана.
    if args.report.exists():
        parser.error("Файл отчёта уже существует")
    factory = create_session_factory(config)
    lock = editor._editor_lock(config, restore_ownership=True) if args.apply_plan else markup_read_lock(config)
    with lock, factory() as session:
        if args.apply_plan:
            expected = json.loads(args.apply_plan.read_text(encoding="utf-8"))
            result = apply_duplicates(session, config, expected)
        else:
            result = inspect_duplicates(session, config)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = result.get("report", result)
    counts = Counter()
    for group in report["groups"]:
        if not group["skip"]:
            counts[group["class"]] += len(group["members"]) - 1
    print(json.dumps({"повторные_версии_по_классам": dict(counts),
                      "отложенных_групп": sum(bool(group["skip"]) for group in report["groups"]),
                      "предупреждений": len(report["warnings"]), "объединено": result.get("merged_versions"),
                      "коммит": result.get("commit")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
