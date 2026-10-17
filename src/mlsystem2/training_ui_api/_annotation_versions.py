"""Сущности снимков и вариантов разметки; Git получает только собранные датасеты."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from ._config import TrainingUIAPIConfig
from ._models import (
    AnnotationImageRow, AnnotationVersionRow, DatasetAnnotationRow, DatasetClassRow,
    DatasetRow, ManagedDatasetSourceRow,
)
from .contracts import (
    AnnotationVersionCatalog, AnnotationVersionDraft, AnnotationVersionEdit,
    AnnotationVersionInfo, AnnotationVersionUsage, DatasetInfo, TrainingUIAPIError,
)


class AnnotationVersionConflict(RuntimeError):
    """Выбранная версия или её использование изменились после открытия редактора."""


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_path(config: TrainingUIAPIConfig, row: DatasetRow, annotation_name: str) -> Path:
    root = config.mlmarkup_editor_root.resolve()
    path = root.joinpath(*PurePosixPath(row.source_path).parts, annotation_name).resolve()
    if not path.is_relative_to(root):
        raise TrainingUIAPIError("Источник разметки выходит за корень каталога")
    return path


class AnnotationCatalog:
    """Один снимок и один класс. Вызов выполняется под общей блокировкой публикации."""

    def __init__(self, session: Session, config: TrainingUIAPIConfig, dataset: DatasetInfo,
                 image_path: Path, annotation_name: str, *,
                 ensure_bindings: Callable[[DatasetRow], None] | None = None):
        self.session, self.config, self.dataset = session, config, dataset
        self.annotation_name = annotation_name
        self.ensure_bindings = ensure_bindings
        self.dataset_row = session.scalar(select(DatasetRow).where(DatasetRow.key == dataset.key, DatasetRow.deleted_at.is_(None)))
        if self.dataset_row is None or dataset.managed:
            raise TrainingUIAPIError("Версии выбираются в исходном датасете отдельного класса")
        imagery_type = dataset.imagery_type.value
        root = config.images_root / ("orto" if imagery_type == "ortho" else "kanopus")
        relative = image_path.resolve().relative_to(root.resolve()).as_posix()
        image = session.scalar(select(AnnotationImageRow).where(
            AnnotationImageRow.imagery_type == imagery_type, AnnotationImageRow.relative_path == relative))
        if image is None:
            image = AnnotationImageRow(id=uuid.uuid4(), imagery_type=imagery_type,
                                       relative_path=relative, annotation_name=annotation_name)
            session.add(image)
            session.flush()
        self.image = image
        self.rows = {row.id: row for row in session.scalars(select(DatasetRow).where(
            DatasetRow.class_id == self.dataset_row.class_id,
            DatasetRow.source_type != "managed", DatasetRow.deleted_at.is_(None))).all()}
        self.versions = {str(row.id): row for row in session.scalars(select(AnnotationVersionRow).where(
            AnnotationVersionRow.image_id == image.id,
            AnnotationVersionRow.class_id == self.dataset_row.class_id,
            AnnotationVersionRow.deleted_at.is_(None))).all()}
        bindings = {row.dataset_id: row for row in session.scalars(select(DatasetAnnotationRow).where(
            DatasetAnnotationRow.image_id == image.id,
            DatasetAnnotationRow.dataset_id.in_(self.rows))).all()}
        # Однократный импорт прежних файлов. Внешнее изменение файла создаёт отдельную
        # версию и не перезаписывает общую разметку других датасетов.
        for row in self.rows.values():
            binding = bindings.get(row.id)
            path = source_path(config, row, binding.annotation_name if binding else annotation_name)
            if (binding is None or not path.is_file()) and path.parent.is_dir():
                candidates = [item for item in path.parent.iterdir() if item.is_file() and item.name.casefold() == path.name.casefold()]
                if len(candidates) > 1:
                    raise TrainingUIAPIError("Имена разметок одного снимка отличаются только регистром: " + row.name)
                if len(candidates) == 1:
                    path = candidates[0]
            if any(path.parent.glob("*.txt")):
                continue
            if not path.is_file():
                if binding is not None:
                    session.delete(binding)
                    bindings.pop(row.id)
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            digest = payload_hash(payload)
            if binding is not None and binding.materialized_hash == digest and str(binding.version_id) in self.versions:
                if binding.annotation_name != path.name:
                    binding.annotation_name = path.name
                    binding.revision += 1
                continue
            version = AnnotationVersionRow(
                id=uuid.uuid4(), image_id=image.id, class_id=row.class_id,
                name=(f"Разметка «{row.name}»" if binding is None else f"Импорт из файлов «{row.name}»")[:240],
                geojson=payload, revision=1,
            )
            session.add(version)
            session.flush()
            self.versions[str(version.id)] = version
            if binding is None:
                binding = DatasetAnnotationRow(id=uuid.uuid4(), dataset_id=row.id, image_id=image.id,
                    annotation_name=path.name, version_id=version.id, materialized_hash=digest, revision=1)
                session.add(binding)
                bindings[row.id] = binding
            else:
                binding.version_id = version.id
                binding.annotation_name = path.name
                binding.materialized_hash = digest
                binding.revision += 1
        session.flush()
        self.bindings = bindings

    def catalog(self) -> AnnotationVersionCatalog:
        dependencies: dict[uuid.UUID, list[AnnotationVersionUsage]] = {}
        for source_id, key, name, class_name in self.session.execute(
            select(ManagedDatasetSourceRow.source_dataset_id, DatasetRow.key, DatasetRow.name, DatasetClassRow.name)
            .join(DatasetRow, DatasetRow.id == ManagedDatasetSourceRow.managed_dataset_id)
            .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
            .where(ManagedDatasetSourceRow.source_dataset_id.in_(self.rows), DatasetRow.deleted_at.is_(None))
        ):
            dependencies.setdefault(source_id, []).append(AnnotationVersionUsage(
                dataset_key=key, dataset_name=f"{class_name} / {name}", indirect=True))
        versions = []
        for key, row in sorted(self.versions.items(), key=lambda item: (item[1].created_at, item[0])):
            usages: dict[str, AnnotationVersionUsage] = {}
            for binding in self.bindings.values():
                if str(binding.version_id) != key:
                    continue
                dataset = self.rows[binding.dataset_id]
                usages[dataset.key] = AnnotationVersionUsage(dataset_key=dataset.key, dataset_name=dataset.name)
                for usage in dependencies.get(dataset.id, []):
                    usages[usage.dataset_key] = usage
            versions.append(AnnotationVersionInfo(id=row.id, name=row.name, revision=row.revision,
                usages=sorted(usages.values(), key=lambda usage: usage.dataset_name.casefold())))
        binding = self.bindings.get(self.dataset_row.id)
        revision = payload_hash({
            "versions": sorted((key, row.revision) for key, row in self.versions.items()),
            "bindings": sorted((str(row.dataset_id), str(row.version_id), row.revision) for row in self.bindings.values()),
        })
        return AnnotationVersionCatalog(image_id=self.image.id, revision=revision,
            selected_id=binding.version_id if binding else None, versions=versions)

    def initial_draft(self) -> AnnotationVersionDraft | None:
        catalog = self.catalog()
        return (AnnotationVersionDraft(catalog_revision=catalog.revision, selected_id=catalog.selected_id)
                if catalog.selected_id else None)

    def prepare(self, state: AnnotationVersionDraft, geojson: dict[str, Any],
                normalize: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]], *,
                display: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> AnnotationVersionDraft:
        if state.catalog_revision != self.catalog().revision:
            raise AnnotationVersionConflict("Версии разметки или их использование изменились. Сохраните локальную работу и откройте снимок заново.")
        edits = {str(edit.id): edit for edit in state.edits}
        if len(edits) != len(state.edits):
            raise TrainingUIAPIError("В черновике повторяются идентификаторы версий")
        selected = str(state.selected_id)
        if selected not in edits:
            row = self.versions.get(selected)
            if row is None:
                raise TrainingUIAPIError("Выбранная версия не принадлежит этому снимку и классу")
            # Выбор опубликованной версии не является правкой её геометрии.
            # Редактор показывает пересечение с полезной областью TIFF, но в БД
            # и других датасетах должен сохраниться полный исходный GeoJSON.
            if display is None or payload_hash(geojson) != payload_hash(display(deepcopy(row.geojson))):
                edits[selected] = AnnotationVersionEdit(id=row.id, name=row.name, geojson=geojson)
        else:
            edits[selected] = edits[selected].model_copy(update={"geojson": geojson})
        prepared: dict[str, AnnotationVersionEdit] = {}

        def prepare_one(key: str, visiting: set[str]) -> AnnotationVersionEdit:
            if key in prepared:
                return prepared[key]
            if key in visiting:
                raise TrainingUIAPIError("Циклическое копирование версий недопустимо")
            edit = edits[key]
            existing = self.versions.get(key)
            if existing:
                previous = existing.geojson
            else:
                if self.session.get(AnnotationVersionRow, edit.id) is not None:
                    raise TrainingUIAPIError("Идентификатор новой версии уже занят")
                source = str(edit.source_id) if edit.source_id else None
                if source in self.versions:
                    previous = self.versions[source].geojson
                elif source in edits:
                    previous = prepare_one(source, visiting | {key}).geojson
                elif source:
                    raise TrainingUIAPIError("Источник копии не принадлежит снимку и классу")
                else:
                    previous = {**edit.geojson, "features": []}
            name = " ".join(edit.name.split())
            if not name:
                raise TrainingUIAPIError("Укажите название версии")
            payload = normalize(deepcopy(edit.geojson), deepcopy(previous))
            prepared[key] = edit.model_copy(update={"name": name, "geojson": payload})
            return prepared[key]

        for key in edits:
            prepare_one(key, set())
        available = set(self.versions) | set(prepared)
        replacements = {str(key): str(value) for key, value in state.replacements.items()}
        for key, target in replacements.items():
            if key not in available or target not in available or key == target or target in replacements:
                raise TrainingUIAPIError("Для каждой удаляемой версии выберите другую, сохраняемую версию этого снимка")
        if selected in replacements or selected not in available:
            raise TrainingUIAPIError("Датасет должен использовать сохраняемую версию разметки")
        names = {row.name.casefold() for key, row in self.versions.items() if key not in replacements}
        for key, edit in prepared.items():
            if key in self.versions or key in replacements:
                continue
            if edit.name.casefold() in names:
                raise TrainingUIAPIError("Версия с таким названием уже есть у этого снимка")
            names.add(edit.name.casefold())
        return state.model_copy(update={"edits": list(prepared.values())})

    def plan(self, state: AnnotationVersionDraft | None, *, remove_scene: bool = False) -> dict[str, Any]:
        """Подготовить изменения БД и файлов без изменения опубликованных сущностей."""
        if state and state.catalog_revision != self.catalog().revision:
            raise AnnotationVersionConflict("Версии разметки изменились после сохранения черновика")
        versions = {key: {"id": key, "image_id": str(row.image_id), "class_id": str(row.class_id),
            "name": row.name, "geojson": deepcopy(row.geojson), "revision": row.revision, "deleted": False}
            for key, row in self.versions.items()}
        if state:
            for edit in state.edits:
                key = str(edit.id)
                old = versions.get(key)
                versions[key] = {"id": key, "image_id": str(self.image.id), "class_id": str(self.dataset_row.class_id),
                    "name": edit.name, "geojson": edit.geojson,
                    "revision": (old["revision"] + int(old["geojson"] != edit.geojson or old["name"] != edit.name)) if old else 1,
                    "deleted": False}
        replacements = {str(key): str(value) for key, value in state.replacements.items()} if state else {}
        for key in replacements:
            versions[key]["deleted"] = True
        bindings = []
        files: dict[str, dict[str, Any]] = {}
        for row_id, binding in self.bindings.items():
            row = self.rows[row_id]
            key = str(binding.version_id)
            key = replacements.get(key, key)
            if row_id == self.dataset_row.id and state:
                key = str(state.selected_id)
            removed = row_id == self.dataset_row.id and remove_scene
            version = versions[key]
            digest = payload_hash(version["geojson"])
            changed = key != str(binding.version_id) or digest != binding.materialized_hash
            bindings.append({"id": str(binding.id), "dataset_id": str(row_id), "dataset_key": row.key, "image_id": str(self.image.id),
                "version_id": key, "annotation_name": binding.annotation_name,
                "revision": binding.revision + int(changed), "materialized_hash": digest, "deleted": removed})
            if changed and not removed:
                relative = source_path(self.config, row, binding.annotation_name).relative_to(self.config.mlmarkup_editor_root.resolve()).as_posix()
                files[relative] = version["geojson"]
            if changed or removed:
                selection_path = PurePosixPath(row.source_path) / ".mlsystem2-annotation-selections.json"
                files[selection_path.as_posix()] = self._selection_manifest(row, key, version["revision"], removed)
        if state and self.dataset_row.id not in self.bindings and not remove_scene:
            key = str(state.selected_id)
            version = versions[key]
            bindings.append({"id": str(uuid.uuid4()), "dataset_id": str(self.dataset_row.id), "dataset_key": self.dataset_row.key, "image_id": str(self.image.id),
                "version_id": key, "annotation_name": self.annotation_name, "revision": 1,
                "materialized_hash": payload_hash(version["geojson"]), "deleted": False})
            path = source_path(self.config, self.dataset_row, self.annotation_name)
            files[path.relative_to(self.config.mlmarkup_editor_root.resolve()).as_posix()] = version["geojson"]
            files[(PurePosixPath(self.dataset_row.source_path) / ".mlsystem2-annotation-selections.json").as_posix()] = self._selection_manifest(self.dataset_row, key, version["revision"], False)
        return {"versions": list(versions.values()), "bindings": bindings, "files": files}

    def _selection_manifest(self, row: DatasetRow, version_id: str, revision: int, removed: bool) -> dict[str, Any]:
        # Манифест должен включать и снимки, ещё не открывавшиеся в редакторе.
        if self.ensure_bindings is not None:
            self.ensure_bindings(row)
        selections = {}
        for binding, version in self.session.execute(select(DatasetAnnotationRow, AnnotationVersionRow)
            .join(AnnotationVersionRow, AnnotationVersionRow.id == DatasetAnnotationRow.version_id)
            .where(DatasetAnnotationRow.dataset_id == row.id)):
            selections[binding.annotation_name] = {"image_id": str(binding.image_id), "version_id": str(version.id), "revision": version.revision}
        binding = self.bindings.get(row.id)
        annotation_name = binding.annotation_name if binding else self.annotation_name
        if removed:
            selections.pop(annotation_name, None)
        else:
            selections[annotation_name] = {"image_id": str(self.image.id), "version_id": version_id, "revision": revision}
        return {"schema_version": 1, "dataset_key": row.key, "annotations": selections}


def merge_plans(plans: list[dict[str, Any]]) -> dict[str, Any]:
    versions, bindings, files = {}, {}, {}
    for plan in plans:
        versions.update({item["id"]: item for item in plan["versions"]})
        bindings.update({item["id"]: item for item in plan["bindings"]})
        for path, payload in plan["files"].items():
            if path.endswith("/.mlsystem2-annotation-selections.json"):
                files.setdefault(path, deepcopy(payload))
            else:
                files[path] = payload
    # Полный итоговый состав пересобирается по изменениям, в том числе удалениям.
    for path, payload in files.items():
        if path.endswith("/.mlsystem2-annotation-selections.json"):
            for binding in bindings.values():
                if binding["dataset_key"] != payload["dataset_key"]:
                    continue
                if binding["deleted"]:
                    payload["annotations"].pop(binding["annotation_name"], None)
                else:
                    payload["annotations"][binding["annotation_name"]] = {
                        "image_id": binding["image_id"], "version_id": binding["version_id"],
                        "revision": versions[binding["version_id"]]["revision"],
                    }
    return {"versions": list(versions.values()), "bindings": list(bindings.values()), "files": files}


def apply_plan(session: Session, plan: dict[str, Any]) -> None:
    """Идемпотентное завершение подтверждённой публикации, включая восстановление после сбоя."""
    for item in plan["versions"]:
        key = uuid.UUID(item["id"])
        row = session.get(AnnotationVersionRow, key)
        if row is None:
            row = AnnotationVersionRow(id=key, image_id=uuid.UUID(item["image_id"]), class_id=uuid.UUID(item["class_id"]))
            session.add(row)
        row.name, row.geojson, row.revision = item["name"], item["geojson"], item["revision"]
        row.deleted_at = datetime.now(timezone.utc) if item["deleted"] else None
    session.flush()
    for item in plan["bindings"]:
        key = uuid.UUID(item["id"])
        row = session.get(DatasetAnnotationRow, key)
        if item["deleted"]:
            if row is not None:
                session.delete(row)
            continue
        if row is None:
            row = DatasetAnnotationRow(id=key, dataset_id=uuid.UUID(item["dataset_id"]), image_id=uuid.UUID(item["image_id"]))
            session.add(row)
        row.version_id = uuid.UUID(item["version_id"])
        row.annotation_name, row.materialized_hash, row.revision = item["annotation_name"], item["materialized_hash"], item["revision"]
    session.flush()
