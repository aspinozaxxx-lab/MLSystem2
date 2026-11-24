"""Фиксация опубликованной комбо-разметки и однократная публикация независимой копии."""

from __future__ import annotations

import logging
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import load_dataset_manifest
from mlsystem2.dataset_preparing.contracts import DatasetManifest, DatasetPreparationError

from ._config import TrainingUIAPIConfig
from ._dataset_editor import (
    _commit, _dataset_copy_name, _dataset_copy_target, _editor_lock,
    _ensure_dataset_copy_source_available, _git, _git_optional, _push_with_retry,
    _synchronize_editor_clone, _tree_object_revision,
)
from ._datasets import imagery_images_dir
from ._managed_datasets import (
    SOURCE_MANAGED, SOURCE_SNAPSHOT_PENDING, _write_managed_directory, managed_manifest,
    managed_scenes, managed_source_infos,
)
from ._models import (
    DatasetClassRow, DatasetEditorDraftRow, DatasetRow, ManagedDatasetSceneRow,
    ManagedDatasetSourceRow,
)
from .contracts import TrainingUIAPIError

LOGGER = logging.getLogger(__name__)


class _InvalidSnapshotData(Exception):
    """Ошибка исходных данных, обнаруженная до изменения рабочего дерева."""


def request_snapshot(session: Session, config: TrainingUIAPIConfig, row: DatasetRow,
                     owner: DatasetClassRow, username: str, *, detach_existing: bool = False) -> None:
    """Вызывается под блокировкой публикации. Геометрия собирается позже в worker."""
    if session.scalar(select(DatasetEditorDraftRow.id).where(
        DatasetEditorDraftRow.dataset_key == row.key).limit(1)) is not None:
        raise TrainingUIAPIError("Сначала опубликуйте или отмените черновики этого комбо-датасета.")
    _synchronize_editor_clone(config)
    revision = _git(config, "rev-parse", f"origin/{config.mlmarkup_editor_branch}").stdout.strip()
    manifest = managed_manifest(session, row)
    empty_paths = []
    for source in manifest.sources:
        if _tree_object_revision(config, revision, PurePosixPath(source.path)) is None:
            # Git не хранит пустые каталоги. Допускаем их, если опубликованный каталог действительно пуст.
            folder = config.mlmarkup_root / source.path
            if not folder.is_dir() or any(folder.glob("*.geojson")):
                raise TrainingUIAPIError(f"В опубликованной разметке отсутствует источник {source.path}.")
            empty_paths.append(source.path)
    target = _dataset_copy_target(session, config, PurePosixPath(_dataset_copy_name(owner.name)),
                                  _dataset_copy_name(row.name))
    _ensure_dataset_copy_source_available(session, target, config.mlmarkup_editor_root / target)
    options = {key: value for key, value in (row.composition or {}).items() if key != "detach_error"}
    row.composition = {
        **options, "revision": revision,
        "classes": [item.model_dump(mode="json") for item in managed_source_infos(session, row.id)],
        "pending": {
            "manifest": manifest.model_dump(mode="json"), "author": username,
            "imagery_type": owner.imagery_type,
            "empty_paths": empty_paths,
            "previous_dynamic": {
                "path": row.source_path, "revision": row.config_revision, "legacy": row.legacy_version,
                "options": options,
            } if detach_existing else None,
            "scenes": [[s.annotation_name, s.image_relative_path] for s in managed_scenes(session, row.id)],
        },
    }
    row.source_type = SOURCE_SNAPSHOT_PENDING
    row.source_path = target.as_posix()
    row.config_revision += 1
    row.legacy_version = False
    if not detach_existing:
        _remove_sources(session, row)
    session.flush()


def _remove_sources(session: Session, row: DatasetRow) -> None:
    for model in (ManagedDatasetSourceRow, ManagedDatasetSceneRow):
        session.execute(delete(model).where(model.managed_dataset_id == row.id))


def _pending_snapshots():
    retry_at = DatasetRow.composition["retry_at"].as_float()
    invalid = DatasetRow.composition["invalid"].as_boolean()
    return select(DatasetRow).where(
        DatasetRow.source_type == SOURCE_SNAPSHOT_PENDING, DatasetRow.deleted_at.is_(None),
        or_(retry_at.is_(None), retry_at <= time.time()),
        or_(invalid.is_(None), invalid.is_(False)),
    ).order_by(DatasetRow.created_at, DatasetRow.id).limit(1)


def has_pending_snapshot(session: Session) -> bool:
    return session.scalar(_pending_snapshots().with_only_columns(DatasetRow.id)) is not None


def process_next_snapshot(session: Session, config: TrainingUIAPIConfig) -> bool:
    row = session.scalar(_pending_snapshots().with_for_update(skip_locked=True))
    if row is None:
        return False
    data = dict(row.composition)
    try:
        _publish_snapshot(config, row, data)
    except _InvalidSnapshotData as exc:
        previous = data["pending"].get("previous_dynamic")
        if previous:
            row.source_type = SOURCE_MANAGED
            row.source_path, row.config_revision, row.legacy_version = previous["path"], previous["revision"], previous["legacy"]
            row.composition = {**previous["options"], "detach_error": f"Связь сохранена. Исправьте исходные данные и повторите отделение. {exc}"}
        else:
            row.composition = {**data, "invalid": True, "error": f"Копия не создана. Удалите заявку и создайте копию после исправления исходных данных. {exc}"}
        LOGGER.warning("Не удалось собрать разметку %s: %s", row.key, exc)
    except Exception as exc:
        attempts = data.get("attempts", 0) + 1
        row.composition = {**data, "error": str(exc), "attempts": attempts,
                           "retry_at": time.time() + min(900, 30 * 2 ** min(attempts, 5))}
        LOGGER.exception("Не удалось подготовить независимую копию %s", row.key)
    else:
        if data["pending"].get("previous_dynamic"):
            _remove_sources(session, row)
        row.composition = {k: v for k, v in data.items() if k not in {"pending", "error", "attempts", "retry_at"}}
        row.source_type = "mlmarkup"
    session.flush()
    return True


def _publish_snapshot(config: TrainingUIAPIConfig, row: DatasetRow, data: dict) -> None:
    pending = data["pending"]
    manifest = DatasetManifest.model_validate(pending["manifest"])
    revision = data["revision"]
    relative = PurePosixPath(row.source_path)
    target = (config.mlmarkup_editor_root / relative).resolve()
    if not target.is_relative_to(config.mlmarkup_editor_root.resolve()) or len(relative.parts) != 2:
        raise TrainingUIAPIError("Некорректная папка независимой копии.")
    if target.exists():
        with _editor_lock(config, restore_ownership=True):
            if _resume_snapshot(config, row, relative, target, pending["author"]):
                return
    config.scratch_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="combo-", dir=config.scratch_root) as temporary:
        work = Path(temporary).resolve()
        archive = work / "sources.tar"
        root = work / "sources"
        root.mkdir()
        tracked_paths = []
        for source in manifest.sources:
            path = (root / source.path).resolve()
            if not path.is_relative_to(root):
                raise TrainingUIAPIError("Путь разметки выходит за пределы копии.")
            if source.path in pending.get("empty_paths", []):
                path.mkdir(parents=True, exist_ok=True)
            else:
                tracked_paths.append(source.path)
        if tracked_paths:
            _git(config, "archive", "--format=tar", "-o", str(archive), revision, "--", *tracked_paths)
            with tarfile.open(archive) as stream:
                for member in stream:
                    path = (root / member.name).resolve()
                    if not path.is_relative_to(root):
                        raise TrainingUIAPIError("Путь разметки выходит за пределы копии.")
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    elif member.isfile() and (path.suffix.lower() == ".geojson" or path.name == ".mlsystem2-dataset.json"):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with stream.extractfile(member) as source, path.open("wb") as dest:
                            shutil.copyfileobj(source, dest)
        (root / ".mlsystem2-release").write_text(revision, encoding="utf-8")
        # Каталог Git находится на том же диске, что и рабочее дерево обычного editor-клона.
        # Готовая папка появляется в дереве атомарно, в том числе при перезапуске worker.
        git_directory = _git(config, "rev-parse", "--absolute-git-dir").stdout.strip()
        with TemporaryDirectory(prefix="combo-", dir=git_directory) as staging:
            output = Path(staging) / "dataset"
            try:
                _write_managed_directory(
                    config, manifest, root, imagery_images_dir(config.images_root, pending["imagery_type"]),
                    pending["scenes"], output, revision, snapshot_key=row.key,
                )
            except (TrainingUIAPIError, DatasetPreparationError) as exc:
                raise _InvalidSnapshotData(str(exc)) from exc
            with _editor_lock(config, restore_ownership=True):
                if _resume_snapshot(config, row, relative, target, pending["author"]):
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                output.rename(target)
                _commit_snapshot(config, row, relative, target, pending["author"])


def _resume_snapshot(config: TrainingUIAPIConfig, row: DatasetRow, relative: PurePosixPath,
                     target: Path, author: str) -> bool:
    if target.is_dir() and not _tree_object_revision(config, "HEAD", relative):
        existing = load_dataset_manifest(target)
        if existing is None or existing.snapshot_key != row.key:
            raise TrainingUIAPIError("Папка независимой копии уже занята в MLMarkup.")
        # После атомарного перемещения процесс мог остановиться до Git-коммита.
        outside = _git(config, "status", "--porcelain", "--", ".", f":(exclude,literal){relative.as_posix()}")
        if outside.stdout.strip():
            raise TrainingUIAPIError("В рабочем клоне есть незавершённые изменения вне копии.")
        _commit_snapshot(config, row, relative, target, author)
        return True
    _synchronize_editor_clone(config)
    if not target.exists():
        return False
    existing = load_dataset_manifest(target)
    if existing is None or existing.snapshot_key != row.key:
        raise TrainingUIAPIError("Папка независимой копии уже занята в MLMarkup.")
    if not _tree_object_revision(config, f"origin/{config.mlmarkup_editor_branch}", relative):
        _push_with_retry(config, expected_revisions={}, expected_tree_revisions={relative: None})
    return True


def _commit_snapshot(config: TrainingUIAPIConfig, row: DatasetRow, relative: PurePosixPath,
                     target: Path, author: str) -> None:
    try:
        _git(config, "add", "--", relative.as_posix())
        _commit(config, f"Сохранить независимую разметку {row.name}", author)
        _push_with_retry(config, expected_revisions={}, expected_tree_revisions={relative: None})
    except Exception:
        _git_optional(config, "reset", "--quiet", "HEAD", "--", relative.as_posix())
        if not _tree_object_revision(config, "HEAD", relative) and target.is_dir():
            shutil.rmtree(target)
        raise
