"""Открытые вкладки сохраняют доступ к своим модулям после обновления сайта."""

import os
from pathlib import Path

import pytest

from frontend import deploy


def _build(root, name, assets):
    path = root / name
    (path / "assets").mkdir(parents=True)
    (path / "index.html").write_text(name, encoding="utf-8")
    for asset, content in assets.items():
        (path / "assets" / asset).write_text(content, encoding="utf-8")
    return path


def test_open_tab_keeps_lazy_modules_after_next_deploy(tmp_path):
    target = tmp_path / "published"
    first = _build(tmp_path, "first", {"page-old.js": "старый просмотр", "decoder-old.js": "декодер"})
    second = _build(tmp_path, "second", {"page-new.js": "новый просмотр"})
    deploy.deploy_frontend(first, target)
    deploy.deploy_frontend(second, target)
    assert (target / "index.html").read_text(encoding="utf-8") == "second"
    assert {p.name for p in (target / "assets").iterdir()} == {"page-old.js", "decoder-old.js", "page-new.js"}


def test_prune_expires_old_assets_and_bounds_size_but_keeps_current(tmp_path):
    target = tmp_path / "published"
    old = _build(tmp_path, "old", {"expired.js": "a" * 20, "older.js": "b" * 20, "recent.js": "c" * 20})
    current = _build(tmp_path, "new", {"current.js": "d" * 80})
    deploy.deploy_frontend(old, target)
    for name, timestamp in [("expired.js", 0), ("older.js", 950), ("recent.js", 980)]:
        os.utime(target / "assets" / name, (timestamp, timestamp))
    os.utime(current / "assets/current.js", (0, 0))
    deploy.deploy_frontend(current, target, now=1000, ttl=100, limit=100)
    assert {p.name for p in (target / "assets").iterdir()} == {"current.js", "recent.js"}
    assert sum(p.stat().st_size for p in (target / "assets").iterdir()) == 100


def test_incomplete_copy_does_not_publish_new_entry(tmp_path, monkeypatch):
    target = tmp_path / "published"
    old = _build(tmp_path, "old", {"old.js": "рабочий модуль"})
    new = _build(tmp_path, "new", {"new.js": "новый модуль"})
    deploy.deploy_frontend(old, target)
    original = deploy._copy_atomic

    def fail_asset(source, destination, root):
        if source.name == "new.js":
            raise OSError("Копирование прервано")
        original(source, destination, root)

    monkeypatch.setattr(deploy, "_copy_atomic", fail_asset)
    with pytest.raises(OSError):
        deploy.deploy_frontend(new, target)
    assert (target / "index.html").read_text(encoding="utf-8") == "old"
    assert (target / "assets/old.js").is_file()


@pytest.mark.parametrize("target_name", ["build", "build/output"])
def test_rejects_overlapping_build_and_publication(tmp_path, target_name):
    source = _build(tmp_path, "build", {"current.js": "модуль"})
    with pytest.raises(ValueError):
        deploy.deploy_frontend(source, tmp_path / target_name)
    assert (source / "assets/current.js").is_file()
