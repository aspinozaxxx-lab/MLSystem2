"""Доставка frontend без удаления файлов, нужных уже открытым вкладкам."""

from __future__ import annotations

import argparse
import os
import shutil
import time
import uuid
from pathlib import Path

ASSET_TTL_SECONDS = 7 * 24 * 60 * 60
ASSET_LIMIT_BYTES = 128 * 1024 * 1024


def _copy_atomic(source: Path, destination: Path, root: Path) -> None:
    if not destination.parent.resolve().is_relative_to(root):
        raise ValueError("Путь публикации выходит за каталог frontend")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def deploy_frontend(source: Path, target: Path, *, now: float | None = None,
                    ttl: int = ASSET_TTL_SECONDS, limit: int = ASSET_LIMIT_BYTES) -> None:
    source, target = source.resolve(), target.resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Сборка и каталог публикации должны быть разными и не вложенными")
    if not (source / "index.html").is_file() or not (source / "assets").is_dir():
        raise ValueError("Сборка frontend должна содержать index.html и assets")
    assets = target / "assets"
    if not assets.resolve().is_relative_to(target):
        raise ValueError("Каталог assets выходит за каталог публикации")
    current = set()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path == source / "index.html":
            continue
        relative = path.relative_to(source)
        _copy_atomic(path, target / relative, target)
        if relative.parts[0] == "assets":
            current.add(relative.relative_to("assets"))
    # Новый вход публикуется только после всех его зависимостей, атомарной заменой.
    _copy_atomic(source / "index.html", target / "index.html", target)
    now = time.time() if now is None else now
    files = [path for path in assets.rglob("*") if path.is_file() and not path.is_symlink()]
    total = sum(path.stat().st_size for path in files)
    obsolete = sorted((path for path in files if path.relative_to(assets) not in current),
                      key=lambda path: path.stat().st_mtime)
    for path in obsolete:
        if not path.resolve().is_relative_to(assets.resolve()):
            raise ValueError("Путь очистки выходит за каталог assets")
        stat = path.stat()
        if stat.st_mtime < now - ttl or total > limit:
            path.unlink()
            total -= stat.st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Готовая сборка Vite")
    parser.add_argument("--target", type=Path, required=True, help="Каталог сайта на сервере")
    args = parser.parse_args()
    deploy_frontend(args.source, args.target)
    print("Frontend опубликован; недавние файлы открытых вкладок сохранены.")


if __name__ == "__main__":
    main()
