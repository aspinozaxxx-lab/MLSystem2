from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist"
REQUIRED = [
    SRC / "index.html",
    SRC / "app.js",
    SRC / "assets" / "app.css",
]


def main() -> None:
    for path in REQUIRED:
        if not path.is_file():
            raise SystemExit(f"Не найден файл фронта: {path}")
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(SRC, DIST)
    index = (DIST / "index.html").read_text(encoding="utf-8")
    script = (DIST / "app.js").read_text(encoding="utf-8")
    if "dashboard" in index.lower() or "grafana-frame" in script:
        raise SystemExit("На главной не должно быть старого dashboard")
    print(f"frontend build ok: {DIST}")


if __name__ == "__main__":
    main()

