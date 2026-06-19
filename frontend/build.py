from __future__ import annotations

from hashlib import sha256
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
    index_path = DIST / "index.html"
    script_path = DIST / "app.js"
    style_path = DIST / "assets" / "app.css"
    index = index_path.read_text(encoding="utf-8-sig")
    script = script_path.read_text(encoding="utf-8-sig")
    index = index.replace("./app.js", f"./app.js?v={_asset_hash(script_path)}")
    index = index.replace("./assets/app.css", f"./assets/app.css?v={_asset_hash(style_path)}")
    index_path.write_text(index, encoding="utf-8")
    if "dashboard" in index.lower() or "grafana-frame" in script:
        raise SystemExit("На главной не должно быть старого dashboard")
    print(f"frontend build ok: {DIST}")


def _asset_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    main()
