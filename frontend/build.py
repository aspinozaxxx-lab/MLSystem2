from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def run_frontend_build() -> None:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    subprocess.run([npm, "run", "build"], cwd=ROOT, check=True)


def main() -> None:
    if not (ROOT / "package.json").is_file():
        raise SystemExit(f"Не найден package.json фронта: {ROOT / 'package.json'}")
    run_frontend_build()
    index_path = DIST / "index.html"
    if not index_path.is_file():
        raise SystemExit(f"Vite build не создал {index_path}")
    index = index_path.read_text(encoding="utf-8")
    if "dashboard" in index.lower() or "grafana-frame" in index:
        raise SystemExit("На главной не должно быть старого dashboard")
    print(f"frontend build ok: {DIST}")


if __name__ == "__main__":
    main()
