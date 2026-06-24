from __future__ import annotations

import json
import os
import sys
from pathlib import Path


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRONTEND_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["MLSYSTEM2_TRAINING_UI_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA"] = ""
os.environ["MLSYSTEM2_TRAINING_UI_WORKER_ENABLED"] = "false"

from mlsystem2.training_ui_api.api import create_app  # noqa: E402


def main() -> None:
    schema = create_app().openapi()
    (FRONTEND_ROOT / "openapi.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
