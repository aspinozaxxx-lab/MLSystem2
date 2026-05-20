from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "settings",
    "storage",
    "dataset_preparing",
    "tile_preparation",
    "models",
    "metrics",
    "train",
    "train_pipeline",
    "inference",
    "inference_pipeline",
    "mlflow_adapter",
    "cli",
]
REQUIRED_SECTIONS = ["Назначение", "Публичный интерфейс", "Контракты", "Запрещенные пересечения"]


def test_core_docs_exist() -> None:
    for name in ("architecture.md", "module_rules.md", "codex_rules.md"):
        assert (ROOT / "docs" / name).is_file()


def test_module_docs_have_required_sections() -> None:
    for module in MODULES:
        path = ROOT / "docs" / "modules" / f"{module}_module.md"
        text = path.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{path} не содержит раздел {section}"
