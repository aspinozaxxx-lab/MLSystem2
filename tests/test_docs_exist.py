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
REQUIRED_SECTIONS = [
    "Назначение",
    "Публичный интерфейс",
    "Публичные контракты",
    "Список используемых данным модулем модулей и с какой целью",
    "Алгоритм работы и его особенности",
]


def test_core_docs_exist() -> None:
    for name in ("architecture.md", "module_rules.md"):
        assert (ROOT / "docs" / name).is_file()


def test_module_docs_have_required_sections() -> None:
    for module in MODULES:
        path = ROOT / "docs" / "modules" / f"{module}_module.md"
        text = path.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            assert f"## {section}" in text, f"{path} не содержит раздел {section}"


def test_module_docs_have_only_required_sections() -> None:
    expected_headings = {f"## {section}" for section in REQUIRED_SECTIONS}
    for module in MODULES:
        path = ROOT / "docs" / "modules" / f"{module}_module.md"
        text = path.read_text(encoding="utf-8")
        headings = {line.strip() for line in text.splitlines() if line.startswith("## ")}
        assert headings == expected_headings
