from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "mlsystem2"
MODULES = {
    "cli",
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
}


def test_forbidden_directories_absent() -> None:
    for name in ("ansible", "deploy", "frontend", "monitoring"):
        assert not (ROOT / name).exists()


def test_forbidden_files_absent() -> None:
    for name in ("Dockerfile", "docker-compose.yml"):
        assert not (ROOT / name).exists()


def test_no_unapproved_top_level_package_modules() -> None:
    allowed_files = {"__init__.py", "_typing.py"}
    found = set()
    for path in PACKAGE_ROOT.iterdir():
        if path.name == "__pycache__":
            continue
        if path.is_dir():
            found.add(path.name)
        elif path.is_file() and path.name not in allowed_files:
            found.add(path.name)
    assert found == MODULES


def test_each_module_has_documentation() -> None:
    for module in MODULES:
        assert (ROOT / "docs" / "modules" / f"{module}_module.md").is_file()


def test_each_module_has_api() -> None:
    for module in MODULES:
        assert (PACKAGE_ROOT / module / "api.py").is_file() or module == "cli"


def test_each_non_cli_module_has_contracts() -> None:
    for module in MODULES - {"cli"}:
        assert (PACKAGE_ROOT / module / "contracts.py").is_file()

