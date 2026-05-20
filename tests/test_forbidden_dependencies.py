from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mlsystem2"
FORBIDDEN_IMPORTS = {
    "fastapi",
    "uvicorn",
    "airflow",
    "aio_pika",
    "pika",
    "tritonclient",
    "prometheus_client",
}


def test_forbidden_imports_are_absent() -> None:
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root not in FORBIDDEN_IMPORTS, f"{path}: запрещенный импорт {alias.name}"
                    _assert_no_cross_module_internal_import(path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                assert not any(alias.name == "*" for alias in node.names), f"{path}: звездочный импорт"
                if node.module is not None:
                    root = node.module.split(".", 1)[0]
                    assert root not in FORBIDDEN_IMPORTS, f"{path}: запрещенный импорт {node.module}"
                    _assert_no_cross_module_internal_import(path, node.module)
                _assert_no_cross_module_relative_internal_import(path, node)


def test_mlflow_imports_are_adapter_only() -> None:
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names.append(node.module)
            for name in names:
                if name.split(".", 1)[0] == "mlflow":
                    assert "mlflow_adapter" in path.relative_to(SRC).parts, f"{path}: импорт MLflow"


def _assert_no_cross_module_internal_import(path: Path, module: str) -> None:
    parts = module.split(".")
    if len(parts) < 3 or parts[0] != "mlsystem2":
        return
    current_top = path.relative_to(SRC).parts[0]
    imported_top = parts[1]
    if imported_top != current_top and any(part.startswith("_") for part in parts[2:]):
        raise AssertionError(f"{path}: импортирует приватный модуль из {imported_top}: {module}")


def _assert_no_cross_module_relative_internal_import(path: Path, node: ast.ImportFrom) -> None:
    if node.level < 2:
        return
    module_parts = [] if node.module is None else node.module.split(".")
    alias_parts = [alias.name for alias in node.names]
    if any(part.startswith("_") for part in [*module_parts, *alias_parts]):
        raise AssertionError(f"{path}: относительный импорт пересекает приватную реализацию")
