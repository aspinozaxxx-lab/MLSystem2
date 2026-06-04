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
ALLOWED_MODULE_IMPORTS = {
    "training_ui_api": {"fastapi", "uvicorn"},
}


def test_forbidden_imports_are_absent() -> None:
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    _assert_allowed_import(path, root, alias.name)
                    _assert_no_cross_module_internal_import(path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                assert not any(alias.name == "*" for alias in node.names), f"{path}: звездочный импорт"
                if node.module is not None:
                    root = node.module.split(".", 1)[0]
                    _assert_allowed_import(path, root, node.module)
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


def test_training_ui_does_not_write_mlflow_metrics() -> None:
    forbidden_names = {
        "start_run",
        "log_dataset_preparation",
        "log_tile_preparation",
        "log_run_config",
        "log_training_epoch",
        "log_training_metrics",
        "log_training_artifacts",
        "log_timing_report",
        "log_pipeline_report",
        "end_run",
    }
    for path in (SRC / "training_ui_api").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "mlsystem2.mlflow_adapter.api":
                continue
            imported = {alias.name for alias in node.names}
            used_forbidden = sorted(imported & forbidden_names)
            assert not used_forbidden, f"{path}: training_ui_api не должен писать MLflow: {used_forbidden}"


def test_removed_storage_settings_class_is_absent() -> None:
    removed_class_name = "Storage" + "Settings"
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert removed_class_name not in text, f"{path}: найден удаленный класс настроек хранилища"


def _assert_no_cross_module_internal_import(path: Path, module: str) -> None:
    parts = module.split(".")
    if len(parts) < 3 or parts[0] != "mlsystem2":
        return
    current_top = path.relative_to(SRC).parts[0]
    imported_top = parts[1]
    if imported_top != current_top and any(part.startswith("_") for part in parts[2:]):
        raise AssertionError(f"{path}: импортирует приватный модуль из {imported_top}: {module}")


def _assert_allowed_import(path: Path, root: str, imported: str) -> None:
    if root not in FORBIDDEN_IMPORTS:
        return
    current_top = path.relative_to(SRC).parts[0]
    if root in ALLOWED_MODULE_IMPORTS.get(current_top, set()):
        return
    raise AssertionError(f"{path}: запрещенный импорт {imported}")


def _assert_no_cross_module_relative_internal_import(path: Path, node: ast.ImportFrom) -> None:
    if node.level < 2:
        return
    module_parts = [] if node.module is None else node.module.split(".")
    alias_parts = [alias.name for alias in node.names]
    if any(part.startswith("_") for part in [*module_parts, *alias_parts]):
        raise AssertionError(f"{path}: относительный импорт пересекает приватную реализацию")
