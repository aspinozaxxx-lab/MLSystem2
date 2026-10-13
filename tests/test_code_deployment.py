"""Доставка интерфейса не должна прерывать работу исполнителя очереди."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "cd-code.yml"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def _restart_decision(root: Path, previous: str, current: str) -> bool:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["copy-code"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == "Скопировать код репозитория")
    function = re.search(r"(?ms)^worker_restart_required\(\) \{.*?^\}", script)
    assert function is not None
    bash = shutil.which("bash")
    git_bash = Path(shutil.which("git") or "").parent.parent / "bin" / "bash.exe"
    if git_bash.is_file():
        bash = str(git_bash)
    if bash is None:
        pytest.skip("Оболочка Bash недоступна")
    result = subprocess.run(
        [bash, "-c", function.group() + '\nworker_restart_required "$1" "$2"', "проверка", previous, current],
        cwd=root, check=False, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode in {0, 1}, result.stderr
    return result.returncode == 0


def _repository(root: Path) -> str:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Проверка доставки")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "README.md").write_text("Начальная версия", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "Начальное состояние")
    return _git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize(("path", "expected"), [
    ("frontend/src/App.tsx", False),
    ("frontend/public/news/page.svg", False),
    ("docs/architecture.md", False),
    ("tests/test_example.py", False),
    ("src/mlsystem2/training_ui_api/_dataset_catalog.py", True),
    ("src/mlsystem2/train/_runner.py", True),
    ("configs/settings.server.yaml", True),
    ("migrations/versions/example.py", True),
    ("pyproject.toml", True),
    ("alembic.ini", True),
    ("requirements.txt", True),
])
def test_worker_restart_depends_on_runtime_changes(tmp_path: Path, path: str, expected: bool):
    previous = _repository(tmp_path)
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Новая версия", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "Изменение")
    assert _restart_decision(tmp_path, previous, _git(tmp_path, "rev-parse", "HEAD")) is expected


@pytest.mark.parametrize("previous", ["", "неизвестно", "0" * 40])
def test_unknown_deployed_revision_requires_safe_restart(tmp_path: Path, previous: str):
    current = _repository(tmp_path)
    assert _restart_decision(tmp_path, previous, current)
    assert not _restart_decision(tmp_path, current, current)


def test_runtime_removal_also_requires_restart(tmp_path: Path):
    _repository(tmp_path)
    target = tmp_path / "src" / "mlsystem2" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("Прежний код", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "Исполнитель")
    previous = _git(tmp_path, "rev-parse", "HEAD")
    target.unlink()
    _git(tmp_path, "add", "-u")
    _git(tmp_path, "commit", "-qm", "Удаление")
    assert _restart_decision(tmp_path, previous, _git(tmp_path, "rev-parse", "HEAD"))


def test_workflow_reads_previous_revision_before_copy_and_guards_restart():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["copy-code"]["steps"]
    assert steps[0]["with"]["fetch-depth"] == 0
    script = next(step["run"] for step in steps if step.get("name") == "Скопировать код репозитория")
    assert script.index("PREVIOUS_COMMIT=") < script.index("rsync -az --delete")
    assert 'if [ "${WORKER_RESTART_REQUIRED}" = true ]; then' in script
