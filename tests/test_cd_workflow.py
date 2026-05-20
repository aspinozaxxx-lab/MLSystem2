from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cd-code.yml"


def test_cd_workflow_exists() -> None:
    assert WORKFLOW.is_file()


def test_cd_workflow_is_code_copy_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8").lower()
    assert "docker" not in text
    assert "ansible" not in text
    assert "compose" not in text
    assert any(tool in text for tool in ("rsync", "scp", "ssh"))


def test_cd_workflow_uses_secrets() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets.MLSYSTEM2_SERVER_HOST" in text
    assert "secrets.MLSYSTEM2_SERVER_USER" in text
    assert "secrets.MLSYSTEM2_SSH_KEY" in text
    assert "BEGIN OPENSSH PRIVATE KEY" not in text
    assert "BEGIN RSA PRIVATE KEY" not in text

