from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cd-code.yml"


def test_cd_workflow_exists() -> None:
    assert WORKFLOW.is_file()


def test_cd_workflow_does_not_deploy_infrastructure() -> None:
    text = WORKFLOW.read_text(encoding="utf-8").lower()
    assert "docker" not in text
    assert "ansible" not in text
    assert "compose" not in text
    assert any(tool in text for tool in ("rsync", "scp", "ssh"))


def test_cd_workflow_updates_frontend_api_and_migrations() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '"frontend/**"' in text
    assert "python frontend/build.py" in text
    assert "python -m alembic upgrade head" in text
    assert "systemctl restart '${TRAINING_UI_API_SERVICE}'" in text
    assert "curl --fail --silent --show-error '${TRAINING_UI_HEALTH_URL}'" in text


def test_cd_workflow_uses_secrets() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets.MLSYSTEM2_SERVER_HOST" in text
    assert "secrets.MLSYSTEM2_SERVER_USER" in text
    assert "secrets.MLSYSTEM2_SSH_KEY" in text
    assert "BEGIN OPENSSH PRIVATE KEY" not in text
    assert "BEGIN RSA PRIVATE KEY" not in text
