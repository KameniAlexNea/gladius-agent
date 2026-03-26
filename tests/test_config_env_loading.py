from __future__ import annotations

from pathlib import Path

from gladius.config import load_project_env


def test_load_project_env_loads_project_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GLADIUS_MODEL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("GLADIUS_MODEL=from_project_env\n", encoding="utf-8")

    loaded = load_project_env(tmp_path)

    assert loaded == env_path.resolve()
    assert __import__("os").environ.get("GLADIUS_MODEL") == "from_project_env"


def test_load_project_env_returns_none_when_missing(tmp_path: Path):
    loaded = load_project_env(tmp_path)
    assert loaded is None
