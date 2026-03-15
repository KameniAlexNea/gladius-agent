"""Tests for project setup helpers."""

import json
from pathlib import Path

from gladius.state import CompetitionState
from gladius.utils.project_setup import setup_project_dir

_SUBAGENT_NAMES = {
    "team-lead",
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "domain-expert",
    "evaluator",
    "validator",
    "memory-keeper",
}


def _state(tmp_path: Path) -> CompetitionState:
    return CompetitionState(
        competition_id="comp-1",
        data_dir=str((tmp_path / "data").resolve()),
        output_dir=str((tmp_path / ".gladius").resolve()),
        target_metric="auc_roc",
        metric_direction="maximize",
    )


def test_setup_writes_fake_mcp_server(tmp_path):
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    mcp_path = tmp_path / ".mcp.json"
    cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "fake-tools" in cfg["mcpServers"]
    cmd = cfg["mcpServers"]["fake-tools"]["args"][1]
    assert "fake_server" in cmd
    assert "fake_server.run()" in cmd


def test_setup_copies_all_subagents(tmp_path):
    """All six coordinator subagents must be written to .claude/agents/ on setup."""
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = tmp_path / ".claude" / "agents"
    written = {p.stem for p in agents_dir.glob("*.md")}
    assert _SUBAGENT_NAMES.issubset(written), (
        f"Missing subagents: {_SUBAGENT_NAMES - written}"
    )


def test_setup_subagents_idempotent(tmp_path):
    """Subagent files written on first setup are NOT overwritten on second setup."""
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = (
        tmp_path / ".clone" / "agents" if False else tmp_path / ".claude" / "agents"
    )
    # Corrupt one subagent file to verify it is preserved on re-run.
    sentinel = "# CUSTOM_SENTINEL\n"
    target = agents_dir / "ml-engineer.md"
    target.write_text(sentinel, encoding="utf-8")

    setup_project_dir(state, str(tmp_path), platform="fake")

    assert target.read_text(encoding="utf-8") == sentinel, (
        "ml-engineer.md was overwritten — subagents must be idempotent"
    )


def test_setup_all_agents_are_idempotent(tmp_path):
    """All agent templates are written idempotently — no file is overwritten on re-run."""
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = tmp_path / ".claude" / "agents"
    sentinel = "# CUSTOM_SENTINEL\n"
    for agent_path in agents_dir.glob("*.md"):
        agent_path.write_text(sentinel, encoding="utf-8")

    setup_project_dir(state, str(tmp_path), platform="fake")

    for agent_path in agents_dir.glob("*.md"):
        assert agent_path.read_text(encoding="utf-8") == sentinel, (
            f"{agent_path.name} was overwritten — all agents must be idempotent"
        )


def test_subagent_files_contain_experiment_state_reference(tmp_path):
    """Executing roles must reference EXPERIMENT_STATE.json (artifact protocol).
    team-lead (plan-only), validator (read-only), and memory-keeper (writes MEMORY.md)
    are exempt — they never touch EXPERIMENT_STATE.json by design.
    """
    _EXECUTOR_ROLES = {
        "data-expert",
        "feature-engineer",
        "ml-engineer",
        "domain-expert",
        "evaluator",
    }
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = tmp_path / ".claude" / "agents"
    missing = [
        name
        for name in _EXECUTOR_ROLES
        if "EXPERIMENT_STATE"
        not in (agents_dir / f"{name}.md").read_text(encoding="utf-8")
    ]
    assert not missing, f"Missing EXPERIMENT_STATE.json reference in: {missing}"


def test_subagent_small_model_default_is_inherit(tmp_path, monkeypatch):
    """Without GLADIUS_SMALL_MODEL set, model: inherit must appear for scaffolder/evaluator."""
    monkeypatch.delenv("GLADIUS_SMALL_MODEL", raising=False)
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = tmp_path / ".claude" / "agents"
    for name in ("evaluator", "memory-keeper", "validator"):
        content = (agents_dir / f"{name}.md").read_text(encoding="utf-8")
        assert "model: inherit" in content, (
            f"{name}.md should default to model: inherit"
        )
        assert "haiku" not in content, f"{name}.md must not contain hardcoded 'haiku'"


def test_subagent_small_model_env_var_substituted(tmp_path, monkeypatch):
    """GLADIUS_SMALL_MODEL env var is substituted into the small-model role templates."""
    monkeypatch.setenv("GLADIUS_SMALL_MODEL", "claude-haiku-4-5")
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = tmp_path / ".claude" / "agents"
    for name in ("evaluator", "validator", "memory-keeper"):
        content = (agents_dir / f"{name}.md").read_text(encoding="utf-8")
        assert "model: claude-haiku-4-5" in content, (
            f"{name}.md should contain the substituted model name"
        )
        assert "{{GLADIUS_SMALL_MODEL}}" not in content, (
            f"{name}.md must not contain the raw placeholder"
        )


def test_subagent_no_tilde_claude_path(tmp_path):
    """No subagent template should reference ~/.claude — only local .claude/ paths."""
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    agents_dir = tmp_path / ".claude" / "agents"
    offenders = [
        name
        for name in _SUBAGENT_NAMES
        if "~/.claude" in (agents_dir / f"{name}.md").read_text(encoding="utf-8")
    ]
    assert not offenders, f"Templates must not reference ~/.claude: {offenders}"
