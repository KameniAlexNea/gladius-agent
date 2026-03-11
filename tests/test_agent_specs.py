"""Tests for the single-agent (solver) specification."""

import asyncio
from unittest.mock import patch

from gladius.agents._base import _SUBAGENT_DEFINITIONS
from gladius.agents.specs.solver_spec import (
    SOLVER_OUTPUT_SCHEMA,
    SOLVER_SYSTEM_PROMPT,
    build_solver_prompt,
)
from gladius.agents.specs.validation_spec import build_validation_prompt


# ── Solver prompt tests ───────────────────────────────────────────────────────


def test_build_solver_prompt_includes_metric():
    prompt = build_solver_prompt(target_metric="auc_roc")
    assert "auc_roc" in prompt
    assert "oof_score" in prompt


def test_build_solver_prompt_open_ended():
    prompt = build_solver_prompt(target_metric=None)
    assert "oof_score = null" in prompt
    assert "quality" in prompt.lower()


def test_solver_system_prompt_mandates_skills_first():
    assert "MANDATORY FIRST ACTION" in SOLVER_SYSTEM_PROMPT
    assert "mcp__skills-on-demand__search_skills" in SOLVER_SYSTEM_PROMPT


def test_solver_system_prompt_continuous_improvement():
    assert "Continuous improvement" in SOLVER_SYSTEM_PROMPT
    assert "search" in SOLVER_SYSTEM_PROMPT.lower()
    assert "iterate" in SOLVER_SYSTEM_PROMPT.lower()


def test_solver_system_prompt_has_no_subagent_instructions():
    """Single-agent: no coordinator pattern, no Agent() calls."""
    assert "coordinator" not in SOLVER_SYSTEM_PROMPT.lower()
    assert "Agent(" not in SOLVER_SYSTEM_PROMPT
    assert "ml-scaffolder" not in SOLVER_SYSTEM_PROMPT
    assert "ml-developer" not in SOLVER_SYSTEM_PROMPT


# ── Solver schema tests ───────────────────────────────────────────────────────


def test_solver_schema_required_keys_stable():
    required = SOLVER_OUTPUT_SCHEMA["required"]
    assert "status" in required
    assert "oof_score" in required
    assert "quality_score" in required


def test_solver_schema_status_enum():
    statuses = SOLVER_OUTPUT_SCHEMA["properties"]["status"]["enum"]
    assert "success" in statuses
    assert "error" in statuses


# ── SUBAGENT_DEFINITIONS registry tests ──────────────────────────────────────


def test_registry_has_solver_not_worker_subagents():
    """Single-agent architecture: registry must contain solver, NOT the old 6 workers."""
    assert "solver" in _SUBAGENT_DEFINITIONS
    for removed in (
        "planner",
        "implementer",
        "ml-scaffolder",
        "ml-developer",
        "ml-scientist",
        "ml-evaluator",
        "code-reviewer",
        "submission-builder",
    ):
        assert removed not in _SUBAGENT_DEFINITIONS, (
            f"{removed!r} should not be in registry after single-agent refactor"
        )


def test_solver_agent_def_has_all_implementation_tools():
    agent_def = _SUBAGENT_DEFINITIONS["solver"]
    tools = agent_def.tools
    for required in ("Read", "Write", "Edit", "Bash", "Grep", "TodoWrite"):
        assert required in tools, f"{required!r} missing from solver tools"


def test_solver_agent_def_has_no_agent_tool():
    """Solver must NOT spawn subagents — no Agent() in its tool list."""
    agent_def = _SUBAGENT_DEFINITIONS["solver"]
    for tool in agent_def.tools:
        assert not tool.startswith("Agent("), (
            f"Solver must not have Agent() tool; found {tool!r}"
        )


def test_run_solver_uses_bypassPermissions(tmp_path):
    """run_solver must call run_agent with default (bypassPermissions) permission mode."""
    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {"status": "success", "oof_score": 0.9, "quality_score": 80}, ""

    from gladius.agents import solver as solver_module

    with patch.object(solver_module, "run_agent", side_effect=fake_run_agent):
        asyncio.run(
            solver_module.run_solver(
                state=type("S", (), {"target_metric": "f1"})(),
                project_dir=str(tmp_path),
            )
        )

    # Must NOT use disallowed_tools
    assert not captured.get("disallowed_tools")
    # Must NOT override permission_mode away from bypassPermissions
    assert captured.get("permission_mode") in (None, "bypassPermissions")
    # Must NOT use can_use_tool callback
    assert captured.get("can_use_tool") is None


# ── Validation prompt (unchanged) ────────────────────────────────────────────


def test_validation_prompt_metric_mode_contains_submission_checks():
    prompt = build_validation_prompt(
        solution_path="solution.py",
        oof_score=0.8123,
        quality_score=0,
        submission_path="submission.csv",
        target_metric="auc_roc",
        metric_direction="maximize",
        best_oof_score=0.801,
        best_quality_score=None,
        submission_count=1,
        max_submissions_per_day=5,
        quota_instruction="",
        project_dir="/tmp/project",
    )

    assert "OOF score     : 0.812300" in prompt
    assert "Use Read to open submission.csv" in prompt
