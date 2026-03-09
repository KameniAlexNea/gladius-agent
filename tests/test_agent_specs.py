import asyncio
from unittest.mock import patch

from gladius.agents._base import _SUBAGENT_DEFINITIONS
from gladius.agents.specs.implementer_spec import (
    IMPLEMENTER_OUTPUT_SCHEMA,
    IMPLEMENTER_SYSTEM_PROMPT,
    build_implementer_prompt,
)
from gladius.agents.specs.planner_spec import build_planner_prompt
from gladius.agents.specs.validation_spec import build_validation_prompt


def test_build_planner_prompt_includes_parallel_instruction_without_leading_blank_block():
    prompt = build_planner_prompt(
        iteration=2,
        max_iterations=10,
        project_dir="/tmp/project",
        n_parallel=2,
    )

    assert "IMPORTANT: Generate exactly 2 independent approaches" in prompt
    assert "## Approach 1" in prompt
    assert "## Approach 2" in prompt


def test_build_implementer_prompt_includes_plan_and_metric():
    prompt = build_implementer_prompt(
        plan={
            "approach_summary": "Try catboost baseline",
            "plan_text": "1. Load data\n2. Train model",
        },
        target_metric="auc_roc",
    )

    assert "Try catboost baseline" in prompt
    assert "1. Load data" in prompt
    assert "auc_roc" in prompt


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


def test_implementer_schema_required_keys_stable():
    required = IMPLEMENTER_OUTPUT_SCHEMA["required"]
    assert "status" in required
    assert "oof_score" in required
    assert "quality_score" in required


# ── Coordinator-specific tests ────────────────────────────────────────────────


def test_implementer_system_prompt_is_coordinator_not_code_writer():
    """Coordinator prompt must describe routing, not direct code execution."""
    assert "coordinator" in IMPLEMENTER_SYSTEM_PROMPT.lower()
    assert "EXPERIMENT_STATE.json" in IMPLEMENTER_SYSTEM_PROMPT
    # Coordinator explicitly states it does NOT write code.
    assert "do not write code" in IMPLEMENTER_SYSTEM_PROMPT.lower()
    # Must NOT contain old monolithic-engineer patterns.
    assert "Skill(" not in IMPLEMENTER_SYSTEM_PROMPT
    assert "mcp__jupyter" not in IMPLEMENTER_SYSTEM_PROMPT


def test_implementer_system_prompt_describes_routing_graph():
    """All six subagent names and the routing keywords must be present."""
    for subagent in (
        "ml-scaffolder",
        "ml-developer",
        "ml-evaluator",
        "code-reviewer",
        "ml-scientist",
        "submission-builder",
    ):
        assert subagent in IMPLEMENTER_SYSTEM_PROMPT, (
            f"{subagent!r} missing from prompt"
        )
    for keyword in ("SCAFFOLD", "DEVELOP", "EVALUATE", "REVIEW", "SUBMIT"):
        assert keyword in IMPLEMENTER_SYSTEM_PROMPT, f"{keyword!r} missing from routing"


def test_implementer_agent_def_uses_agent_tool_not_bash():
    """Coordinator must use Agent() to delegate — never Bash, Edit, or Grep."""
    agent_def = _SUBAGENT_DEFINITIONS["implementer"]
    tool_str = " ".join(agent_def.tools)
    assert "Agent(" in tool_str
    assert "ml-scaffolder" in tool_str
    assert "ml-developer" in tool_str
    assert "ml-scientist" in tool_str
    assert "ml-evaluator" in tool_str
    assert "code-reviewer" in tool_str
    assert "submission-builder" in tool_str
    # These must NOT appear — coordinator should not run code directly.
    for forbidden in ("Bash", "Edit", "Grep", "Skill"):
        assert forbidden not in tool_str, (
            f"{forbidden!r} must not be in coordinator tools"
        )


def test_run_implementer_uses_bypassPermissions_no_workarounds(tmp_path):
    """The coordinator must use bypassPermissions (the default) with no
    disallowed_tools or can_use_tool workarounds — subagents get their tool
    access from their own AgentDefinition inside _SUBAGENT_DEFINITIONS."""
    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {"status": "success", "oof_score": 0.9, "quality_score": 80}, ""

    from gladius.agents import implementer as impl_module

    with patch.object(impl_module, "run_agent", side_effect=fake_run_agent):
        asyncio.run(
            impl_module.run_implementer(
                plan={"approach_summary": "test", "plan_text": "step 1"},
                state=type("S", (), {"target_metric": "f1"})(),
                project_dir=str(tmp_path),
            )
        )

    # Must NOT use disallowed_tools — it propagates to subagents.
    assert not captured.get("disallowed_tools"), (
        f"run_implementer must not use disallowed_tools. Got: {captured.get('disallowed_tools')}"
    )
    # Must NOT override permission_mode away from the default bypassPermissions.
    assert captured.get("permission_mode") in (
        None,
        "bypassPermissions",
    ), f"Expected bypassPermissions (or unset), got {captured.get('permission_mode')!r}"
    # Must NOT use a can_use_tool callback — that approach is fragile.
    assert captured.get("can_use_tool") is None, (
        "run_implementer must not use a can_use_tool workaround"
    )
