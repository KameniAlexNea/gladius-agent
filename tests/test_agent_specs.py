"""Tests for role-based agent specs and catalog (Phase 3 refactoring)."""

from gladius.agents.roles.catalog import ROLE_CATALOG
from gladius.agents.roles.specs import (
    ITERATION_RESULT_SCHEMA,
    MEMORY_KEEPER_OUTPUT_SCHEMA,
    VALIDATOR_OUTPUT_SCHEMA,
    build_memory_keeper_prompt,
    build_team_lead_prompt,
    build_validator_prompt,
)
from gladius.agents._agent_defs import SUBAGENT_DEFINITIONS

_REQUIRED_ROLES = {
    "team-lead",
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "domain-expert",
    "evaluator",
    "validator",
    "memory-keeper",
}


# ── Role catalog ──────────────────────────────────────────────────────────────


def test_role_catalog_has_all_required_roles():
    assert _REQUIRED_ROLES.issubset(ROLE_CATALOG.keys()), (
        f"Missing roles: {_REQUIRED_ROLES - ROLE_CATALOG.keys()}"
    )


def test_role_catalog_roles_have_nonempty_descriptions():
    for name, role in ROLE_CATALOG.items():
        assert role.description, f"Role {name!r} has empty description"
        assert role.system_prompt, f"Role {name!r} has empty system_prompt"


def test_role_catalog_roles_registered_in_subagent_definitions():
    """All roles from ROLE_CATALOG must be present in SUBAGENT_DEFINITIONS."""
    missing = _REQUIRED_ROLES - SUBAGENT_DEFINITIONS.keys()
    assert not missing, f"Roles missing from SUBAGENT_DEFINITIONS: {missing}"


def test_team_lead_is_plan_mode():
    assert ROLE_CATALOG["team-lead"].is_plan_mode is True


def test_non_plan_mode_roles_have_tools():
    for name, role in ROLE_CATALOG.items():
        if not role.is_plan_mode:
            assert role.tools, f"Role {name!r} is not plan-mode but has no tools"


# ── build_team_lead_prompt ────────────────────────────────────────────────────


def test_build_team_lead_prompt_includes_parallel_instruction():
    prompt = build_team_lead_prompt(
        iteration=2,
        max_iterations=10,
        project_dir="/tmp/project",
        n_parallel=2,
    )
    assert "IMPORTANT: Generate exactly 2 independent approaches" in prompt
    assert "## Approach 1" in prompt
    assert "## Approach 2" in prompt


def test_build_team_lead_prompt_no_parallel_section_when_n1():
    prompt = build_team_lead_prompt(
        iteration=1,
        max_iterations=5,
        project_dir="/tmp/project",
        n_parallel=1,
    )
    assert "Approach 1" not in prompt
    assert "independent approaches" not in prompt


def test_build_team_lead_prompt_mentions_skill_discovery():
    prompt = build_team_lead_prompt(
        iteration=1,
        max_iterations=3,
        project_dir="/tmp/project",
        n_parallel=1,
    )
    assert "mcp__skills-on-demand__search_skills" in prompt


# ── ITERATION_RESULT_SCHEMA ───────────────────────────────────────────────────


def test_iteration_result_schema_required_keys_stable():
    required = ITERATION_RESULT_SCHEMA["required"]
    assert "status" in required
    assert "oof_score" in required
    assert "quality_score" in required


def test_iteration_result_schema_status_enum():
    status_prop = ITERATION_RESULT_SCHEMA["properties"]["status"]
    assert "success" in status_prop["enum"]
    assert "error" in status_prop["enum"]


# ── build_validator_prompt ────────────────────────────────────────────────────


def test_build_validator_prompt_metric_mode_contains_score_info():
    prompt = build_validator_prompt(
        oof_score=0.8123,
        quality_score=None,
        best_oof_score=0.801,
        best_quality_score=None,
        submission_path="submission.csv",
        target_metric="auc_roc",
        metric_direction="maximize",
        submission_quota_remaining=4,
        project_dir="/tmp/project",
    )
    assert "0.8123" in prompt
    assert "auc_roc" in prompt
    assert "submission.csv" in prompt


def test_build_validator_prompt_open_ended_mode():
    prompt = build_validator_prompt(
        oof_score=None,
        quality_score=75,
        best_oof_score=None,
        best_quality_score=60,
        submission_path=None,
        target_metric=None,
        metric_direction=None,
        submission_quota_remaining=0,
        project_dir="/tmp/project",
    )
    assert "75" in prompt
    assert "60" in prompt


# ── VALIDATOR_OUTPUT_SCHEMA / MEMORY_KEEPER_OUTPUT_SCHEMA ────────────────────


def test_validator_output_schema_required_keys():
    required = VALIDATOR_OUTPUT_SCHEMA["required"]
    assert "is_improvement" in required
    assert "submit" in required
    assert "stop" in required


def test_memory_keeper_schema_required_keys():
    required = MEMORY_KEEPER_OUTPUT_SCHEMA["required"]
    assert "summary" in required
    assert "memory_content" in required


# ── build_memory_keeper_prompt ────────────────────────────────────────────────


def test_build_memory_keeper_prompt_includes_competition_info():
    prompt = build_memory_keeper_prompt(
        iteration=3,
        competition_id="test-comp",
        target_metric="auc_roc",
        metric_direction="maximize",
        experiments=[{"iteration": 2, "oof_score": 0.85}],
        failed_runs=[],
        latest_result={"status": "success", "oof_score": 0.87},
        validator_notes="improvement detected",
    )
    assert "test-comp" in prompt
    assert "auc_roc" in prompt
    assert "0.87" in prompt
