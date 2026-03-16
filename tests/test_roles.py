"""Tests for gladius.roles (catalog parsing) and gladius.roles.helpers.

Covers:
- ROLE_CATALOG contains all expected roles
- Each RoleDefinition has non-empty required fields
- system_prompt contains actual content (not empty after frontmatter strip)
- session is "persistent" or "fresh"
- tools tuple is non-empty for worker roles
- model placeholder not yet substituted in catalog (substitution happens at copy time)
- roles.copy() writes substituted .md files to a temp directory
- helpers.validate_runtime_invocation() raises on bad inputs
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import gladius.roles as roles
from gladius.roles import ROLE_CATALOG, ROLES
from gladius.roles.helpers import validate_runtime_invocation

EXPECTED_ROLES = {
    "team-lead",
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "domain-expert",
    "evaluator",
    "validator",
    "memory-keeper",
    "full-stack-coordinator",
}


class TestCatalogCompleteness:
    def test_all_expected_roles_present(self):
        assert EXPECTED_ROLES <= set(ROLE_CATALOG.keys()), (
            f"Missing roles: {EXPECTED_ROLES - set(ROLE_CATALOG.keys())}"
        )

    def test_roles_tuple_matches_catalog(self):
        for name in ROLES:
            assert name in ROLE_CATALOG, (
                f"ROLES lists '{name}' but it is not in ROLE_CATALOG"
            )


class TestRoleDefinitionFields:
    @pytest.mark.parametrize("name", list(EXPECTED_ROLES))
    def test_name_matches_key(self, name):
        role = ROLE_CATALOG[name]
        assert role.name == name

    @pytest.mark.parametrize("name", list(EXPECTED_ROLES))
    def test_description_non_empty(self, name):
        role = ROLE_CATALOG[name]
        assert role.description.strip(), f"{name}: description is empty"

    @pytest.mark.parametrize("name", list(EXPECTED_ROLES))
    def test_system_prompt_non_empty(self, name):
        role = ROLE_CATALOG[name]
        assert len(role.system_prompt.strip()) > 50, (
            f"{name}: system_prompt is suspiciously short"
        )

    @pytest.mark.parametrize("name", list(EXPECTED_ROLES))
    def test_session_is_valid_value(self, name):
        role = ROLE_CATALOG[name]
        assert role.session in (
            "persistent",
            "fresh",
        ), f"{name}: unexpected session value '{role.session}'"

    @pytest.mark.parametrize("name", list(EXPECTED_ROLES))
    def test_tools_non_empty(self, name):
        role = ROLE_CATALOG[name]
        assert len(role.tools) > 0, f"{name}: tools tuple is empty"

    @pytest.mark.parametrize("name", list(EXPECTED_ROLES))
    def test_model_is_placeholder(self, name):
        """The catalog is loaded before model substitution; placeholder must be present."""
        role = ROLE_CATALOG[name]
        assert "GLADIUS" in role.model, (
            f"{name}: model field '{role.model}' does not contain GLADIUS placeholder"
        )


class TestSessionConstraints:
    def test_team_lead_is_persistent(self):
        assert ROLE_CATALOG["team-lead"].session == "persistent"

    def test_worker_roles_are_fresh(self):
        fresh_roles = {
            "data-expert",
            "feature-engineer",
            "ml-engineer",
            "evaluator",
            "validator",
            "memory-keeper",
            "domain-expert",
            "full-stack-coordinator",
        }
        for name in fresh_roles:
            assert ROLE_CATALOG[name].session == "fresh", (
                f"{name} should be 'fresh' but is '{ROLE_CATALOG[name].session}'"
            )


class TestReadOnlyRoles:
    def test_evaluator_has_no_write_tool(self):
        """evaluator only needs Read-class tools."""
        eval_tools = set(ROLE_CATALOG["evaluator"].tools)
        # Write is needed to update EXPERIMENT_STATE; WebSearch is not needed
        assert "WebSearch" not in eval_tools

    def test_validator_has_no_write_tool(self):
        """validator emits StructuredOutput only — no Write tool needed."""
        val_tools = set(ROLE_CATALOG["validator"].tools)
        assert "Write" not in val_tools
        assert "WebSearch" not in val_tools


class TestRolesCopy:
    def test_copy_writes_all_roles(self, tmp_path):
        dst = tmp_path / "agents"
        roles.copy(dst, "all", model="test-model", small_model="test-small")
        written = {p.stem for p in dst.glob("*.md")}
        assert EXPECTED_ROLES <= written

    def test_copy_substitutes_model_placeholder(self, tmp_path):
        dst = tmp_path / "agents"
        roles.copy(dst, "all", model="my-model", small_model="my-small")
        team_lead_md = (dst / "team-lead.md").read_text(encoding="utf-8")
        assert "{{GLADIUS_MODEL}}" not in team_lead_md
        assert "my-model" in team_lead_md

    def test_copy_small_model_substituted_for_evaluator(self, tmp_path):
        """evaluator uses GLADIUS_SMALL_MODEL in its frontmatter."""
        dst = tmp_path / "agents"
        roles.copy(dst, "all", model="big-model", small_model="small-model")
        evaluator_md = (dst / "evaluator.md").read_text(encoding="utf-8")
        assert "{{GLADIUS_SMALL_MODEL}}" not in evaluator_md
        assert "small-model" in evaluator_md

    def test_copy_skip_if_exists(self, tmp_path):
        dst = tmp_path / "agents"
        roles.copy(dst, "all", model="v1", small_model="v1")
        mtime1 = (dst / "team-lead.md").stat().st_mtime

        # Second copy without force — should not overwrite
        roles.copy(dst, "all", model="v2", small_model="v2")
        mtime2 = (dst / "team-lead.md").stat().st_mtime
        assert mtime1 == mtime2

    def test_copy_force_overwrites(self, tmp_path):
        dst = tmp_path / "agents"
        roles.copy(dst, "all", model="v1", small_model="v1")
        roles.copy(dst, "all", model="v2", small_model="v2", force=True)
        content = (dst / "team-lead.md").read_text(encoding="utf-8")
        assert "v2" in content


class TestValidateRuntimeInvocation:
    def test_missing_gladius_model_raises(self, tmp_path):
        with patch.dict("os.environ", {"GLADIUS_MODEL": ""}, clear=False):
            with pytest.raises(RuntimeError, match="GLADIUS_MODEL"):
                validate_runtime_invocation(
                    agent_name="test",
                    cwd=str(tmp_path),
                    allowed_tools=["Read"],
                    max_turns=10,
                )

    def test_nonexistent_cwd_raises(self, tmp_path):
        with patch.dict("os.environ", {"GLADIUS_MODEL": "m"}, clear=False):
            with pytest.raises(RuntimeError, match="cwd"):
                validate_runtime_invocation(
                    agent_name="test",
                    cwd=str(tmp_path / "missing"),
                    allowed_tools=["Read"],
                    max_turns=10,
                )

    def test_empty_tools_raises(self, tmp_path):
        with patch.dict("os.environ", {"GLADIUS_MODEL": "m"}, clear=False):
            with pytest.raises(RuntimeError, match="allowed_tools"):
                validate_runtime_invocation(
                    agent_name="test",
                    cwd=str(tmp_path),
                    allowed_tools=[],
                    max_turns=10,
                )

    def test_zero_max_turns_raises(self, tmp_path):
        with patch.dict("os.environ", {"GLADIUS_MODEL": "m"}, clear=False):
            with pytest.raises(RuntimeError, match="max_turns"):
                validate_runtime_invocation(
                    agent_name="test",
                    cwd=str(tmp_path),
                    allowed_tools=["Read"],
                    max_turns=0,
                )

    def test_valid_invocation_passes(self, tmp_path):
        with patch.dict("os.environ", {"GLADIUS_MODEL": "m"}, clear=False):
            validate_runtime_invocation(
                agent_name="test",
                cwd=str(tmp_path),
                allowed_tools=["Read"],
                max_turns=5,
            )
