"""Tests for gladius.topologies._catalog and gladius.topologies.base.

Covers:
- TOPOLOGY_CATALOG contains all five expected topologies
- Each TopologyDefinition has non-empty name / style / flow / claude_md_section
- flow string matches the topology name in the frontmatter
- claude_md_section is the full body after the frontmatter (not just a line)
- IterationResult: default values, required fields, field mutation
- BaseTopology._budget_ok(): returns False when either guardrail fires
"""

from __future__ import annotations

import pytest

from gladius.topologies._catalog import TOPOLOGY_CATALOG, TopologyDefinition
from gladius.topologies.base import BaseTopology, IterationResult


EXPECTED_TOPOLOGIES = {"functional", "two-pizza", "platform", "autonomous", "matrix"}


class TestCatalogCompleteness:
    def test_all_five_topologies_present(self):
        assert EXPECTED_TOPOLOGIES <= set(TOPOLOGY_CATALOG.keys()), (
            f"Missing: {EXPECTED_TOPOLOGIES - set(TOPOLOGY_CATALOG.keys())}"
        )

    def test_extra_topologies_not_present(self):
        assert set(TOPOLOGY_CATALOG.keys()) == EXPECTED_TOPOLOGIES


class TestTopologyDefinitionFields:
    @pytest.mark.parametrize("name", sorted(EXPECTED_TOPOLOGIES))
    def test_name_matches_key(self, name):
        assert TOPOLOGY_CATALOG[name].name == name

    @pytest.mark.parametrize("name", sorted(EXPECTED_TOPOLOGIES))
    def test_style_non_empty(self, name):
        topo = TOPOLOGY_CATALOG[name]
        assert topo.style.strip(), f"{name}: style is empty"

    @pytest.mark.parametrize("name", sorted(EXPECTED_TOPOLOGIES))
    def test_flow_non_empty(self, name):
        topo = TOPOLOGY_CATALOG[name]
        assert topo.flow.strip(), f"{name}: flow is empty"

    @pytest.mark.parametrize("name", sorted(EXPECTED_TOPOLOGIES))
    def test_claude_md_section_is_substantial(self, name):
        topo = TOPOLOGY_CATALOG[name]
        # Should contain the Mermaid diagram at minimum
        assert len(topo.claude_md_section) > 200, (
            f"{name}: claude_md_section is suspiciously short ({len(topo.claude_md_section)} chars)"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_TOPOLOGIES))
    def test_claude_md_section_contains_mermaid(self, name):
        topo = TOPOLOGY_CATALOG[name]
        assert "mermaid" in topo.claude_md_section or "graph" in topo.claude_md_section


class TestTopologyFlowContent:
    def test_functional_flow_contains_key_agents(self):
        topo = TOPOLOGY_CATALOG["functional"]
        for agent in ("team-lead", "data-expert", "ml-engineer"):
            assert agent in topo.flow

    def test_autonomous_flow_mentions_parallel(self):
        topo = TOPOLOGY_CATALOG["autonomous"]
        assert "N" in topo.flow or "parallel" in topo.flow.lower() or "×" in topo.flow

    def test_matrix_flow_contains_review(self):
        topo = TOPOLOGY_CATALOG["matrix"]
        assert "review" in topo.flow.lower()


class TestTopologyImmutability:
    def test_definition_is_frozen(self):
        topo = TOPOLOGY_CATALOG["functional"]
        with pytest.raises((AttributeError, TypeError)):
            topo.name = "mutated"  # type: ignore[misc]


class TestIterationResult:
    def test_default_status_is_success(self):
        r = IterationResult(status="success")
        assert r.status == "success"

    def test_scores_default_to_none_zero(self):
        r = IterationResult(status="success")
        assert r.oof_score is None
        assert r.quality_score is None

    def test_is_improvement_defaults_false(self):
        r = IterationResult(status="success")
        assert r.is_improvement is False

    def test_submit_defaults_false(self):
        r = IterationResult(status="success")
        assert r.submit is False

    def test_stop_defaults_false(self):
        r = IterationResult(status="success")
        assert r.stop is False

    def test_solution_files_independent(self):
        a = IterationResult(status="success")
        b = IterationResult(status="success")
        a.solution_files.append("src/models.py")
        assert b.solution_files == []

    def test_error_result(self):
        r = IterationResult(status="error", error_message="OOM")
        assert r.status == "error"
        assert r.error_message == "OOM"


class TestBudgetOk:
    """BaseTopology._budget_ok via a concrete stub."""

    class _StubTopology(BaseTopology):
        async def run_iteration(self, state, project_dir, platform, **kwargs):
            return IterationResult(status="success")

    def setup_method(self):
        self.topo = self._StubTopology()

    def test_no_guardrails_always_true(self):
        assert self.topo._budget_ok("label") is True

    def test_check_budget_false_returns_false(self):
        assert self.topo._budget_ok("label", check_budget=lambda: False) is False

    def test_consume_agent_call_false_returns_false(self):
        assert self.topo._budget_ok("label", consume_agent_call=lambda l: False) is False

    def test_both_guardrails_true_returns_true(self):
        assert (
            self.topo._budget_ok(
                "label",
                consume_agent_call=lambda l: True,
                check_budget=lambda: True,
            )
            is True
        )

    def test_check_budget_takes_priority(self):
        """Even if consume_agent_call would pass, a False check_budget wins."""
        assert (
            self.topo._budget_ok(
                "label",
                consume_agent_call=lambda l: True,
                check_budget=lambda: False,
            )
            is False
        )
