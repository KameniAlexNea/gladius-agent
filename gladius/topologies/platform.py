"""
Platform topology — Google/Amazon-style platform-based architecture.

Architecture:
  A platform layer (data-expert + evaluator) sets up shared infrastructure
  (project scaffold, data loading, evaluation harness) that all product-layer
  experiment agents consume.

  team-lead → platform-layer (data-expert, evaluator)
           → product-layer (feature-engineer, ml-engineer)
           → validator → memory-keeper

The platform layer exposes artefacts (src/, artifacts/, train.log) as
"internal APIs".  Product agents consume these without duplicating
infrastructure work.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gladius.roles import ROLE_CATALOG
from gladius.roles.agent_runner import run_agent
from gladius.roles.helpers import get_runtime_model
from gladius.roles.specs import (
    ITERATION_RESULT_SCHEMA,
    TEAM_LEAD_OUTPUT_SCHEMA,
    build_team_lead_prompt,
)
from gladius.topologies.base import BaseTopology, IterationResult
from gladius.topologies.functional import (
    FunctionalTopology,
    _build_pipeline_agent_defs,
    _first_nonblank_line,
    _mcp_servers,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState


def _build_platform_prompt(plan_text: str, state: "CompetitionState") -> str:
    return f"""\
Competition: {state.competition_id}
Metric: {state.target_metric or 'open-ended'}  Direction: {state.metric_direction or 'n/a'}
Data dir: {state.data_dir}

## Plan (provision infrastructure for this approach)
{plan_text}

Provision the platform layer: scaffold src/, validate data loading, confirm
the OOF contract.  Write platform status to EXPERIMENT_STATE.json.
Emit StructuredOutput when ready.
"""


def _build_product_prompt(plan_text: str, state: "CompetitionState") -> str:
    return f"""\
Competition: {state.competition_id}
Metric: {state.target_metric or 'open-ended'}  Direction: {state.metric_direction or 'n/a'}
Data dir: {state.data_dir}
Best OOF so far: {state.best_oof_score}

## Plan (implement the experiment using platform infrastructure)
{plan_text}

Platform layer is ready. Implement the ML experiment on top of it.
Report final results via StructuredOutput.
"""


class PlatformTopology(BaseTopology):
    """
    Google/Amazon-style platform architecture.

    Platform agents provide shared infra; product agents do the experiment.
    """

    async def run_iteration(
        self,
        state: "CompetitionState",
        project_dir: str,
        platform: str,
        *,
        n_parallel: int = 1,
        consume_agent_call=None,
        check_budget=None,
    ) -> IterationResult:
        result = IterationResult(status="error")
        team_session_ids: dict[str, str] = dict(state.team_session_ids or {})
        mcp = _mcp_servers(project_dir)
        ft = FunctionalTopology()

        # ── 1. team-lead: plan ───────────────────────────────────────────────
        if not self._budget_ok("team-lead", consume_agent_call, check_budget):
            result.error_message = "budget exceeded before team-lead"
            return result

        role_lead = ROLE_CATALOG["team-lead"]
        plan_prompt = build_team_lead_prompt(
            iteration=state.iteration,
            max_iterations=state.max_iterations,
            project_dir=project_dir,
            n_parallel=1,
        )
        try:
            plan_result, lead_sid = await run_agent(
                agent_name="team-lead",
                prompt=plan_prompt,
                system_prompt=role_lead.system_prompt,
                allowed_tools=list(role_lead.tools),
                output_schema=TEAM_LEAD_OUTPUT_SCHEMA,
                cwd=project_dir,
                resume=team_session_ids.get("team-lead"),
                mcp_servers=mcp,
            )
        except Exception as exc:
            logger.error(f"[team-lead] failed: {exc}", exc_info=True)
            result.error_message = str(exc)
            return result

        plan_text = plan_result["plan"]
        team_session_ids["team-lead"] = lead_sid
        result.plan_text = plan_text
        result.approach_summary = plan_result.get("approach_summary") or _first_nonblank_line(plan_text)

        plans_dir = Path(project_dir) / ".claude" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"iter-{state.iteration:02d}.md").write_text(plan_text)

        exp_state_path = Path(project_dir) / ".claude" / "EXPERIMENT_STATE.json"
        exp_state_path.parent.mkdir(parents=True, exist_ok=True)
        exp_state_path.write_text("{}")

        runtime_model = get_runtime_model()
        agent_defs_all = _build_pipeline_agent_defs(runtime_model)

        # ── 2. Platform layer ────────────────────────────────────────────────
        if not self._budget_ok("platform-layer", consume_agent_call, check_budget):
            result.error_message = "budget exceeded before platform layer"
            result.team_session_ids = team_session_ids
            return result

        _PLATFORM_SCHEMA = {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"type": "string"},
                "notes": {"type": "string"},
                "error_message": {"type": "string"},
            },
            "additionalProperties": True,
        }

        try:
            plat_result, plat_sid = await run_agent(
                agent_name="platform-layer",
                prompt=_build_platform_prompt(plan_text, state),
                system_prompt=ROLE_CATALOG["platform-layer"].system_prompt,
                allowed_tools=[
                    "Agent(data-expert,evaluator)",
                    "Read",
                    "Write",
                    "Glob",
                    "TodoWrite",
                    "mcp__skills-on-demand__search_skills",
                ],
                output_schema=_PLATFORM_SCHEMA,
                cwd=project_dir,
                mcp_servers=mcp,
                max_turns=20,
            )
        except Exception as exc:
            logger.error(f"[platform-layer] failed: {exc}", exc_info=True)
            result.error_message = f"platform-layer: {exc}"
            result.team_session_ids = team_session_ids
            return result

        team_session_ids["platform-layer"] = plat_sid
        if plat_result.get("status") == "error":
            result.error_message = plat_result.get("error_message") or "platform error"
            result.team_session_ids = team_session_ids
            return result

        # ── 3. Product layer ─────────────────────────────────────────────────
        if not self._budget_ok("product-layer", consume_agent_call, check_budget):
            result.error_message = "budget exceeded before product layer"
            result.team_session_ids = team_session_ids
            return result

        try:
            prod_result, prod_sid = await run_agent(
                agent_name="product-layer",
                prompt=_build_product_prompt(plan_text, state),
                system_prompt=ROLE_CATALOG["product-layer"].system_prompt,
                allowed_tools=[
                    "Agent(feature-engineer,ml-engineer)",
                    "Read",
                    "Write",
                    "Glob",
                    "TodoWrite",
                    "mcp__skills-on-demand__search_skills",
                ],
                output_schema=ITERATION_RESULT_SCHEMA,
                cwd=project_dir,
                mcp_servers=mcp,
                max_turns=40,
            )
        except Exception as exc:
            logger.error(f"[product-layer] failed: {exc}", exc_info=True)
            result.error_message = f"product-layer: {exc}"
            result.team_session_ids = team_session_ids
            return result

        team_session_ids["product-layer"] = prod_sid
        result.oof_score = prod_result.get("oof_score")
        result.quality_score = float(prod_result.get("quality_score") or 0)
        result.solution_files = prod_result.get("solution_files") or []
        result.submission_file = prod_result.get("submission_file") or ""
        result.notes = prod_result.get("notes") or ""
        result.status = prod_result.get("status", "error")

        if result.status == "error":
            result.error_message = prod_result.get("error_message") or "product error"
            result.team_session_ids = team_session_ids
            return result

        # ── 4. validator & memory-keeper ─────────────────────────────────────
        if self._budget_ok("validator", consume_agent_call, check_budget):
            val_result = await ft._run_validator(
                state=state,
                project_dir=project_dir,
                platform=platform,
                oof_score=result.oof_score,
                quality_score=result.quality_score,
                submission_path=result.submission_file or None,
            )
            result.is_improvement = val_result.get("is_improvement", False)
            result.submit = val_result.get("submit", False)
            result.format_ok = val_result.get("format_ok", True)
            result.stop = val_result.get("stop", False)
            result.next_directions = val_result.get("next_directions") or []
            validator_notes = val_result.get("reasoning", "")

            if self._budget_ok("memory-keeper", consume_agent_call, check_budget):
                mem_result = await ft._run_memory_keeper(
                    state=state,
                    project_dir=project_dir,
                    latest_result={
                        "status": result.status,
                        "oof_score": result.oof_score,
                        "quality_score": result.quality_score,
                        "notes": result.notes,
                        "approach_summary": result.approach_summary,
                    },
                    validator_notes=validator_notes,
                )
                result.memory_content = mem_result.get("memory_content")
                result.memory_summary = mem_result.get("summary")

        result.team_session_ids = team_session_ids
        return result
