"""
Two-pizza topology — Amazon-style small cross-functional team.

Architecture:
  team-lead produces a SINGLE plan.
  One cross-functional agent (owns data + features + model + evaluation) executes
  the plan end-to-end.  The team is intentionally small (≤ 6 agents including all
  specialists) so every member has full ownership.

  team-lead → full-stack-ml-agent → validator → memory-keeper

The full-stack agent has access to all specialist sub-agents but coordinates
them itself rather than through a separate coordinator layer.
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
    _build_platform_mcp,
    _first_nonblank_line,
    _mcp_servers,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState


_COORDINATOR_PROMPT = """\
You are the full-stack ML engineer on a two-pizza team.

Your team is intentionally small — you own the whole experiment: data loading,
feature engineering, model training, and evaluation.

Coordinate your specialist colleagues when useful:
  data-expert       → initial data understanding and scaffold
  feature-engineer  → feature creation
  ml-engineer       → model training and OOF evaluation
  evaluator         → score extraction and artifact verification

Full ownership rules:
- You decide the order and scope of work.
- You read EXPERIMENT_STATE.json after each specialist to gate the next step.
- If a specialist errors, decide locally whether to retry, skip, or fail.
- Report final results via StructuredOutput.

STRICT RULES:
- NEVER modify CLAUDE.md.
- Only write to .claude/EXPERIMENT_STATE.json — no other files directly.
- Read a file before rewriting it.
- Once StructuredOutput is emitted, stop immediately.
"""


def _build_full_stack_prompt(
    plan_text: str,
    state: "CompetitionState",
) -> str:
    return f"""\
Competition: {state.competition_id}
Metric: {state.target_metric or 'open-ended'}  Direction: {state.metric_direction or 'n/a'}
Data dir: {state.data_dir}
Best OOF so far: {state.best_oof_score}
Team size (two-pizza): 4 specialists + you

## Your plan this iteration
{plan_text}

You have full ownership. Coordinate your specialists in whatever order makes
sense for this plan. Report results via StructuredOutput when done.
"""


class TwoPizzaTopology(BaseTopology):
    """
    Amazon two-pizza: small cross-functional team with full ownership.

    One full-stack coordinator owns the whole experiment and coordinates
    a ≤4 specialist sub-team.
    """

    async def run_iteration(
        self,
        state: "CompetitionState",
        project_dir: str,
        platform: str,
        *,
        n_parallel: int = 1,
        max_turns: dict | None = None,
        consume_agent_call=None,
        check_budget=None,
    ) -> IterationResult:
        mt = max_turns or {}
        result = IterationResult(status="error")
        team_session_ids: dict[str, str] = dict(state.team_session_ids or {})
        mcp = _mcp_servers(project_dir)

        # ── 1. team-lead: plan ───────────────────────────────────────────────
        if not self._budget_ok("team-lead", consume_agent_call, check_budget):
            result.error_message = "budget exceeded before team-lead"
            return result

        role_lead = ROLE_CATALOG["team-lead"]
        plan_prompt = build_team_lead_prompt(
            iteration=state.iteration,
            max_iterations=state.max_iterations,
            project_dir=project_dir,
            n_parallel=1,  # two-pizza: single plan, full team owns it
        )
        try:
            plan_result, session_id = await run_agent(
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
        team_session_ids["team-lead"] = session_id
        result.plan_text = plan_text
        result.approach_summary = plan_result.get("approach_summary") or _first_nonblank_line(plan_text)

        plans_dir = Path(project_dir) / ".claude" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"iter-{state.iteration:02d}.md").write_text(plan_text)

        exp_state_path = Path(project_dir) / ".claude" / "EXPERIMENT_STATE.json"
        exp_state_path.parent.mkdir(parents=True, exist_ok=True)
        exp_state_path.write_text("{}")

        # ── 2. Full-stack agent executes the whole plan ──────────────────────
        if not self._budget_ok("full-stack-agent", consume_agent_call, check_budget):
            result.error_message = "budget exceeded before full-stack agent"
            result.team_session_ids = team_session_ids
            return result

        runtime_model = get_runtime_model()
        _agent_defs = _build_pipeline_agent_defs(runtime_model)

        full_stack_prompt = _build_full_stack_prompt(plan_text, state)
        specialists = ("data-expert", "feature-engineer", "ml-engineer", "evaluator")

        try:
            impl_result, impl_session = await run_agent(
                agent_name="two-pizza-agent",
                prompt=full_stack_prompt,
                system_prompt=_COORDINATOR_PROMPT,
                allowed_tools=[
                    f"Agent({','.join(specialists)})",
                    "Read",
                    "Write",
                    "Glob",
                    "TodoWrite",
                    "mcp__skills-on-demand__search_skills",
                ],
                output_schema=ITERATION_RESULT_SCHEMA,
                cwd=project_dir,
                mcp_servers=mcp,
                max_turns=mt.get("full_stack", 40),
            )
        except Exception as exc:
            logger.error(f"[two-pizza-agent] failed: {exc}", exc_info=True)
            result.error_message = str(exc)
            result.team_session_ids = team_session_ids
            return result

        team_session_ids["two-pizza-agent"] = impl_session
        result.oof_score = impl_result.get("oof_score")
        result.quality_score = float(impl_result.get("quality_score") or 0)
        result.solution_files = impl_result.get("solution_files") or []
        result.submission_file = impl_result.get("submission_file") or ""
        result.notes = impl_result.get("notes") or ""
        result.status = impl_result.get("status", "error")

        if result.status == "error":
            result.error_message = impl_result.get("error_message") or "unknown error"
            result.team_session_ids = team_session_ids
            return result

        # ── 3. validator ─────────────────────────────────────────────────────
        if not self._budget_ok("validator", consume_agent_call, check_budget):
            result.team_session_ids = team_session_ids
            return result

        _ft = FunctionalTopology()
        val_result = await _ft._run_validator(
            state=state,
            project_dir=project_dir,
            platform=platform,
            oof_score=result.oof_score,
            quality_score=result.quality_score,
            submission_path=result.submission_file or None,
            max_turns=max_turns,
        )
        result.is_improvement = val_result.get("is_improvement", False)
        result.submit = val_result.get("submit", False)
        result.format_ok = val_result.get("format_ok", True)
        result.stop = val_result.get("stop", False)
        result.next_directions = val_result.get("next_directions") or []
        validator_notes = val_result.get("reasoning", "")

        # ── 4. memory-keeper ─────────────────────────────────────────────────
        if self._budget_ok("memory-keeper", consume_agent_call, check_budget):
            mem_result = await _ft._run_memory_keeper(
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
                max_turns=max_turns,
            )
            result.memory_content = mem_result.get("memory_content")
            result.memory_summary = mem_result.get("summary")

        result.team_session_ids = team_session_ids
        return result
