"""
Functional topology — Apple-style deep-expertise pipeline.

Sequential role pipeline:
  team-lead → data-expert → feature-engineer → ml-engineer
  → evaluator → validator → memory-keeper

team-lead plans the iteration; MiniTeamTopology executes the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gladius.roles import ROLE_CATALOG
from gladius.roles.agent_runner import run_agent
from gladius.roles.specs import (
    TEAM_LEAD_OUTPUT_SCHEMA,
    build_team_lead_prompt,
)
from gladius.topologies.base import IterationResult
from gladius.topologies.mini_team import MiniTeamTopology, _build_platform_mcp, _mcp_servers  # re-export for sibling topologies

if TYPE_CHECKING:
    from gladius.state import CompetitionState


def _first_nonblank_line(text: str) -> str:
    lines = [ln.lstrip("#").strip() for ln in text.splitlines() if ln.strip()]
    return lines[0][:300] if lines else text[:300]


class FunctionalTopology(MiniTeamTopology):
    """
    Adds a persistent team-lead planning phase on top of MiniTeamTopology.
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

        role = ROLE_CATALOG["team-lead"]
        plan_prompt = build_team_lead_prompt(
            iteration=state.iteration,
            max_iterations=state.max_iterations,
            project_dir=project_dir,
            n_parallel=n_parallel,
        )
        try:
            plan_result, session_id = await run_agent(
                agent_name="team-lead",
                prompt=plan_prompt,
                system_prompt=role.system_prompt,
                allowed_tools=list(role.tools),
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
        result.team_session_ids = team_session_ids

        plans_dir = Path(project_dir) / ".claude" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"iter-{state.iteration:02d}.md").write_text(plan_text)

        # ── 2. Reset experiment state ────────────────────────────────────────
        exp_state_path = Path(project_dir) / ".claude" / "EXPERIMENT_STATE.json"
        exp_state_path.parent.mkdir(parents=True, exist_ok=True)
        exp_state_path.write_text("{}")

        # ── 3. Pipeline (MiniTeamTopology) ───────────────────────────────────
        impl_status, impl_result = await self._run_pipeline(
            plan_text=plan_text,
            project_dir=project_dir,
            mt=mt,
            consume_agent_call=consume_agent_call,
            check_budget=check_budget,
            mcp=mcp,
        )

        result.oof_score = impl_result.get("oof_score")
        result.quality_score = float(impl_result.get("quality_score") or 0)
        result.solution_files = impl_result.get("solution_files") or []
        result.submission_file = impl_result.get("submission_file") or ""
        result.notes = impl_result.get("notes") or ""
        result.status = impl_status

        if result.status == "error":
            result.error_message = impl_result.get("error_message") or "unknown error"
            result.team_session_ids = team_session_ids
            return result

        # ── 4. validator ─────────────────────────────────────────────────────
        if not self._budget_ok("validator", consume_agent_call, check_budget):
            result.team_session_ids = team_session_ids
            return result

        val_result = await self._run_validator(
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

        # ── 5. memory-keeper ─────────────────────────────────────────────────
        if self._budget_ok("memory-keeper", consume_agent_call, check_budget):
            mem_result = await self._run_memory_keeper(
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

