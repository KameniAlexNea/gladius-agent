"""
Autonomous topology — Meta-style autonomous parallel teams.

Architecture:
  team-lead produces N independent plans (n_parallel).
  N independent mini-teams (each a functional pipeline) run concurrently
  via asyncio.gather.  The validator picks the best result.

  team-lead (N plans)
    ├── mini-team-1 (functional pipeline)
    ├── mini-team-2 (functional pipeline)
    └── ... × n_parallel
  validator (picks best) → memory-keeper
"""

from __future__ import annotations

import asyncio
import re
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
    _build_coordinator_prompt,
    _build_pipeline_agent_defs,
    _first_nonblank_line,
    _mcp_servers,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState


def _extract_parallel_plans(plan_text: str, n_parallel: int) -> list[str]:
    """Extract ## Approach N sections from team-lead output."""
    heading_re = re.compile(r"(?im)^#{1,6}\s*approach\s+\d+\s*$")
    matches = list(heading_re.finditer(plan_text))
    if not matches:
        return [plan_text]  # fallback: single plan

    sections: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_text)
        section = plan_text[start:end].strip()
        if section:
            sections.append(section)

    seen: set[str] = set()
    unique: list[str] = []
    for s in sections:
        key = " ".join(s.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique[:n_parallel]


def _pick_best(
    results: list[tuple[int, IterationResult]],
    target_metric: str | None,
    metric_direction: str | None,
) -> IterationResult | None:
    """Return the best successful result from parallel branches."""
    successful = [(i, r) for i, r in results if r.status == "success"]
    if not successful:
        return None

    if target_metric:
        valid = [(i, r) for i, r in successful if r.oof_score is not None]
        if not valid:
            return successful[0][1]
        return max(
            valid,
            key=lambda x: (x[1].oof_score if metric_direction != "minimize" else -x[1].oof_score),  # type: ignore[operator]
        )[1]
    else:
        return max(successful, key=lambda x: (x[1].quality_score or 0.0))[1]


async def _run_one_branch(
    branch_idx: int,
    plan_text: str,
    state: "CompetitionState",
    project_dir: str,
    mcp: dict,
) -> tuple[int, IterationResult]:
    """Run a single autonomous mini-team branch."""
    branch_state_path = (
        Path(project_dir) / ".claude" / f"EXPERIMENT_STATE_branch{branch_idx}.json"
    )
    branch_state_path.write_text("{}")

    runtime_model = get_runtime_model()

    try:
        result, _ = await run_agent(
            agent_name=f"autonomous-branch-{branch_idx}",
            prompt=_build_coordinator_prompt(
                plan_text=plan_text,
                state=state,
                pipeline_roles=("data-expert", "feature-engineer", "ml-engineer", "evaluator"),
            ),
            system_prompt=ROLE_CATALOG["functional-coordinator"].system_prompt,
            allowed_tools=[
                "Agent(data-expert,feature-engineer,ml-engineer,evaluator)",
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
        ir = IterationResult(
            status=result.get("status", "error"),
            oof_score=result.get("oof_score"),
            quality_score=float(result.get("quality_score") or 0),
            solution_files=result.get("solution_files") or [],
            submission_file=result.get("submission_file") or "",
            notes=result.get("notes") or "",
            error_message=result.get("error_message") or "",
            plan_text=plan_text,
            approach_summary=_first_nonblank_line(plan_text),
        )
        return branch_idx, ir
    except Exception as exc:
        logger.error(f"[autonomous-branch-{branch_idx}] failed: {exc}", exc_info=True)
        return branch_idx, IterationResult(
            status="error",
            error_message=str(exc),
            plan_text=plan_text,
            approach_summary=_first_nonblank_line(plan_text),
        )


class AutonomousTopology(BaseTopology):
    """
    Meta-style autonomous teams: N independent pipelines run in parallel,
    best result wins.
    """

    async def run_iteration(
        self,
        state: "CompetitionState",
        project_dir: str,
        platform: str,
        *,
        n_parallel: int = 2,
        consume_agent_call=None,
        check_budget=None,
    ) -> IterationResult:
        n_parallel = max(1, n_parallel)
        result = IterationResult(status="error")
        team_session_ids: dict[str, str] = dict(state.team_session_ids or {})
        mcp = _mcp_servers(project_dir)
        ft = FunctionalTopology()

        # ── 1. team-lead: N plans ────────────────────────────────────────────
        if not self._budget_ok("team-lead", consume_agent_call, check_budget):
            result.error_message = "budget exceeded before team-lead"
            return result

        role_lead = ROLE_CATALOG["team-lead"]
        plan_prompt = build_team_lead_prompt(
            iteration=state.iteration,
            max_iterations=state.max_iterations,
            project_dir=project_dir,
            n_parallel=n_parallel,
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

        # ── 2. Extract parallel plans ────────────────────────────────────────
        plans = _extract_parallel_plans(plan_text, n_parallel)
        logger.info(f"[autonomous] spawning {len(plans)} parallel branch(es)")

        for i in range(len(plans)):
            if not self._budget_ok(f"branch-{i}", consume_agent_call, check_budget):
                plans = plans[:i]
                break

        if not plans:
            result.error_message = "no budget for any branch"
            result.team_session_ids = team_session_ids
            return result

        # ── 3. Run all branches concurrently ─────────────────────────────────
        branch_tasks = [
            _run_one_branch(i, plan, state, project_dir, mcp)
            for i, plan in enumerate(plans)
        ]
        branch_results: list[tuple[int, IterationResult]] = await asyncio.gather(
            *branch_tasks
        )

        best = _pick_best(branch_results, state.target_metric, state.metric_direction)
        if best is None:
            result.error_message = "all branches failed"
            result.team_session_ids = team_session_ids
            result.notes = "; ".join(
                f"branch-{i}: {r.error_message}" for i, r in branch_results
            )
            return result

        result.oof_score = best.oof_score
        result.quality_score = best.quality_score
        result.solution_files = best.solution_files
        result.submission_file = best.submission_file
        result.notes = best.notes
        result.status = best.status
        result.approach_summary = best.approach_summary

        # ── 4. validator ─────────────────────────────────────────────────────
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
                        "branches_run": len(plans),
                    },
                    validator_notes=validator_notes,
                )
                result.memory_content = mem_result.get("memory_content")
                result.memory_summary = mem_result.get("summary")

        result.team_session_ids = team_session_ids
        return result
