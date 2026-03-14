"""
Functional topology — Apple-style deep-expertise pipeline.

Sequential role pipeline:
  team-lead → data-expert → feature-engineer → ml-engineer
  → evaluator → validator → memory-keeper

Each role hands off to the next via EXPERIMENT_STATE.json.  The team-lead is
persistent (resumed).  All others are fresh per iteration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gladius.agents.roles.catalog import ROLE_CATALOG
from gladius.agents.roles.specs import (
    ITERATION_RESULT_SCHEMA,
    MEMORY_KEEPER_OUTPUT_SCHEMA,
    VALIDATOR_OUTPUT_SCHEMA,
    build_memory_keeper_prompt,
    build_team_lead_prompt,
    build_validator_prompt,
)
from gladius.agents.runtime.agent_runner import run_agent
from gladius.agents.runtime.helpers import build_runtime_agents, get_runtime_model
from gladius.agents.runtime.planning_runner import run_planning_agent
from gladius.agents.topologies.base import BaseTopology, IterationResult

if TYPE_CHECKING:
    from gladius.state import CompetitionState


def _mcp_servers(project_dir: str) -> dict:
    skills_dir = str(Path(project_dir) / ".claude" / "skills")
    return {
        "skills-on-demand": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "skills_on_demand.server"],
            "env": {"SKILLS_DIR": skills_dir},
        }
    }


def _first_nonblank_line(text: str) -> str:
    lines = [ln.lstrip("#").strip() for ln in text.splitlines() if ln.strip()]
    return lines[0][:300] if lines else text[:300]


class FunctionalTopology(BaseTopology):
    """
    Apple-style deep-expertise pipeline.

    Agents execute in strict sequence: each specialist hands off context
    to the next via EXPERIMENT_STATE.json.
    """

    PIPELINE: tuple[str, ...] = (
        "data-expert",
        "feature-engineer",
        "ml-engineer",
        "evaluator",
    )

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
            plan_text, session_id = await run_planning_agent(
                agent_name="team-lead",
                prompt=plan_prompt,
                system_prompt=role.system_prompt,
                allowed_tools=list(role.tools),
                cwd=project_dir,
                resume=team_session_ids.get("team-lead"),
                mcp_servers=mcp,
            )
        except Exception as exc:
            logger.error(f"[team-lead] failed: {exc}", exc_info=True)
            result.error_message = str(exc)
            return result

        team_session_ids["team-lead"] = session_id
        result.plan_text = plan_text
        result.approach_summary = _first_nonblank_line(plan_text)
        result.team_session_ids = team_session_ids

        # Save plan to .claude/plans/
        plans_dir = Path(project_dir) / ".claude" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"iter-{state.iteration:02d}.md").write_text(plan_text)

        # ── 2. Reset experiment state ────────────────────────────────────────
        exp_state_path = Path(project_dir) / ".claude" / "EXPERIMENT_STATE.json"
        exp_state_path.parent.mkdir(parents=True, exist_ok=True)
        exp_state_path.write_text("{}")

        # ── 3. Pipeline roles ────────────────────────────────────────────────
        # Build a compound prompt for the implementer-equivalent phase:
        # we spawn a single coordinator agent that has the full role suite
        # as sub-agents, exactly as the old implementer did.
        coordinator_role = ROLE_CATALOG["ml-engineer"]  # coordinator owns the pipeline

        coordinator_prompt = _build_coordinator_prompt(
            plan_text=plan_text,
            state=state,
            pipeline_roles=self.PIPELINE,
        )

        # Build agent registry with all pipeline roles
        runtime_model = get_runtime_model()
        agent_defs = _build_pipeline_agent_defs(runtime_model)

        try:
            impl_result, impl_session = await run_agent(
                agent_name="functional-coordinator",
                prompt=coordinator_prompt,
                system_prompt=_COORDINATOR_SYSTEM_PROMPT,
                allowed_tools=[
                    f"Agent({','.join(self.PIPELINE)})",
                    "Read",
                    "Write",
                    "Glob",
                    "TodoWrite",
                    "mcp__skills-on-demand__search_skills",
                    "mcp__skills-on-demand__list_skills",
                ],
                output_schema=ITERATION_RESULT_SCHEMA,
                cwd=project_dir,
                mcp_servers={
                    **mcp,
                    **_build_agent_defs_mcp(agent_defs),
                },
                max_turns=40,
            )
        except Exception as exc:
            logger.error(f"[functional-coordinator] failed: {exc}", exc_info=True)
            result.error_message = str(exc)
            result.team_session_ids = team_session_ids
            return result

        team_session_ids["functional-coordinator"] = impl_session
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

    # ── Shared role runners ───────────────────────────────────────────────────

    async def _run_validator(
        self,
        *,
        state: "CompetitionState",
        project_dir: str,
        platform: str,
        oof_score: float | None,
        quality_score: float | None,
        submission_path: str | None,
    ) -> dict:
        quota = state.max_submissions_per_day - state.submission_count
        prompt = build_validator_prompt(
            oof_score=oof_score,
            quality_score=quality_score,
            best_oof_score=state.best_oof_score,
            best_quality_score=state.best_quality_score,
            submission_path=submission_path,
            target_metric=state.target_metric,
            metric_direction=state.metric_direction,
            submission_quota_remaining=max(0, quota),
            project_dir=project_dir,
        )
        role = ROLE_CATALOG["validator"]
        mcp = _build_platform_mcp(platform)
        try:
            val_out, _ = await run_agent(
                agent_name="validator",
                prompt=prompt,
                system_prompt=role.system_prompt,
                allowed_tools=list(role.tools),
                output_schema=VALIDATOR_OUTPUT_SCHEMA,
                cwd=project_dir,
                mcp_servers=mcp,
                max_turns=20,
            )
            return val_out
        except Exception as exc:
            logger.error(f"[validator] failed: {exc}", exc_info=True)
            return {
                "is_improvement": False,
                "submit": False,
                "format_ok": True,
                "stop": False,
                "reasoning": f"validator error: {exc}",
                "next_directions": [],
            }

    async def _run_memory_keeper(
        self,
        *,
        state: "CompetitionState",
        project_dir: str,
        latest_result: dict,
        validator_notes: str,
    ) -> dict:
        role = ROLE_CATALOG["memory-keeper"]
        prompt = build_memory_keeper_prompt(
            iteration=state.iteration,
            competition_id=state.competition_id,
            target_metric=state.target_metric,
            metric_direction=state.metric_direction,
            experiments=state.experiments,
            failed_runs=state.failed_runs,
            latest_result=latest_result,
            validator_notes=validator_notes,
        )
        try:
            mem_out, _ = await run_agent(
                agent_name="memory-keeper",
                prompt=prompt,
                system_prompt=role.system_prompt,
                allowed_tools=list(role.tools),
                output_schema=MEMORY_KEEPER_OUTPUT_SCHEMA,
                cwd=project_dir,
                max_turns=15,
            )
            # Write MEMORY.md
            mem_path = (
                Path(project_dir) / ".claude" / "agent-memory" / "team-lead" / "MEMORY.md"
            )
            mem_path.parent.mkdir(parents=True, exist_ok=True)
            content = mem_out.get("memory_content", "")
            if content:
                mem_path.write_text(content)
            return mem_out
        except Exception as exc:
            logger.error(f"[memory-keeper] failed: {exc}", exc_info=True)
            return {}


# ── Coordinator agent (runs the pipeline via Agent() tool) ────────────────────

_COORDINATOR_SYSTEM_PROMPT = """\
You are the functional pipeline coordinator.

Your job: run a complete ML experiment by delegating to specialist agents in
strict sequence: data-expert → feature-engineer → ml-engineer → evaluator.

After each agent completes, READ .claude/EXPERIMENT_STATE.json to check status
before spawning the next agent. If a status is "error", do NOT advance; report
the experiment as failed.

Pipeline contract:
- data-expert     → sets up scaffold; writes data_expert.status
- feature-engineer → adds features; writes feature_engineer.status
- ml-engineer     → trains model; writes ml_engineer.status + oof_score
- evaluator       → confirms score; writes evaluator.status + oof_score

Always pass the current EXPERIMENT_STATE.json contents in the task prompt.
Emit StructuredOutput when the pipeline completes (success or error).

STRICT RULES:
- NEVER modify CLAUDE.md.
- Only write to .claude/EXPERIMENT_STATE.json — no other files directly.
- Read a file before rewriting it.
- Once StructuredOutput is emitted, stop immediately.
"""


def _build_coordinator_prompt(
    plan_text: str,
    state: "CompetitionState",
    pipeline_roles: tuple[str, ...],
) -> str:
    return f"""\
Competition: {state.competition_id}
Metric: {state.target_metric or 'open-ended'}  Direction: {state.metric_direction or 'n/a'}
Data dir: {state.data_dir}
Best OOF so far: {state.best_oof_score}

Pipeline roles: {' → '.join(pipeline_roles)}

## Plan for this iteration
{plan_text}

Run the pipeline in order. Check EXPERIMENT_STATE.json after each agent.
Report final results via StructuredOutput.
"""


def _build_pipeline_agent_defs(runtime_model: str) -> dict:
    """Build AgentDefinition objects from the role catalog for pipeline roles."""
    from claude_agent_sdk import AgentDefinition

    defs = {}
    for name in ("data-expert", "feature-engineer", "ml-engineer", "evaluator"):
        role = ROLE_CATALOG[name]
        defs[name] = AgentDefinition(
            description=role.description,
            prompt=role.system_prompt,
            tools=list(role.tools),
            model=runtime_model,
        )
    return defs


def _build_agent_defs_mcp(agent_defs: dict) -> dict:
    """Placeholder — the SDK picks up agent defs from the options.agents dict."""
    return {}


def _build_platform_mcp(platform: str) -> dict:
    if platform == "kaggle":
        return {
            "kaggle": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "gladius.tools.kaggle_tools"],
            }
        }
    if platform == "zindi":
        return {
            "zindi": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "gladius.tools.zindi_tools"],
            }
        }
    if platform == "fake":
        return {
            "fake": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "gladius.tools.fake_platform_tools"],
            }
        }
    return {}
