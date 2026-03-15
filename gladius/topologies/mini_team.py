"""
Mini-team topology — lean ML pipeline, no planning phase.

Runs the four specialist agents directly in sequence:
  data-expert → feature-engineer → ml-engineer → evaluator
  → validator → memory-keeper
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gladius.roles import ROLE_CATALOG
from gladius.roles.agent_runner import run_agent
from gladius.roles.specs import (
    MEMORY_KEEPER_OUTPUT_SCHEMA,
    PIPELINE_AGENT_OUTPUT_SCHEMA,
    VALIDATOR_OUTPUT_SCHEMA,
    build_memory_keeper_prompt,
    build_validator_prompt,
)
from gladius.topologies.base import BaseTopology, IterationResult

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


def _build_platform_mcp(platform: str) -> dict:
    if platform == "kaggle":
        return {"kaggle": {"type": "stdio", "command": sys.executable, "args": ["-m", "gladius.tools.kaggle_tools"]}}
    if platform == "zindi":
        return {"zindi": {"type": "stdio", "command": sys.executable, "args": ["-m", "gladius.tools.zindi_tools"]}}
    if platform == "fake":
        return {"fake": {"type": "stdio", "command": sys.executable, "args": ["-m", "gladius.tools.fake_platform_tools"]}}
    return {}

_PIPELINE = (
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "evaluator",
)


class MiniTeamTopology(BaseTopology):
    """
    Lean ML pipeline — no team-lead planning phase.

    Python drives four specialists in sequence, passing compressed
    structured outputs as context to each successive agent.
    """

    PIPELINE: tuple[str, ...] = _PIPELINE

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
        mcp = _mcp_servers(project_dir)

        exp_state_path = Path(project_dir) / ".claude" / "EXPERIMENT_STATE.json"
        exp_state_path.parent.mkdir(parents=True, exist_ok=True)
        exp_state_path.write_text("{}")

        impl_status, impl_result = await self._run_pipeline(
            plan_text="",
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
            return result

        if not self._budget_ok("validator", consume_agent_call, check_budget):
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

        return result

    # ── Pipeline execution ────────────────────────────────────────────────────

    async def _run_pipeline(
        self,
        *,
        plan_text: str,
        project_dir: str,
        mt: dict,
        consume_agent_call=None,
        check_budget=None,
        mcp: dict,
    ) -> tuple[str, dict]:
        pipeline_state: dict[str, dict] = {}

        for role_name in self.PIPELINE:
            if not self._budget_ok(role_name, consume_agent_call, check_budget):
                return "error", {"error_message": f"budget exceeded before {role_name}"}

            role = ROLE_CATALOG[role_name]
            agent_def = role
            prev_parts = [
                f"[{r}] {pipeline_state[r].get('summary', '')}"
                + (
                    f" (OOF: {pipeline_state[r]['oof_score']})"
                    if pipeline_state[r].get("oof_score") is not None
                    else ""
                )
                for r in self.PIPELINE
                if r in pipeline_state
            ]
            prompt = (f"## Plan\n{plan_text}" if plan_text else "") + (
                ("\n\n" if plan_text else "") + "## Previous stages\n" + "\n".join(prev_parts)
                if prev_parts
                else ""
            )
            logger.info(f"[pipeline] running {role_name}")
            try:
                agent_out, _ = await run_agent(
                    agent_name=role_name,
                    prompt=prompt,
                    system_prompt=agent_def.system_prompt,
                    allowed_tools=list(agent_def.tools),
                    output_schema=PIPELINE_AGENT_OUTPUT_SCHEMA,
                    cwd=project_dir,
                    mcp_servers=mcp,
                    max_turns=mt.get(
                        role_name, mt.get(role_name.replace("-", "_"), role.max_turns)
                    ),
                )
            except Exception as exc:
                logger.error(f"[{role_name}] failed: {exc}", exc_info=True)
                return "error", {"error_message": f"[{role_name}] {exc}"}

            pipeline_state[role_name] = agent_out
            if agent_out.get("status") == "error":
                logger.warning(f"[{role_name}] reported error: {agent_out.get('error_message', '')}")
                return "error", {
                    "error_message": f"[{role_name}] {agent_out.get('error_message', 'agent error')}",
                    **agent_out,
                }

        evaluator_out = pipeline_state.get("evaluator", {})
        ml_out = pipeline_state.get("ml-engineer", {})
        notes_parts = [
            f"[{r}] {pipeline_state[r].get('summary', '')}"
            for r in self.PIPELINE
            if r in pipeline_state
        ]
        return "success", {
            "status": "success",
            "oof_score": evaluator_out.get("oof_score") or ml_out.get("oof_score"),
            "quality_score": float(evaluator_out.get("quality_score") or 0),
            "solution_files": ml_out.get("files_modified") or [],
            "submission_file": ml_out.get("submission_file") or evaluator_out.get("submission_file") or "",
            "notes": "\n".join(notes_parts),
        }

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
        max_turns: dict | None = None,
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
                max_turns=(max_turns or {}).get("validator", 20),
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
        max_turns: dict | None = None,
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
                max_turns=(max_turns or {}).get("memory_keeper", 15),
            )
            mem_path = Path(project_dir) / ".claude" / "agent-memory" / "team-lead" / "MEMORY.md"
            mem_path.parent.mkdir(parents=True, exist_ok=True)
            content = mem_out.get("memory_content", "")
            if content:
                mem_path.write_text(content)
            return mem_out
        except Exception as exc:
            logger.error(f"[memory-keeper] failed: {exc}", exc_info=True)
            return {}
