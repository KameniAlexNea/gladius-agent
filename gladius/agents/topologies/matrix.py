"""
Matrix topology — Microsoft-style dual-authority coordination.

Architecture:
  Both team-lead AND domain-expert must approve before each phase advances.
  This creates dual accountability: technical lead (team-lead) + domain
  reviewer (domain-expert).

  team-lead (plan)
    ↓
  ml-engineer (implement)
    ↓ (report to both)
  team-lead review  →  APPROVE
  domain-expert review  →  APPROVE
    ↓ (only if both approve)
  evaluator → validator → memory-keeper

If either reviewer rejects with CRITICAL issues, the domain-expert provides
fixes and ml-engineer re-runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gladius.agents.roles.catalog import ROLE_CATALOG
from gladius.agents.roles.specs import (
    ITERATION_RESULT_SCHEMA,
    build_team_lead_prompt,
)
from gladius.agents.runtime.agent_runner import run_agent
from gladius.agents.runtime.helpers import get_runtime_model
from gladius.agents.runtime.planning_runner import run_planning_agent
from gladius.agents.topologies.base import BaseTopology, IterationResult
from gladius.agents.topologies.functional import (
    _COORDINATOR_SYSTEM_PROMPT,
    _build_coordinator_prompt,
    _build_pipeline_agent_defs,
    _first_nonblank_line,
    _mcp_servers,
    FunctionalTopology,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState


_TECHNICAL_REVIEW_SCHEMA = {
    "type": "object",
    "required": ["decision", "reasoning"],
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "reject"]},
        "critical_issues": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "additionalProperties": False,
}

_TECHNICAL_REVIEW_SYSTEM_PROMPT = """\
You are the technical lead reviewer (team-lead in review mode).

Your job: review the ML experiment outcome from a technical perspective.
Approve if the code runs correctly and the results are technically sound.
Reject if there are execution failures, invalid OOF scores, or structural problems.

Emit structured output: {"decision": "approve"|"reject", "critical_issues": [...], "warnings": [...], "reasoning": "..."}
"""

_DOMAIN_REVIEW_SYSTEM_PROMPT = """\
You are the domain expert reviewer.

Your job: review the ML experiment from a domain/scientific perspective.
Check for:
- Data leakage or CV contamination
- Scientifically invalid features/assumptions
- Wrong metric or target encoding
- Train/test distribution issues

Approve if no CRITICAL scientific flaws. Reject otherwise.

Emit structured output: {"decision": "approve"|"reject", "critical_issues": [...], "warnings": [...], "reasoning": "..."}
"""


def _build_review_prompt(
    experiment_summary: dict,
    state: "CompetitionState",
    project_dir: str,
) -> str:
    import json
    return f"""\
Competition: {state.competition_id}
Metric: {state.target_metric or 'open-ended'}  Direction: {state.metric_direction or 'n/a'}
Project dir: {project_dir}

## Experiment result
{json.dumps(experiment_summary, indent=2)}

Review this result. Read relevant files if needed.
Emit structured JSON with your decision.
"""


class MatrixTopology(BaseTopology):
    """
    Microsoft-style matrix: dual-authority approval before advancing.

    Both team-lead (technical) and domain-expert (scientific) must approve
    the experiment before it proceeds to validation.
    """

    MAX_REVIEW_ROUNDS = 2  # max re-runs when reviewers reject

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
            plan_text, lead_sid = await run_planning_agent(
                agent_name="team-lead",
                prompt=plan_prompt,
                system_prompt=role_lead.system_prompt,
                allowed_tools=list(role_lead.tools),
                cwd=project_dir,
                resume=team_session_ids.get("team-lead"),
                mcp_servers=mcp,
            )
        except Exception as exc:
            logger.error(f"[team-lead] failed: {exc}", exc_info=True)
            result.error_message = str(exc)
            return result

        team_session_ids["team-lead"] = lead_sid
        result.plan_text = plan_text
        result.approach_summary = _first_nonblank_line(plan_text)

        plans_dir = Path(project_dir) / ".claude" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"iter-{state.iteration:02d}.md").write_text(plan_text)

        runtime_model = get_runtime_model()

        # ── 2. Implement → dual review loop ──────────────────────────────────
        last_impl: dict = {}
        approved = False

        for review_round in range(self.MAX_REVIEW_ROUNDS + 1):
            exp_state_path = Path(project_dir) / ".claude" / "EXPERIMENT_STATE.json"
            exp_state_path.parent.mkdir(parents=True, exist_ok=True)
            exp_state_path.write_text("{}")

            if not self._budget_ok(
                f"ml-engineer-round-{review_round}", consume_agent_call, check_budget
            ):
                break

            pipeline_roles = (
                "data-expert",
                "feature-engineer",
                "ml-engineer",
                "evaluator",
            )
            try:
                impl_out, impl_sid = await run_agent(
                    agent_name=f"matrix-impl-{review_round}",
                    prompt=_build_coordinator_prompt(
                        plan_text=plan_text,
                        state=state,
                        pipeline_roles=pipeline_roles,
                    ),
                    system_prompt=_COORDINATOR_SYSTEM_PROMPT,
                    allowed_tools=[
                        f"Agent({','.join(pipeline_roles)})",
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
                logger.error(f"[matrix-impl-{review_round}] failed: {exc}", exc_info=True)
                result.error_message = str(exc)
                result.team_session_ids = team_session_ids
                return result

            last_impl = impl_out

            if impl_out.get("status") == "error":
                result.error_message = (
                    impl_out.get("error_message") or "implementation error"
                )
                break

            experiment_summary = {
                "status": impl_out.get("status"),
                "oof_score": impl_out.get("oof_score"),
                "quality_score": impl_out.get("quality_score"),
                "notes": impl_out.get("notes"),
                "solution_files": impl_out.get("solution_files"),
            }
            review_prompt = _build_review_prompt(experiment_summary, state, project_dir)

            # ── Technical review (team-lead) ─────────────────────────────────
            tech_decision = "approve"
            tech_issues: list[str] = []
            if self._budget_ok("tech-review", consume_agent_call, check_budget):
                try:
                    tech_out, _ = await run_agent(
                        agent_name="technical-review",
                        prompt=review_prompt,
                        system_prompt=_TECHNICAL_REVIEW_SYSTEM_PROMPT,
                        allowed_tools=["Read", "Glob", "Grep", "Bash"],
                        output_schema=_TECHNICAL_REVIEW_SCHEMA,
                        cwd=project_dir,
                        max_turns=10,
                    )
                    tech_decision = tech_out.get("decision", "approve")
                    tech_issues = tech_out.get("critical_issues") or []
                except Exception as exc:
                    logger.warning(f"[technical-review] failed (defaulting approve): {exc}")

            # ── Domain review (domain-expert) ────────────────────────────────
            domain_decision = "approve"
            domain_issues: list[str] = []
            if self._budget_ok("domain-review", consume_agent_call, check_budget):
                try:
                    dom_out, _ = await run_agent(
                        agent_name="domain-review",
                        prompt=review_prompt,
                        system_prompt=_DOMAIN_REVIEW_SYSTEM_PROMPT,
                        allowed_tools=[
                            "Read",
                            "Glob",
                            "Grep",
                            "Bash",
                            "Skill",
                            "mcp__skills-on-demand__search_skills",
                        ],
                        output_schema=_TECHNICAL_REVIEW_SCHEMA,
                        cwd=project_dir,
                        mcp_servers=mcp,
                        max_turns=10,
                    )
                    domain_decision = dom_out.get("decision", "approve")
                    domain_issues = dom_out.get("critical_issues") or []
                except Exception as exc:
                    logger.warning(f"[domain-review] failed (defaulting approve): {exc}")

            all_issues = tech_issues + domain_issues
            if tech_decision == "approve" and domain_decision == "approve":
                approved = True
                logger.info(f"[matrix] dual approval granted (round {review_round})")
                break

            # Reviewers rejected — run domain-expert fix then retry
            logger.warning(
                f"[matrix] round {review_round}: "
                f"tech={tech_decision} domain={domain_decision}, "
                f"issues={all_issues}. Fixing…"
            )

            if review_round < self.MAX_REVIEW_ROUNDS:
                if not self._budget_ok("domain-fix", consume_agent_call, check_budget):
                    break
                role_de = ROLE_CATALOG["domain-expert"]
                fix_prompt = (
                    f"Fix the following CRITICAL issues found by the dual reviewers:\n"
                    + "\n".join(f"- {iss}" for iss in all_issues)
                    + "\n\nRead relevant files and apply minimal targeted fixes."
                )
                _DOMAIN_FIX_SCHEMA = {
                    "type": "object",
                    "required": ["status"],
                    "properties": {
                        "status": {"type": "string"},
                        "issues_addressed": {"type": "array", "items": {"type": "string"}},
                        "message": {"type": "string"},
                    },
                    "additionalProperties": True,
                }
                try:
                    await run_agent(
                        agent_name="domain-fix",
                        prompt=fix_prompt,
                        system_prompt=role_de.system_prompt,
                        allowed_tools=list(role_de.tools),
                        output_schema=_DOMAIN_FIX_SCHEMA,
                        cwd=project_dir,
                        mcp_servers=mcp,
                        max_turns=15,
                    )
                except Exception as exc:
                    logger.warning(f"[domain-fix] failed: {exc}")

        # ── 3. Use last impl result (approved or not) ─────────────────────────
        result.oof_score = last_impl.get("oof_score")
        result.quality_score = float(last_impl.get("quality_score") or 0)
        result.solution_files = last_impl.get("solution_files") or []
        result.submission_file = last_impl.get("submission_file") or ""
        result.notes = last_impl.get("notes") or (
            "dual review not approved" if not approved else ""
        )
        result.status = last_impl.get("status", "error") if last_impl else "error"

        if result.status == "error":
            result.error_message = last_impl.get("error_message") or "matrix loop error"
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
                        "dual_approved": approved,
                        "approach_summary": result.approach_summary,
                    },
                    validator_notes=validator_notes,
                )
                result.memory_content = mem_result.get("memory_content")
                result.memory_summary = mem_result.get("summary")

        result.team_session_ids = team_session_ids
        return result
