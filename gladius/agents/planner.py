"""
Planner — explores the competition data and decides what to try next.

This is the only persistent agent (resumed every iteration). It accumulates
understanding of the competition, the data, what has worked and what hasn't.

It is NOT told how to structure files, what to name things, or how to compute
metrics. Its only job is to produce a concrete, ordered action plan.
"""

import re
from typing import TYPE_CHECKING

from gladius.agents._base import run_planning_agent
from gladius.agents.specs.planner_spec import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_alternative_prompt,
    build_planner_prompt,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState

from loguru import logger


def _first_nonblank_line(text: str) -> str:
    lines = [ln.lstrip("#").strip() for ln in text.splitlines() if ln.strip()]
    return lines[0][:300] if lines else text[:300]


def _plan_dict_from_text(plan_text: str) -> dict:
    return {
        "approach_summary": _first_nonblank_line(plan_text),
        "plan_text": plan_text,
        "plan": [{"step": 1, "description": plan_text}],
    }


def _extract_parallel_plans(plan_text: str, n_parallel: int) -> list[dict]:
    """Extract approach sections from planner markdown.

    Expected shape (heading level may vary):
      ## Approach 1
      ...content...
      ## Approach 2
      ...content...
    """
    if n_parallel <= 1:
        return []

    heading_re = re.compile(r"(?im)^#{1,6}\s*approach\s+\d+\s*$")
    matches = list(heading_re.finditer(plan_text))
    if not matches:
        return []

    sections: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(plan_text)
        section = plan_text[start:end].strip()
        if section:
            sections.append(section)

    # Deduplicate by normalized body while preserving order.
    seen: set[str] = set()
    plans: list[dict] = []
    for section in sections:
        key = " ".join(section.lower().split())
        if key in seen:
            continue
        seen.add(key)
        plans.append(_plan_dict_from_text(section))
        if len(plans) >= n_parallel:
            break
    return plans


# ── Output schema ─────────────────────────────────────────────────────────────
# NOTE: kept for reference / documentation only.
# With permission_mode="plan" the planner exits via the built-in ExitPlanMode
# tool (a markdown string), so JSON schema validation is not used here.
# The plan dict returned by run_planner() is assembled from the plan text.
#
# Downstream code (orchestrator + implementer) looks for:
#   plan["approach_summary"]  – first non-blank line of the markdown plan
#   plan["plan_text"]         – full markdown plan (preferred by implementer)
#   plan["plan"]              – [{"step": 1, "description": plan_text}] fallback
#   plan["plans"]             – [] (parallel unsupported in plan mode)


async def run_planner(
    state: "CompetitionState",
    data_dir: str,
    project_dir: str,
    platform: str = "kaggle",
    n_parallel: int = 1,
) -> tuple[dict, str]:
    """
    Explore, analyse, plan.

    Returns (plan_dict, session_id).
    session_id is stored in state.planner_session_id to resume next iteration.

    When n_parallel > 1, the planner is asked to produce that many independent
    approaches in the `plans` list field.
    """
    # Inject skills-on-demand MCP so the planner can search the skill catalog
    # without bulk-loading every SKILL.md file.
    import sys
    from pathlib import Path as _Path

    skills_dir = str(_Path(project_dir) / ".claude" / "skills")
    mcp_servers: dict = {
        "skills-on-demand": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "skills_on_demand.server"],
            "env": {"SKILLS_DIR": skills_dir},
        }
    }

    prompt = build_planner_prompt(
        iteration=state.iteration,
        max_iterations=state.max_iterations,
        project_dir=project_dir,
        n_parallel=n_parallel,
    )
    _session = state.planner_session_id
    _kwargs = dict(
        agent_name="planner",
        prompt=prompt,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        allowed_tools=[
            "Read",
            "Glob",
            "Grep",
            "WebSearch",
            "Skill",
            "TodoWrite",
            "ExitPlanMode",
            "mcp__skills-on-demand__search_skills",
            "mcp__skills-on-demand__list_skills",
        ],
        cwd=project_dir,
        mcp_servers=mcp_servers,
        max_turns=40,
    )
    try:
        plan_text, session_id = await run_planning_agent(**_kwargs, resume=_session)
    except Exception as exc:
        if _session is not None:
            logger.warning(
                f"Planner with resumed session {_session[:8]}… failed ({exc}), "
                "retrying with a fresh session."
            )
            plan_text, session_id = await run_planning_agent(**_kwargs, resume=None)
        else:
            raise

    primary_plan = _plan_dict_from_text(plan_text)
    plans: list[dict] = []

    if n_parallel > 1:
        plans = _extract_parallel_plans(plan_text, n_parallel)
        if not plans:
            plans = [primary_plan]

        # If the initial planning response did not include enough distinct
        # approach sections, request additional alternatives explicitly.
        attempts = 0
        while len(plans) < n_parallel and attempts < n_parallel * 2:
            attempts += 1
            existing_summaries = [p.get("approach_summary", "") for p in plans]
            alt_prompt = build_planner_alternative_prompt(existing_summaries)
            alt_kwargs = dict(_kwargs)
            alt_kwargs["prompt"] = alt_prompt
            alt_kwargs["resume"] = session_id
            alt_text, _ = await run_planning_agent(
                **alt_kwargs,
            )
            alt_plan = _plan_dict_from_text(alt_text)
            alt_key = " ".join(alt_plan["plan_text"].lower().split())
            existing_keys = {" ".join(p["plan_text"].lower().split()) for p in plans}
            if alt_key not in existing_keys:
                plans.append(alt_plan)

    plan_dict: dict = {
        "approach_summary": primary_plan["approach_summary"],
        "plan_text": primary_plan["plan_text"],
        "plan": primary_plan["plan"],
        "plans": plans,
    }
    return plan_dict, session_id
