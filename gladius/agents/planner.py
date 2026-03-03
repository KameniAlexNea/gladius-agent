"""
Planner — explores the competition data and decides what to try next.

This is the only persistent agent (resumed every iteration). It accumulates
understanding of the competition, the data, what has worked and what hasn't.

It is NOT told how to structure files, what to name things, or how to compute
metrics. Its only job is to produce a concrete, ordered action plan.
"""

import logging
from typing import TYPE_CHECKING

from gladius.agents._base import run_planning_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

logger = logging.getLogger(__name__)

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
    # SDK MCP servers race with the result message (end_input closes stdin
    # before control-response can be written back).  The planner only needs
    # to *plan*; it doesn't submit or check the real leaderboard, so we
    # skip MCP entirely here.  Platform tools are injected for validation/
    # submission agents instead.
    mcp_servers: dict = {}

    parallel_instruction = ""
    if n_parallel > 1:
        parallel_instruction = (
            f"\n\nIMPORTANT: Generate {n_parallel} independent approaches for parallel "
            f"execution. Put the primary (highest confidence) approach in `plan` / "
            f"`approach_summary` as usual, and put all {n_parallel} approaches "
            f"(including the primary) in the `plans` list. "
            f"Each approach must be substantially different — different models, "
            f"feature strategies, or architectures — so parallel runs are not redundant."
        )

    prompt = f"""\
Read CLAUDE.md first — it contains the full competition state.

Iteration   : {state.iteration} / {state.max_iterations}
Project dir : {project_dir}

Your job:
- Read CLAUDE.md and your memory (.claude/agent-memory/planner/MEMORY.md).
- Explore the data directory and any existing solution files at your discretion.
- Decide the highest-impact next thing to try.
- Output a concrete, ordered plan the implementer can follow without further guidance.
- Update your memory with anything worth remembering across iterations.

Be specific. The implementer will execute your plan blindly.{parallel_instruction}
"""
    _session = state.planner_session_id
    _kwargs = dict(
        agent_name="planner",
        prompt=prompt,
        system_prompt=(
            "You are an expert ML competition analyst. "
            "You explore first, then plan. "
            "You never implement code yourself — you produce plans for an implementer. "
            "Your plans are specific, ordered, and self-contained. "
            "Always read CLAUDE.md at the start of every session. "
            "If CLAUDE.md shows a STAGNATION WARNING, your top priority is to "
            "break out of the local optimum: explore different data representations, "
            "fundamentally different model families, or go back to raw data exploration "
            "rather than incrementally tweaking the current approach."
        ),
        allowed_tools=[
            "Read",
            "Glob",
            "Grep",
            "Bash",
            "WebSearch",
            "TodoWrite",
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

    # Extract a short approach_summary from the first non-blank line of the plan.
    summary_lines = [
        ln.lstrip("#").strip() for ln in plan_text.splitlines() if ln.strip()
    ]
    approach_summary = summary_lines[0][:300] if summary_lines else plan_text[:300]

    plan_dict: dict = {
        "approach_summary": approach_summary,
        "plan_text": plan_text,
        # Fallback list with a single entry used by older code paths
        "plan": [{"step": 1, "description": plan_text}],
        # Parallel plans are not supported in plan mode; orchestrator falls back
        # to single-plan execution when this list is empty.
        "plans": [],
    }
    return plan_dict, session_id
