"""
Planner — explores the competition data and decides what to try next.

This is the only persistent agent (resumed every iteration). It accumulates
understanding of the competition, the data, what has worked and what hasn't.

It is NOT told how to structure files, what to name things, or how to compute
metrics. Its only job is to produce a concrete, ordered action plan.
"""

from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── Output schema ─────────────────────────────────────────────────────────────
# Minimal. Claude writes the steps; orchestrator passes them to the implementer.
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["plan", "approach_summary"],
    "properties": {
        "approach_summary": {
            "type": "string",
            "description": "One paragraph — what you decided to try and why",
        },
        "plan": {
            "type": "array",
            "description": "Ordered implementation steps for the implementer to follow",
            "items": {
                "type": "object",
                "required": ["step", "description"],
                "properties": {
                    "step": {"type": "integer"},
                    "description": {"type": "string"},
                },
            },
        },
        "expected_metric_delta": {
            "type": "number",
            "description": "Rough estimate of OOF improvement. Can be 0 if unknown.",
        },
        "plans": {
            "type": "array",
            "description": (
                "Alternative approaches for parallel execution. "
                "Only populated when n_parallel > 1 is requested. "
                "Each entry has {approach_summary, plan} with the same structure as the primary plan."
            ),
            "items": {
                "type": "object",
                "required": ["approach_summary", "plan"],
                "properties": {
                    "approach_summary": {"type": "string"},
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["step", "description"],
                            "properties": {
                                "step": {"type": "integer"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
    "additionalProperties": False,
}


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
    return await run_agent(
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
        allowed_tools=["Read", "Glob", "Grep", "Bash", "WebSearch", "Task"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        resume=state.planner_session_id,
        mcp_servers=mcp_servers,
        max_turns=40,
    )


def _build_platform_mcp(platform: str, data_dir: str) -> dict:
    """Return MCP server dict for the given platform so planner can query it."""
    servers: dict = {}
    try:
        if platform == "fake":
            import os

            os.environ.setdefault(
                "FAKE_ANSWERS_PATH",
                str(__import__("pathlib").Path(data_dir) / ".answers.csv"),
            )
            from gladius.tools.fake_platform_tools import fake_server

            servers["fake_platform"] = fake_server
        elif platform == "kaggle":
            from gladius.tools.kaggle_tools import kaggle_server

            servers["kaggle"] = kaggle_server
        # zindi: no leaderboard MCP yet, skip
    except Exception:
        pass  # MCP servers are optional for planner
    return servers
