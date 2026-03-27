from gladius import (
    RUNTIME_DATA_BRIEFING_RELATIVE_PATH,
    RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH,
    TEAM_LEAD_MEMORY_RELATIVE_PATH,
)
from gladius.config import SYSTEM_PROMPT_PATH as _SYSTEM_PROMPT_PATH
from gladius.state import CompetitionState


def _load_system_prompt() -> str:
    """Load and validate the coordinator system prompt markdown."""
    text = (
        _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        .replace(
            "{{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}}",
            RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH,
        )
        .replace(
            "{{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}}",
            RUNTIME_DATA_BRIEFING_RELATIVE_PATH,
        )
        .replace("{{TEAM_LEAD_MEMORY_RELATIVE_PATH}}", TEAM_LEAD_MEMORY_RELATIVE_PATH)
        .strip()
    )
    if len(text) < 500:
        raise RuntimeError(
            f"System prompt at {_SYSTEM_PROMPT_PATH} is unexpectedly short ({len(text)} chars)."
        )
    return text


SYSTEM_PROMPT = _load_system_prompt()


TOP_LEVEL_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "TodoWrite",
    "Agent",
]


_TOPOLOGY_FIRST_STEP: dict[str, str] = {
    "functional": "data-expert → feature-engineer → ml-engineer → evaluator → validator → memory-keeper",
    "two-pizza": "full-stack-coordinator (who decides which specialists to spawn) → validator → memory-keeper",
    "matrix": "ml-engineer (full pipeline) → dual review (team-lead + domain-expert) → evaluator → validator → memory-keeper",
    "autonomous": "N parallel mini-teams each running (data-expert → feature-engineer → ml-engineer → evaluator) → validator → memory-keeper",
    "platform": "platform-layer (data-expert) → product-layer (feature-engineer → ml-engineer) → evaluator → validator → memory-keeper",
}


def make_kickoff_prompt(state: CompetitionState) -> str:
    """Build an iteration-aware kickoff prompt."""
    flow = _TOPOLOGY_FIRST_STEP.get(
        state.topology, "follow the topology defined in CLAUDE.md"
    )

    if state.iteration == 1:
        return (
            "This is the FIRST iteration — no experiments have run yet.\n"
            f"1. Check if `{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}` exists. If NOT, delegate to `scout` — "
            "it will explore the data and write the briefing with shapes, distributions, "
            "risks, and strategic angles. If it already exists, skip scout entirely.\n"
            "2. Delegate to `team-lead` to plan a baseline experiment "
            "(team-lead must read DATA_BRIEFING.md, latest EXPERIMENT_STATE_iter*.json, "
            "current EXPERIMENT_STATE.json if present, and MEMORY.md before planning; "
            "team-lead is non-coding and returns planning output only).\n"
            f"3. Then follow the **{state.topology}** topology: {flow}.\n"
            "4. For every downstream specialist, forward the exact relevant team-lead section verbatim "
            "under `## Your Instructions from the Team-Lead` (no paraphrase).\n"
            "Focus on getting a clean, reproducible baseline score above all else."
        )

    best = (
        f"{state.best_oof_score:.6f}"
        if state.best_oof_score is not None
        else (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
    )
    metric_label = state.target_metric or "quality"
    return (
        f"This is iteration {state.iteration}/{state.max_iterations}. "
        f"Current best {metric_label}: **{best}**.\n\n"
        f"1. Skip `scout` — `{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}` already exists from iteration 1.\n"
        "2. Delegate to `team-lead` to plan the next experiment "
        "(team-lead must read DATA_BRIEFING.md, latest EXPERIMENT_STATE_iter*.json, "
        "current EXPERIMENT_STATE.json, and MEMORY.md before suggesting the next iteration; "
        "team-lead is non-coding and returns planning output only).\n"
        f"3. Then follow the **{state.topology}** topology: {flow}.\n"
        '   Skip any agent whose EXPERIMENT_STATE.json entry is already `"status": "success"`.'
        "\n4. Forward each downstream specialist's team-lead section verbatim under "
        "`## Your Instructions from the Team-Lead`; do not summarize.\n"
        "5. After validation, have `memory-keeper` update MEMORY.md.\n\n"
        "Avoid any approach listed under 'Failed Approaches' in your context."
    )
