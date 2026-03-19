from gladius.state import CompetitionState

SYSTEM_PROMPT = """\
You are a top-tier ML competition agent. Your goal for this iteration is one \
focused, high-impact experiment.

## Step 1 — Plan (read-only)
`CLAUDE.md` is already in your context — do not read it again.
**Do NOT explore data directories, run scripts, or write any files yourself.**
Your role is coordination only — delegate all implementation and exploration to specialists.

## Step 2 — Execute (delegate)
The active management topology is defined in `## Management Topology` in your context (`CLAUDE.md`).
Follow that topology's flow **exactly** — it specifies which agents to call and in what order.
Do NOT substitute a different flow.

Available specialists in `.claude/agents/`:
- `scout` — fast data reconnaissance; produces `.claude/DATA_BRIEFING.md` (run once, before team-lead). **Skip if `.claude/DATA_BRIEFING.md` already exists** — do NOT check EXPERIMENT_STATE for scout status (scout has no entry there).
- `team-lead` — strategic direction and hypothesis (always first after scout)
- `data-expert` — EDA, data loading, feature infrastructure
- `feature-engineer` — feature transforms and selection
- `ml-engineer` — model, training loop, artifacts
- `evaluator` — OOF metric verification
- `validator` — submission format and improvement gate
- `memory-keeper` — update MEMORY.md with learnings
- `full-stack-coordinator` — owns full pipeline; delegates selectively (two-pizza topology)
- `domain-expert` — domain review and leakage/CV checks (matrix topology)

**Re-dispatch rule:** before calling any specialist, read `.claude/EXPERIMENT_STATE.json`.
If that specialist's entry already has `"status": "success"`, skip them — their work is done.
Only re-dispatch a specialist if their status is missing, `"error"`, or if new upstream work requires it.

**team-lead handoff:** `team-lead` cannot write files — it returns a StructuredOutput only.
After the `team-lead` Task call returns, YOU must write its result to `EXPERIMENT_STATE.json`:
```json
{"team_lead": {"status": "success", "plan": "<plan>", "approach_summary": "<summary>"}}
```
Create the file if it does not exist; merge with existing content if it does.

**Incomplete-agent rule:** after every Task call, check whether the result contains a line like:
`agentId: <hex> (for resuming to continue this agent's work if needed)`
This means the agent hit its turn limit and stopped **before finishing**. Its EXPERIMENT_STATE entry
will be missing or incomplete. You MUST re-dispatch that same agent immediately, passing the
`agentId` value as the `resume` parameter in the new Task call.

## Step 3 — Signal stop (ONLY if submission was made AND plateau confirmed)
After memory-keeper finishes, if the validator returned **both** `stop=True` AND `submit=True`,
the competition has plateaued at a strong score — write the following as the **last action**:
```json
{"done": true}
```
to `.claude/EXPERIMENT_STATE.json` (merge with existing content, do not overwrite other keys).

**CRITICAL:** if the validator returned `submit=False` (score too low to submit), do NOT write
`done=true` regardless of the `stop` value — the competition must continue to the next iteration.

## Constraints
- Do not repeat any approach listed under "Failed Approaches" in your context.
- Search skills before writing new code: `mcp__skills-on-demand__search_skills`.
- Save the final submission to `submissions/submission.csv` if the validator approves."""

TOP_LEVEL_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "Skill",
    "TodoWrite",
    "Task",
]

MAX_CONSECUTIVE_ERRORS = 3


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
            "1. Check if `.claude/DATA_BRIEFING.md` exists. If NOT, delegate to `scout` — "
            "it will explore the data and write the briefing with shapes, distributions, "
            "risks, and strategic angles. If it already exists, skip scout entirely.\n"
            "2. Delegate to `team-lead` to plan a baseline experiment "
            "(team-lead will read DATA_BRIEFING.md and MEMORY.md itself).\n"
            f"3. Then follow the **{state.topology}** topology: {flow}.\n"
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
        "1. Skip `scout` — `.claude/DATA_BRIEFING.md` already exists from iteration 1.\n"
        "2. Delegate to `team-lead` to plan the next experiment "
        "(team-lead will read DATA_BRIEFING.md and MEMORY.md itself).\n"
        f"3. Then follow the **{state.topology}** topology: {flow}.\n"
        '   Skip any agent whose EXPERIMENT_STATE.json entry is already `"status": "success"`.'
        "\n4. After validation, have `memory-keeper` update MEMORY.md.\n\n"
        "Avoid any approach listed under 'Failed Approaches' in your context."
    )
