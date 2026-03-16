"""
Agent launcher — writes CLAUDE.md and starts a single Claude agent that
plans its own approach and delegates to sub-agents.

CLAUDE.md is automatically injected into the agent context — no explicit read needed.
No Python orchestration. The agent decides topology from its context.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

import gladius.claude_md as claude_md
from gladius.project_setup import load_competition_config
from gladius.roles.agent_runner import run_agent

_SYSTEM_PROMPT = """\
You are a top-tier ML competition agent. Your goal for this iteration is one \
focused, high-impact experiment.

## Step 1 — Plan (read-only)
`CLAUDE.md` is already in your context — do not read it again.
Read `.claude/agent-memory/team-lead/MEMORY.md` to get the full experiment history.
Identify the single highest-leverage change to make this iteration.

**Do NOT explore data directories, run scripts, or write any files yourself.**
Your role is coordination only — delegate all implementation and exploration to specialists.

## Step 2 — Execute (delegate)
Delegate implementation to the right specialist in `.claude/agents/`:
- `team-lead` — strategic direction and hypothesis
- `data-expert` — EDA, data loading, feature infrastructure
- `feature-engineer` — feature transforms and selection
- `ml-engineer` — model, training loop, artifacts
- `evaluator` — OOF metric verification
- `validator` — submission format and improvement gate
- `memory-keeper` — update MEMORY.md with learnings

**Re-dispatch rule:** before calling any specialist, read `.claude/EXPERIMENT_STATE.json`.
If that specialist's entry already has `"status": "success"`, skip them — their work is done.
Only re-dispatch a specialist if their status is missing, `"error"`, or if new upstream work requires it (e.g. data-expert fixes a bug flagged by ml-engineer).

Sequencing depends on the active topology (see `## Management Topology` in your context):
- **Sequential** (default): each specialist reads EXPERIMENT_STATE written by the previous one.
- **Parallel** (autonomous / multi-branch): spawn independent branches via parallel Task calls, then merge results before calling the validator.

## Constraints
- Do not repeat any approach listed under "Failed Approaches" in your context.
- Search skills before writing new code: `mcp__skills-on-demand__search_skills`.
- Save the final submission to `submissions/submission.csv` if the validator approves."""

_KICKOFF_PROMPT = """\
Kick off the competition with a focused experiment. CLAUDE.md is already in your context — check it for current scores, past approaches, and the active topology, then create a concise ordered plan.
Delegate to specialists as needed and avoid repeating failed approaches.
"""

_TOP_LEVEL_TOOLS = [
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


async def run_competition(
    competition_dir: str,
    max_turns: int | None = None,
) -> None:
    cfg = load_competition_config(competition_dir)
    project_dir = Path(competition_dir)

    claude_md.write_from_project(project_dir, cfg)

    logger.info(f"Launching agent: {cfg['competition_id']}")

    await run_agent(
        agent_name="gladius",
        prompt=_KICKOFF_PROMPT,
        system_prompt=_SYSTEM_PROMPT,
        allowed_tools=_TOP_LEVEL_TOOLS,
        output_schema=None,
        cwd=str(project_dir),
        max_turns=max_turns,
    )
