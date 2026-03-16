"""
Agent launcher — writes CLAUDE.md and starts a single Claude agent that
reads the project, plans its own approach, and delegates to sub-agents.

No Python orchestration. The agent decides topology by reading CLAUDE.md.
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
Use the built-in Plan subagent to research the current state:
- Scores, experiment history, and failed approaches are in CLAUDE.md.
- Explore existing code under `src/` and `scripts/`.
- Identify the single highest-leverage change to make this iteration.

## Step 2 — Execute (delegate)
Delegate implementation to the right specialist in `.claude/agents/`:
- `team-lead` — strategic direction and hypothesis
- `data-expert` — EDA, data loading, feature infrastructure
- `feature-engineer` — feature transforms and selection
- `ml-engineer` — model, training loop, artifacts
- `evaluator` — OOF metric verification
- `validator` — submission format and improvement gate
- `memory-keeper` — update MEMORY.md with learnings

Run specialists sequentially. Each one reads the output of the previous.

## Constraints
- Do not repeat any approach listed under "Failed Approaches" in CLAUDE.md.
- Search skills before writing new code: `mcp__skills-on-demand__search_skills`.
- Save the final submission to `submissions/submission.csv` if the validator approves."""

_KICKOFF_PROMPT = """\
Kick off the competition with a focused experiment. Use the Plan tool to create a concise, ordered plan for this iteration. 
Then delegate to specialists as needed. 

Read CLAUDE.md for current scores and past approaches, and avoid repeating failed approaches.
"""

_TOP_LEVEL_TOOLS = [
    "Read", "Write", "Edit", "MultiEdit", "Bash",
    "Glob", "Grep", "WebSearch", "Skill", "TodoWrite",
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
