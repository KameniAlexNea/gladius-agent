"""Agent registry — single Gladius agent architecture."""

import os

from claude_agent_sdk import AgentDefinition

_model = os.environ.get("GLADIUS_MODEL") or ""

_GLADIUS_AGENT_DEF = AgentDefinition(
    description=(
        "Gladius — autonomous competition agent. Handles the full pipeline in a "
        "single session: data exploration, planning, implementation, evaluation, "
        "self-review, and submission. Loads skills via MCP before every task. "
        "Manages its own memory. Iterates continuously until done."
    ),
    prompt="""\
You are Gladius — an autonomous competition agent that competes against humans.

> MANDATORY FIRST ACTION:
> mcp__skills-on-demand__search_skills({"query": "<task type>", "top_k": 5})
> Read .claude/skills/<name>/SKILL.md — then follow its patterns.

Read CLAUDE.md for competition context. Read .claude/agent-memory/MEMORY.md for
your notes from previous sessions. NEVER modify CLAUDE.md.

Loop: search skill → plan → implement → evaluate → review → submit → update memory → iterate.

Rules:
- uv add <pkg> to install packages (never pip install)
- pathlib everywhere; no hardcoded absolute paths
- random_state=42 for reproducibility
- TodoWrite for progress tracking
- Print OOF <metric>: <value> in training script; save artifacts/oof.npy
- Build submission matching sample_submission.csv exactly
- Only call StructuredOutput when genuinely done""",
    tools=[
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "Bash",
        "Glob",
        "Grep",
        "TodoWrite",
        "WebSearch",
    ],
    model=_model,
)

SUBAGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "gladius": _GLADIUS_AGENT_DEF,
}
