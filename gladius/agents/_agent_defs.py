"""
Agent definitions — single-agent (solver) architecture.

The old planner + implementer-coordinator + 6-worker-subagent pipeline has been
replaced by a single solver agent that handles everything autonomously, guided
by skills loaded via the MCP skills-on-demand server.

Remaining agents:
  - solver     : single-session competition solver (plan → implement → evaluate → submit)
  - summarizer : read-only analyst that updates planner memory after each iteration
  - validation : read-only result validator that recommends submit/hold
"""

import os

from claude_agent_sdk import AgentDefinition

_model = os.environ.get("GLADIUS_MODEL") or ""

# ── Solver ────────────────────────────────────────────────────────────────────

_SOLVER_AGENT_DEF = AgentDefinition(
    description=(
        "Skills-first competition solver. Handles the full pipeline in a single "
        "autonomous session: exploring data, planning (via TodoWrite), implementing, "
        "evaluating, reviewing, and submitting. Searches and loads skills via the "
        "MCP skills-on-demand server before starting any task. Iterates continuously "
        "until results are satisfactory, then reports via StructuredOutput."
    ),
    prompt="""\
You are a competition solver. You handle everything: exploring data, planning,
implementing, evaluating, reviewing, and submitting.
No subagents. No coordinators. Your only collaborators are skills loaded via MCP.

> MANDATORY FIRST ACTION — before anything else:
> mcp__skills-on-demand__search_skills({"query": "<task type>", "top_k": 5})
> Then read the best-match .claude/skills/<name>/SKILL.md and follow its patterns.

Read CLAUDE.md first for competition context. NEVER modify CLAUDE.md.

PATH NOTE: .claude/EXPERIMENT_STATE.json is a local file inside the project directory.
Use it to track phase progress. Reset it (write `{}`) at the start of each experiment.

Execution loop:
1. Search skills → load SKILL.md → follow its patterns
2. Explore data, plan with TodoWrite
3. Implement in src/ and scripts/; install with uv add; run with uv run python
4. Print `OOF <metric>: <value>` in training script; save artifacts/oof.npy
5. Review for leakage, CV contamination, wrong metric, train/test mismatch
6. Build submission matching sample_submission.csv exactly (save to submissions/submission.csv)
7. Search for next improvement skill, reset state, iterate

Rules:
- uv add <pkg> to install (never pip install)
- pathlib everywhere; no hardcoded absolute paths
- random_state=42 for reproducibility
- TodoWrite for progress tracking
- Iterate continuously; call StructuredOutput only when done""",
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

# ── Summarizer ────────────────────────────────────────────────────────────────

_SUMMARIZER_AGENT_DEF = AgentDefinition(
    description=(
        "Expert ML research analyst that reviews experiment results and rewrites the "
        "solver memory file. Read-only: it never edits code or data files — it only "
        "reads existing files and returns structured analysis."
    ),
    prompt="""\
You are an expert ML research analyst maintaining a living knowledge base.

You review experiment results and produce a concise, structured update for the
solver's MEMORY.md file. You NEVER write files yourself — you return the
full updated memory content as structured output.

Always read the existing MEMORY.md before producing the update so you preserve
historical entries.""",
    # Read-only — the orchestrator writes MEMORY.md from the structured output.
    tools=["Read", "Grep"],
    model=_model,
)

# ── Validation ────────────────────────────────────────────────────────────────

_VALIDATION_AGENT_DEF = AgentDefinition(
    description=(
        "Validates experiment results and recommends whether to submit to the platform. "
        "Read-only: it never modifies files or state — it only observes and reports "
        "structured decisions (is_improvement, submit, reasoning)."
    ),
    prompt="""\
You are a competition result validator.

You compare new experiment scores against the current best, check submission
artifact format by reading files, query platform quota via MCP tools, and return
a structured JSON decision. You NEVER write files or mutate state.

STRICT RULES — you are READ-ONLY:
- NEVER run Bash commands.
- NEVER write, edit, or delete any files.
Use only Read, Grep, and any MCP quota tools provided.""",
    # MCP quota tools are injected per-call by run_validation_agent().
    tools=["Read", "Grep"],
    model=_model,
)

# ── Registry ──────────────────────────────────────────────────────────────────

SUBAGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "solver": _SOLVER_AGENT_DEF,
    "summarizer": _SUMMARIZER_AGENT_DEF,
    "validation": _VALIDATION_AGENT_DEF,
}
