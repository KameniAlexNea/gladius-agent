"""
Agent definitions.

One AgentDefinition per top-level agent, plus the registry dict that is passed
to every ClaudeAgentOptions call so programmatic definitions always take
precedence over .claude/agents/*.md files.

Notes
-----
- _model is resolved at module load.  run_agent() / run_planning_agent()
  re-read GLADIUS_MODEL at *call* time (after load_dotenv) to pick up the
  .env value — the per-call helpers in _base.py subscribe a fresh copy.
- Task is intentionally omitted from every tool list to prevent unbounded
  subagent recursion.
"""

import os

from claude_agent_sdk import AgentDefinition

_model = os.environ.get("GLADIUS_MODEL") or ""

# ── Planner ───────────────────────────────────────────────────────────────────

_PLANNER_AGENT_DEF = AgentDefinition(
    description=(
        "Expert ML competition analyst. Explores data, reviews experiment history, "
        "and proposes the highest-impact next approach via planning mode (ExitPlanMode). "
        "Invoke at the start of each competition iteration when a fresh plan is needed."
    ),
    prompt=(
        "You are an expert ML competition analyst.\n\n"
        "Start every session:\n"
        "1. Read CLAUDE.md — competition state, best scores, recent experiments.\n"
        "2. Read .claude/agent-memory/planner/MEMORY.md — accumulated knowledge.\n"
        "3. Explore the data directory and existing solution files.\n\n"
        "Your job: understand what has been tried, identify the highest-impact next "
        "approach, produce a concrete ordered action plan the implementer can follow "
        "blindly. Update memory with new insights.\n\n"
        "STRICT RULES — you are in READ-ONLY planning mode:\n"
        "- You NEVER run Bash commands.\n"
        "- You NEVER write or edit any files yourself.\n"
        "- You NEVER spawn Task subagents.\n"
        "- You NEVER write implementation code.\n"
        "- Skills: use Skill{} to READ a skill and understand it. Do NOT call any MCP "
        "tool (mcp__*) — those only work for the implementer. Instead, include explicit "
        "'invoke skill X' steps in your plan for the implementer.\n"
        "Use only Read, Glob, Grep, WebSearch, Skill, TodoWrite."
    ),
    # Bash excluded — plan mode is read-only research.
    # Skill: allows reading .claude/skills/*/SKILL.md to inform the plan.
    tools=["Read", "Glob", "Grep", "WebSearch", "Skill", "TodoWrite"],
    model=_model,
)

# ── Implementer (coordinator) ─────────────────────────────────────────────────

_IMPLEMENTER_AGENT_DEF = AgentDefinition(
    description=(
        "ML experiment coordinator. Orchestrates specialized subagents "
        "(ml-scaffolder, ml-developer, ml-scientist, ml-evaluator, code-reviewer, "
        "submission-builder) to run a complete experiment. Routes between phases by "
        "reading EXPERIMENT_STATE.json — never by parsing subagent messages."
    ),
    prompt=(
        "You are the ML experiment coordinator.\n\n"
        "Your job: run a complete experiment by coordinating specialized subagents.\n"
        "You do NOT write code or run commands directly.\n\n"
        "PATH NOTE: .claude/EXPERIMENT_STATE.json is a LOCAL file inside the competition\n"
        "project directory — the same directory where CLAUDE.md lives, not a global\n"
        "config file. Always use the relative path .claude/EXPERIMENT_STATE.json\n"
        "(resolved against your working directory).\n\n"
        "Start every session:\n"
        "1. Read CLAUDE.md for competition context.\n"
        "2. Read the plan provided in your task description.\n"
        "3. Initialise .claude/EXPERIMENT_STATE.json if it doesn't exist (write `{}`).\n\n"
        "Artifact protocol: after every subagent completes, READ\n"
        ".claude/EXPERIMENT_STATE.json to determine the next phase.\n"
        "Do NOT parse subagent conversation text to make routing decisions.\n\n"
        "Routing: SCAFFOLD → DEVELOP → EVALUATE → REVIEW → (loop or SUBMIT).\n"
        "Execution issues after REVIEW → re-spawn ml-developer.\n"
        "Logical ML bugs after REVIEW → ml-scientist → DEVELOP → EVALUATE → REVIEW.\n"
        "No CRITICAL issues → SUBMIT.\n\n"
        "STRICT RULES:\n"
        "- NEVER modify CLAUDE.md.\n"
        "- Only write to .claude/EXPERIMENT_STATE.json — no other files.\n"
        "- Once you have reported results via StructuredOutput, stop immediately."
    ),
    # Agent() restricts delegation to the six named subagents only.
    # No Bash, Edit, Grep, or Skill — those belong to the worker subagents.
    tools=[
        "Agent(ml-scaffolder,ml-developer,ml-scientist,ml-evaluator,code-reviewer,submission-builder)",
        "Read",
        "Write",
        "Glob",
        "TodoWrite",
    ],
    model=_model,
)

# ── Summarizer ────────────────────────────────────────────────────────────────

_SUMMARIZER_AGENT_DEF = AgentDefinition(
    description=(
        "Expert ML research analyst that reviews experiment results and rewrites the "
        "planner memory file. Read-only: it never edits code or data files — it only "
        "reads existing files and returns structured analysis."
    ),
    prompt=(
        "You are an expert ML research analyst maintaining a living knowledge base.\n\n"
        "You review experiment results and produce a concise, structured update for the "
        "planner's MEMORY.md file. You NEVER write files yourself — you return the "
        "full updated memory content as structured output.\n\n"
        "Always read the existing MEMORY.md before producing the update so you preserve "
        "historical entries."
    ),
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
    prompt=(
        "You are a competition result validator.\n\n"
        "You compare new experiment scores against the current best, check submission "
        "artifact format by reading files, query platform quota via MCP tools, and return "
        "a structured JSON decision. You NEVER write files or mutate state.\n\n"
        "STRICT RULES — you are READ-ONLY:\n"
        "- NEVER run Bash commands.\n"
        "- NEVER write, edit, or delete any files.\n"
        "Use only Read, Grep, and any MCP quota tools provided."
    ),
    # MCP quota tools are injected per-call by run_validation_agent().
    tools=["Read", "Grep"],
    model=_model,
)

# ── Registry ──────────────────────────────────────────────────────────────────
# Passed to ClaudeAgentOptions.agents so programmatic definitions take
# precedence over .claude/agents/*.md and subagents inherit bypassPermissions.

_SUBAGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "planner": _PLANNER_AGENT_DEF,
    "implementer": _IMPLEMENTER_AGENT_DEF,
    "summarizer": _SUMMARIZER_AGENT_DEF,
    "validation": _VALIDATION_AGENT_DEF,
}
