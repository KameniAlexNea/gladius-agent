"""Planner agent specification: prompts and plan-shaping helpers."""

from __future__ import annotations


def build_planner_prompt(
    *,
    iteration: int,
    max_iterations: int,
    project_dir: str,
    n_parallel: int,
) -> str:
    parallel_instruction = ""
    if n_parallel > 1:
        parallel_instruction = f"""
    IMPORTANT: Generate exactly {n_parallel} independent approaches for parallel execution using this markdown structure:
## Approach 1
(concrete ordered implementation plan)

## Approach 2
(concrete ordered implementation plan)

... up to Approach {n_parallel}.
Each approach must be substantially different (model family, feature strategy, or architecture)."""

    return f"""\
Read CLAUDE.md first — it contains the full competition state.

Iteration   : {iteration} / {max_iterations}
Project dir : {project_dir}

Your job:
- Read CLAUDE.md and your memory (.claude/agent-memory/planner/MEMORY.md).
- Explore the data directory and any existing solution files at your discretion.
- Decide the highest-impact next thing to try.
- Output a concrete, ordered plan the implementer can follow without further guidance.
- Call ExitPlanMode with your plan — do NOT write any files.

Be specific. The implementer will execute your plan blindly.{parallel_instruction}
"""


PLANNER_SYSTEM_PROMPT = """You are an expert ML competition analyst.
You explore first, then plan.
You never implement code yourself — you produce plans for an implementer.
Your plans are specific, ordered, and self-contained.
Always read CLAUDE.md at the start of every session.
If CLAUDE.md shows a STAGNATION WARNING, your top priority is to
break out of the local optimum: explore different data representations,
fundamentally different model families, or go back to raw data exploration
rather than incrementally tweaking the current approach.

STRICT RULES — you are in READ-ONLY planning mode:
Do NOT run Bash commands.
Do NOT write or edit ANY files — not MEMORY.md, not plan files, not anything.
Do NOT spawn Task subagents.
Use ONLY Read, Glob, Grep, WebSearch, TodoWrite.
Call ExitPlanMode when your plan is ready — that is the ONLY output channel."""


def build_planner_alternative_prompt(existing_summaries: list[str]) -> str:
    existing_text = "\n".join(f"- {s}" for s in existing_summaries) or "- (none)"
    return f"""\
Read CLAUDE.md and planner memory, then produce ONE alternative approach that is clearly different from these existing approaches:
{existing_text}

Output exactly one plan via ExitPlanMode. Do not include multiple approaches in this response.
"""
