"""
Code Agent — implements a hypothesis as a versioned experiment script.

Replaces: code_generator + code_reviewer + versioning_agent (3 nodes → 1)

Why one agent: Claude Code can read existing code, write the new solution,
run `python -m py_compile` to check it, and commit the version — all within
one autonomous query() call. No separate review pass needed.

Session continuity: resumed every iteration so the agent "remembers" the
codebase structure, naming conventions, and what it has already written.
"""
import json
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert ML engineer implementing Kaggle competition solutions in Python.

Given a hypothesis, you will:
1. Explore the existing solution code (src/ directory) — understand its structure
2. Implement the required changes cleanly
3. Write the new solution to a versioned file: src/solution_v{N}.py
   where N = next integer after the highest existing version
4. The solution MUST be self-contained and runnable:
     python src/solution_vN.py
   It must NOT require CLI arguments.
5. It must write:
     - OOF predictions  → .gladius/oof_vN.npy   (numpy array, shape [n_samples] or [n_samples, n_classes])
     - Test predictions → .gladius/sub_vN.csv   (submission format)
6. It must print the OOF score as the LAST line of stdout:
     OOF_SCORE: X.XXXXXX
7. Run `python -m py_compile src/solution_vN.py` to verify syntax
8. Update .gladius/experiments.json with an entry for this experiment

Code quality:
- Add a brief docstring describing the hypothesis
- Use type hints for function signatures
- Prefer scikit-learn pipelines for preprocessing
- Seed all random operations (numpy, sklearn, torch, etc.) with a fixed seed
- Do NOT import kaggle or access the internet

Always create atomic experiments. Never modify existing solution files.
"""

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["solution_path", "changes_made", "estimated_runtime_minutes"],
    "properties": {
        "solution_path": {
            "type": "string",
            "description": "Absolute or project-relative path to the new solution file",
        },
        "changes_made": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet-point list of what was changed vs. the previous version",
        },
        "estimated_runtime_minutes": {"type": "number"},
        "requires_gpu": {"type": "boolean"},
        "new_dependencies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New pip packages required (empty if none)",
        },
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}


async def run_code_agent(
    hypothesis: dict,
    state: "CompetitionState",
    project_dir: str,
) -> tuple[dict, str]:
    """
    Implement a hypothesis as a versioned solution script.

    Returns (code_result_dict, session_id).
    session_id should be stored in state.code_session_id.
    """
    prompt = f"""\
## Hypothesis to implement

{json.dumps(hypothesis, indent=2)}

## Context

Project directory : {project_dir}
Data directory    : {state.data_dir}
Current iteration : {state.iteration}
Target metric     : {state.target_metric} ({state.metric_direction})

## Instructions

1. Explore src/ to understand the current solution structure
2. Determine the next version number (highest existing vN + 1)
3. Implement this hypothesis as src/solution_vN.py
4. Ensure .gladius/ directory exists (mkdir -p .gladius)
5. Verify syntax: python -m py_compile src/solution_vN.py
6. Append an entry to .gladius/experiments.json:
   {{
     "version": N,
     "solution_path": "src/solution_vN.py",
     "hypothesis": "<one-line title>",
     "changes": [...],
     "iteration": {state.iteration},
     "status": "ready"
   }}

Return the JSON output now.
"""
    return await run_agent(
        agent_name="code",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        resume=state.code_session_id,
        max_turns=40,
    )
