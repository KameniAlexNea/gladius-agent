"""
Execution Agent — runs a training script, monitors it, and reports results.

Replaces: executor + resource_manager + watchdog (3 nodes → 1)

Why one agent: Claude Code has native background-bash support (Bash tool with
run_in_background=True + BashOutput for polling + KillBash for termination).
The agent autonomously handles resource checks, timeout kills, and OOF extraction.

No session continuity: each execution is an independent, stateless task.
"""
import json
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are responsible for executing ML training runs on this machine.

Given a solution script path and time budget, you will:

1. CHECK RESOURCES
   - Run `nvidia-smi` if GPU is available; check VRAM usage
   - Run `free -h` to check RAM
   - If GPU VRAM < 2 GB free, wait 30s and check again (max 3 attempts)

2. LAUNCH
   - Run: python {solution_path}
   - Use Bash with run_in_background=True — save the shell_id

3. MONITOR (every 60 seconds via BashOutput)
   - Check if still running, completed, or failed
   - If the output contains "nan" or "inf" in a loss line → kill and report error
   - If it exceeds the time budget → kill and report timeout

4. EXTRACT RESULT
   - The last line of stdout must be: OOF_SCORE: X.XXXXXX
   - If not present, report error

5. REPORT
   - Return structured JSON with status, oof_score, runtime_seconds, etc.

Never modify source files. You only execute and observe.
"""

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "oof_score", "runtime_seconds"],
    "properties": {
        "status": {
            "enum": ["success", "timeout", "error", "oom", "nan_detected"],
            "description": "Outcome of the training run",
        },
        "oof_score": {
            "type": ["number", "null"],
            "description": "OOF score extracted from 'OOF_SCORE: X.XXXXXX'. Null on failure.",
        },
        "runtime_seconds": {"type": "number"},
        "stdout_tail": {
            "type": "string",
            "description": "Last ~20 lines of training output",
        },
        "error_message": {"type": ["string", "null"]},
        "peak_memory_gb": {"type": ["number", "null"]},
        "gpu_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


async def run_execution_agent(
    solution_path: str,
    max_runtime_minutes: int,
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """
    Execute a training script and return the result dict.

    No session_id is returned — execution agents are stateless.
    """
    prompt = f"""\
## Training Job

Solution path : {solution_path}
Time budget   : {max_runtime_minutes} minutes
Data directory: {state.data_dir}
Project dir   : {project_dir}

## Steps

1. Check GPU/RAM resources
2. Launch: python {solution_path}
   (use run_in_background=True, save shell_id)
3. Poll every 60 seconds; kill if > {max_runtime_minutes * 60} seconds elapsed
4. Detect NaN/Inf in loss output → kill immediately, status="nan_detected"
5. On completion, verify the last stdout line is: OOF_SCORE: X.XXXXXX
6. Return the JSON result

Note: the solution writes predictions to .gladius/ automatically.
"""
    result, _ = await run_agent(
        agent_name="execution",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Bash", "Read"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        # No resume — stateless
        max_turns=60,  # generous for monitoring loop
    )
    return result
