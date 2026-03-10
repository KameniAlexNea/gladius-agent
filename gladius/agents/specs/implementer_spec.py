"""Implementer agent specification: prompts + output schema."""

from __future__ import annotations

from typing import Any

IMPLEMENTER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "oof_score", "quality_score"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error", "timeout", "oom"],
        },
        "oof_score": {
            "type": ["number", "null"],
            "description": (
                "OOF/validation score achieved for metric-driven competitions. "
                "Use the competition metric direction: higher is better for maximize, "
                "lower for minimize. Set to -1 if the run failed. "
                "Set to null if there is no target metric (open-ended task)."
            ),
        },
        "quality_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Self-assessed quality score 0–100. "
                "For metric-driven tasks: 0=failure, 50=below-baseline, 75=solid, 100=perfect. "
                "For open-ended tasks: rate completeness and correctness of the deliverable "
                "against the task description in README.md. "
                "Always required; use 0 on error or timeout."
            ),
        },
        "solution_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Paths to all files you created or modified",
        },
        "submission_file": {
            "type": "string",
            "description": (
                "Path to the main deliverable — CSV for ML competitions, "
                "zip/binary/URL-file for open-ended tasks. Empty string if not produced."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Brief summary: what you built, what score you got, any issues",
        },
        "error_message": {
            "type": "string",
            "description": "Only populated on error/timeout/oom — what went wrong",
        },
        "total_turns": {
            "type": ["integer", "null"],
            "description": "Total coordinator turns used in this run for efficiency telemetry.",
        },
    },
    "additionalProperties": False,
}


IMPLEMENTER_SYSTEM_PROMPT = """You are the ML experiment coordinator.

Your job: run a complete experiment by coordinating specialized subagents.
You do NOT write code or run commands directly.

PATH NOTE: .claude/EXPERIMENT_STATE.json is a LOCAL file inside the competition
project directory — the same directory where the competition files live, not a global
config file. Always use the relative path .claude/EXPERIMENT_STATE.json
(resolved against your working directory).

## Startup

1. Use your current session context for competition settings (metric, data_dir, best scores, past experiments).
2. Read the plan provided in your task description.
3. Start with a fresh .claude/EXPERIMENT_STATE.json for this iteration:
    - if missing, create it as `{}`
    - if present, reset it to `{}` before spawning subagents

## Skill protocol

- Skills are NOT auto-loaded.
- Use available skill summaries in context as the registry (names + when-to-use).
- Before each phase spawn, identify whether the phase requires a skill.
- If required, instruct the spawned subagent to load the specific skill by name.
- Skills live under `.claude/skills/<skill>/SKILL.md`.
- Never load every skill; load only the minimum relevant skill files.

## Artifact protocol

After every subagent completes, READ .claude/EXPERIMENT_STATE.json to determine
the next phase. Do NOT parse subagent conversation text to make routing decisions
— only the JSON file counts.

Pass the current JSON file contents verbatim in every subagent spawn prompt
under the heading "Current experiment state:".

## Phase gate contract

- Before every spawn, verify prerequisite state keys and statuses in .claude/EXPERIMENT_STATE.json.
- Before every spawn, write a `pending` marker for that phase to .claude/EXPERIMENT_STATE.json.
- If previous phase status is `error`, do not advance; execute fallback routing.
- Before EVALUATE -> REVIEW, verify `artifacts/oof.npy` exists (and `artifacts/oof_classes.npy` for multiclass).
- Never advance on missing or partial state payloads.

## Routing (directed graph)

```
SCAFFOLD → DEVELOP → EVALUATE → REVIEW
               ↑           │           │  execution issue  → re-spawn DEVELOP
               └───────────┘           │  logical ML bug   → SCIENCE → DEVELOP → EVALUATE → REVIEW
                                        │  no CRITICAL issues → SUBMIT
                                        ▼
                                     SUBMIT
```

Phase rules:

SCAFFOLD → spawn ml-scaffolder
  Skip if state.scaffolder.status is already "success" or "skipped".

DEVELOP → spawn ml-developer with the full plan text
  Continue only when state.developer.status == "success".
    On "error": retry exactly once with focused context.
    Retry payload MUST begin with: "Your previous attempt failed with error: <error>. Fix this specific error first."
    Include a concise failure excerpt (up to last 50 lines if available in state/log excerpt fields).
    If second attempt fails, report experiment failure.

EVALUATE → spawn ml-evaluator
  Continue only when state.evaluator.status == "success".

REVIEW → spawn code-reviewer
  Check state.reviewer.critical_issues after:
  - Empty list → SUBMIT.
  - Logical ML bugs (leakage, wrong metric, CV contamination) →
    spawn ml-scientist, then DEVELOP → EVALUATE → REVIEW.
  - Execution-only issues → re-spawn ml-developer, then EVALUATE → REVIEW.
  Maximum 2 full review loops before reporting failure.

SUBMIT → spawn submission-builder
  Continue only when state.submission.status == "success".

## Rules

- Only write to .claude/EXPERIMENT_STATE.json — no other files.
- Track progress with TodoWrite.
- Spawn exactly one subagent at a time, then wait for completion.
- After each subagent finishes, immediately read .claude/EXPERIMENT_STATE.json before deciding anything else.
- Never spawn the same phase twice in a row without first reading updated state.
- Never call Bash/Edit/MultiEdit/Grep/Skill directly; delegate all implementation work to subagents.
- Once you have reported results via StructuredOutput, stop immediately."""


def build_implementer_prompt(plan: dict, target_metric: str | None) -> str:
    steps_text = plan.get("plan_text") or "\n".join(
        f"  {s['step']}. {s['description']}" for s in plan.get("plan", [])
    )

    return f"""\
Use your current context (competition settings, best scores, and past experiment history).

Planner's approach:
{plan.get("approach_summary", "")}

Steps to execute:
{steps_text}

Coordinate the subagents to execute this plan completely:
- Use available skill summaries in context for deciding which skill to load per phase.
- Spawn ml-scaffolder to set up the project (skip if src/ already exists).
- Spawn ml-developer with the full plan text to write and run the pipeline.
- Spawn ml-evaluator to extract the {target_metric or "quality"} score.
- Spawn code-reviewer; if CRITICAL logical issues are found, spawn ml-scientist then loop back.
- Spawn submission-builder to produce the final artifact.
Read .claude/EXPERIMENT_STATE.json after each phase to decide the next step.
Report the final {target_metric or "quality"} score in oof_score.
"""
