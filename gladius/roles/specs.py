"""
Output schemas and task-prompt builders for each role.

Each role's output is captured as structured JSON via the SDK output_schema
mechanism. Prompt-builders inject iteration-specific context without embedding
it in the static system prompts in the role templates.
"""

from __future__ import annotations

from typing import Any


# ── Shared submission-builder schema ─────────────────────────────────────────

SUBMISSION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "submission_file"],
    "properties": {
        "status": {"type": "string", "enum": ["success", "error"]},
        "submission_file": {"type": "string"},
        "notes": {"type": "string"},
        "error_message": {"type": "string"},
    },
    "additionalProperties": False,
}

# ── team-lead ─────────────────────────────────────────────────────────────────

TEAM_LEAD_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["plan"],
    "properties": {
        "plan": {"type": "string"},
        "approach_summary": {"type": "string"},
    },
    "additionalProperties": False,
}


def build_team_lead_prompt(
    *,
    iteration: int,
    max_iterations: int,
    project_dir: str,
    n_parallel: int = 1,
) -> str:
    parallel_instruction = ""
    if n_parallel > 1:
        parallel_instruction = f"""
IMPORTANT: Generate exactly {n_parallel} independent approaches using this structure:

## Approach 1
(concrete ordered implementation plan)

## Approach 2
(concrete ordered implementation plan)

… up to Approach {n_parallel}.
Each approach must target a substantially different hypothesis:
feature-engineering focus / model-architecture focus / target-transformation focus /
data-shift-mitigation focus. Do NOT write two near-identical hyperparameter variants."""

    return f"""\
Use your current session context for competition state.

Iteration   : {iteration} / {max_iterations}
Project dir : {project_dir}

Decide the highest-impact next experiment. Produce a concise ordered strategy plan.
Use CLAUDE.md for current best scores and stagnation warnings.
Call ExitPlanMode when done — provide only the markdown plan text.
Do NOT include allowedPrompts or tool-approval payload fields in ExitPlanMode.

Skill discovery protocol (mandatory):
- `mcp__skills-on-demand__search_skills({{"query": "<task type>", "top_k": 5}})`
- `Skill({{"skill": "<name>"}})`
- Load only the single most relevant skill.
{parallel_instruction}
"""


# ── pipeline agent output schema (shared by all 4 specialist roles) ──────────

PIPELINE_AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "summary"],
    "properties": {
        "status": {"type": "string", "enum": ["success", "error"]},
        "summary": {"type": "string"},
        "files_modified": {"type": "array", "items": {"type": "string"}},
        "oof_score": {"type": ["number", "null"]},
        "quality_score": {"type": ["number", "null"]},
        "submission_file": {"type": "string"},
        "error_message": {"type": "string"},
    },
    "additionalProperties": False,
}

# ── iteration result schema (common across all topologies) ────────────────────

ITERATION_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "oof_score", "quality_score"],
    "properties": {
        "status": {"type": "string", "enum": ["success", "error", "timeout", "oom"]},
        "oof_score": {"type": ["number", "null"]},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 100},
        "solution_files": {"type": "array", "items": {"type": "string"}},
        "submission_file": {"type": "string"},
        "notes": {"type": "string"},
        "error_message": {"type": "string"},
        "total_turns": {"type": ["integer", "null"]},
    },
    "additionalProperties": False,
}

# ── validator ─────────────────────────────────────────────────────────────────

VALIDATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "oof_score",
        "quality_score",
        "is_improvement",
        "submit",
        "stop",
        "reasoning",
        "next_directions",
    ],
    "properties": {
        "oof_score": {"type": ["number", "null"]},
        "quality_score": {"type": ["number", "null"]},
        "is_improvement": {"type": "boolean"},
        "improvement_delta": {"type": ["number", "null"]},
        "submit": {"type": "boolean"},
        "submission_path": {"type": ["string", "null"]},
        "format_ok": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "stop": {"type": "boolean"},
        "next_directions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "critique_list": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


def build_validator_prompt(
    *,
    oof_score: float | None,
    quality_score: float | None,
    best_oof_score: float | None,
    best_quality_score: float | None,
    submission_path: str | None,
    target_metric: str | None,
    metric_direction: str | None,
    submission_quota_remaining: int,
    project_dir: str,
) -> str:
    if target_metric:
        score_block = f"""\
New OOF score   : {oof_score}
Current best    : {best_oof_score}
Metric          : {target_metric} ({metric_direction})
Submission file : {submission_path or 'not provided'}
Quota remaining : {submission_quota_remaining} submissions today"""
    else:
        score_block = f"""\
New quality score   : {quality_score}/100
Current best quality: {best_quality_score}/100
(open-ended task — no target metric)"""

    return f"""\
{score_block}

Project dir: {project_dir}

Assess the result and emit a StructuredOutput JSON judgement.
"""


# ── memory-keeper ──────────────────────────────────────────────────────────────

MEMORY_KEEPER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "memory_content"],
    "properties": {
        "summary": {"type": "string"},
        "memory_content": {"type": "string"},
    },
    "additionalProperties": False,
}


def build_memory_keeper_prompt(
    *,
    iteration: int,
    competition_id: str,
    target_metric: str | None,
    metric_direction: str | None,
    experiments: list[dict],
    failed_runs: list[dict],
    latest_result: dict,
    validator_notes: str,
) -> str:
    import json as _json

    recent_exps = experiments[-10:]
    recent_failed = failed_runs[-5:]
    return f"""\
Competition     : {competition_id}
Iteration       : {iteration}
Metric          : {target_metric or 'open-ended'} {metric_direction or ''}

## Latest result
{_json.dumps(latest_result, indent=2)}

## Validator notes
{validator_notes}

## Recent experiments (last 10)
{_json.dumps(recent_exps, indent=2)}

## Recent failed runs (last 5)
{_json.dumps(recent_failed, indent=2)}

Rewrite .claude/agent-memory/team-lead/MEMORY.md with fresh, concise learnings.
Keep all previous history. Return via StructuredOutput.
"""
