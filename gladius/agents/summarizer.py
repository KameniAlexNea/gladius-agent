"""
Summarizer Agent — updates the planner's persistent memory after each iteration.

Reads the latest experiment result and current MEMORY.md, then rewrites MEMORY.md
with compact, structured learnings: what worked, what failed, patterns, next directions.

Called by the orchestrator at the end of every validation phase so the planner
accumulates knowledge across iterations without reading full experiment history
each time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent
from gladius.agents.specs.summarizer_spec import (
    SUMMARIZER_OUTPUT_SCHEMA,
    SUMMARIZER_SYSTEM_PROMPT,
    build_summarizer_prompt,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# Backwards-compatible aliases for existing imports/tests.
SYSTEM_PROMPT = SUMMARIZER_SYSTEM_PROMPT
OUTPUT_SCHEMA = SUMMARIZER_OUTPUT_SCHEMA


async def run_summarizer(
    state: "CompetitionState",
    project_dir: str,
    latest_experiment: dict,
    validation_notes: str = "",
) -> str:
    """
    Update .claude/agent-memory/planner/MEMORY.md with learnings from the latest
    experiment. Returns the one-sentence summary string.
    """
    memory_path = (
        Path(project_dir) / ".claude" / "agent-memory" / "planner" / "MEMORY.md"
    )

    prompt = build_summarizer_prompt(
        iteration=state.iteration,
        competition_id=state.competition_id,
        target_metric=state.target_metric,
        metric_direction=state.metric_direction,
        best_oof_score=state.best_oof_score,
        best_quality_score=state.best_quality_score,
        experiments=state.experiments,
        failed_runs=state.failed_runs,
        latest_experiment=latest_experiment,
        validation_notes=validation_notes,
        memory_path=str(memory_path),
    )
    result, _ = await run_agent(
        agent_name="summarizer",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Grep"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=15,
    )
    # Write MEMORY.md from Python — the summarizer has no Write permission.
    memory_content = result.get("memory_content", "")
    if memory_content:
        # Strip any code fences the LLM may have wrapped the content in.
        # Handles: ```markdown\n...\n``` or ```\n...\n``` (any language tag).
        # Loops in case of multiple/nested wrapping.
        import re

        stripped = memory_content.strip()
        while True:
            cleaned = re.sub(r"^```[^\n]*\n", "", stripped)  # remove opening fence line
            cleaned = re.sub(r"\n```\s*$", "", cleaned)  # remove closing fence line
            cleaned = cleaned.strip()
            if cleaned == stripped:
                break  # no more fences to strip
            stripped = cleaned
        # Remove leading/trailing --- separators the LLM sometimes adds
        stripped = re.sub(r"^---\s*\n", "", stripped)
        stripped = re.sub(r"\n---\s*$", "", stripped).strip()
        memory_content = stripped
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(memory_content, encoding="utf-8")
    return result.get("summary", "")
