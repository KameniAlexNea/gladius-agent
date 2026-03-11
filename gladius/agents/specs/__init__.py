"""Agent specifications — single Gladius agent."""

from gladius.agents.specs.gladius_spec import (
    GLADIUS_OUTPUT_SCHEMA,
    GLADIUS_SYSTEM_PROMPT,
    build_gladius_prompt,
)

# Backward-compatible aliases
SOLVER_OUTPUT_SCHEMA = GLADIUS_OUTPUT_SCHEMA
SOLVER_SYSTEM_PROMPT = GLADIUS_SYSTEM_PROMPT
build_solver_prompt = build_gladius_prompt

__all__ = [
    "GLADIUS_OUTPUT_SCHEMA",
    "GLADIUS_SYSTEM_PROMPT",
    "build_gladius_prompt",
    "SOLVER_OUTPUT_SCHEMA",
    "SOLVER_SYSTEM_PROMPT",
    "build_solver_prompt",
]
