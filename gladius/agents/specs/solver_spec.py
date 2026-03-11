"""Backward-compatible re-export. Use gladius_spec directly."""

from gladius.agents.specs.gladius_spec import (  # noqa: F401
    GLADIUS_OUTPUT_SCHEMA as SOLVER_OUTPUT_SCHEMA,
    GLADIUS_SYSTEM_PROMPT as SOLVER_SYSTEM_PROMPT,
    build_gladius_prompt as build_solver_prompt,
)

__all__ = ["SOLVER_OUTPUT_SCHEMA", "SOLVER_SYSTEM_PROMPT", "build_solver_prompt"]
