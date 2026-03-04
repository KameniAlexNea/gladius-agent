"""Agent prompt/schema specifications.

Each spec module contains:
- SYSTEM_PROMPT constants
- OUTPUT_SCHEMA constants (where applicable)
- prompt builder helpers
"""

from gladius.agents.specs.implementer_spec import (
    IMPLEMENTER_OUTPUT_SCHEMA,
    IMPLEMENTER_SYSTEM_PROMPT,
    build_implementer_prompt,
)
from gladius.agents.specs.planner_spec import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_alternative_prompt,
    build_planner_prompt,
)
from gladius.agents.specs.summarizer_spec import (
    SUMMARIZER_OUTPUT_SCHEMA,
    SUMMARIZER_SYSTEM_PROMPT,
    build_summarizer_prompt,
)
from gladius.agents.specs.validation_spec import (
    VALIDATION_OUTPUT_SCHEMA,
    VALIDATION_SYSTEM_PROMPT,
    build_validation_prompt,
)

__all__ = [
    "IMPLEMENTER_OUTPUT_SCHEMA",
    "IMPLEMENTER_SYSTEM_PROMPT",
    "build_implementer_prompt",
    "PLANNER_SYSTEM_PROMPT",
    "build_planner_prompt",
    "build_planner_alternative_prompt",
    "VALIDATION_OUTPUT_SCHEMA",
    "VALIDATION_SYSTEM_PROMPT",
    "build_validation_prompt",
    "SUMMARIZER_OUTPUT_SCHEMA",
    "SUMMARIZER_SYSTEM_PROMPT",
    "build_summarizer_prompt",
]
