"""Compatibility exports for agent runtime.

Implementation has been split across:
- gladius.agents.runtime.helpers
- gladius.agents.runtime.agent_runner
- gladius.agents.runtime.planning_runner
"""

from gladius.agents.runtime.agent_runner import run_agent
from gladius.agents._agent_defs import (
    SUBAGENT_DEFINITIONS as _SUBAGENT_DEFINITIONS,
)
from gladius.agents.runtime.helpers import (
    build_runtime_agents as _build_runtime_agents,
    get_runtime_model as _get_runtime_model,
    is_bash_command_scoped_to_cwd as _is_bash_command_scoped_to_cwd,
    is_tool_allowed as _is_tool_allowed,
    stderr_cb as _stderr_cb,
    validate_runtime_invocation as _validate_runtime_invocation,
)
from gladius.agents.runtime.planning_runner import (
    approve_exit_plan_mode as _approve_exit_plan_mode,
    run_planning_agent,
)

__all__ = [
    "run_agent",
    "run_planning_agent",
    "_SUBAGENT_DEFINITIONS",
    "_validate_runtime_invocation",
    "_get_runtime_model",
    "_build_runtime_agents",
    "_stderr_cb",
    "_is_tool_allowed",
    "_is_bash_command_scoped_to_cwd",
    "_approve_exit_plan_mode",
]
