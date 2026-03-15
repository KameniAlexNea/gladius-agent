"""
Agent definitions registry.

``SUBAGENT_DEFINITIONS`` is passed to every ``ClaudeAgentOptions(agents=...)``
call so that programmatic definitions take precedence over any
``.claude/agents/*.md`` files, and subagents inherit the session's permission
mode (``bypassPermissions`` from the coordinator).

All agent roles are defined in ``src.roles`` (ROLE_CATALOG).
This module stamps the live model name into each entry and exposes the
registry that the runtime helpers expect.
"""

import os

from claude_agent_sdk import AgentDefinition

from gladius.roles import ROLE_CATALOG as _ROLE_CATALOG

_model = os.environ.get("GLADIUS_MODEL") or ""

# ── Registry ──────────────────────────────────────────────────────────────────

# Worker roles exposed as subagents to coordinator agents
_WORKER_ROLES = (
    "team-lead",
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "domain-expert",
    "evaluator",
    "validator",
    "memory-keeper",
)

SUBAGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    name: AgentDefinition(
        description=role.description,
        prompt=role.system_prompt,
        tools=list(role.tools),
        model=_model,
    )
    for name, role in _ROLE_CATALOG.items()
    if name in _WORKER_ROLES
}
