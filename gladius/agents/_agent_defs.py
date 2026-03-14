"""
Agent definitions registry.

``SUBAGENT_DEFINITIONS`` is passed to every ``ClaudeAgentOptions(agents=...)``
call so that programmatic definitions take precedence over any
``.claude/agents/*.md`` files, and subagents inherit the session's permission
mode (``bypassPermissions`` from the coordinator).

All agent roles are defined in ``gladius.agents.roles.catalog`` (ROLE_CATALOG).
This module is a thin bridge that stamps the live model name into each entry
and exposes the registry that the runtime helpers expect.

Notes
-----
- ``_model`` is resolved at module load.  ``run_agent()`` / ``run_planning_agent()``
  re-read ``GLADIUS_MODEL`` at *call* time (after ``load_dotenv``) to pick up
  any ``.env`` override — the per-call helpers in ``_base.py`` subscribe a
  fresh copy via ``get_runtime_model()``.
- ``Agent()`` is intentionally omitted from worker role tool lists — subagents
  cannot spawn further subagents per the Claude Code docs.
"""

import os

from claude_agent_sdk import AgentDefinition

from gladius.agents.roles.catalog import ROLE_CATALOG as _ROLE_CATALOG

_model = os.environ.get("GLADIUS_MODEL") or ""

# ── Registry ──────────────────────────────────────────────────────────────────

SUBAGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    name: AgentDefinition(
        description=role.description,
        prompt=role.system_prompt,
        tools=list(role.tools),
        model=_model,
    )
    for name, role in _ROLE_CATALOG.items()
}
