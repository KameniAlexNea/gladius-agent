"""
Role catalog — loads all role definitions from agent templates.

Single source of truth: gladius/utils/templates/agents/<name>.md
Each template has YAML frontmatter (name, description, tools, permissionMode)
and a body that becomes the system_prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TEMPLATES = Path(__file__).parent.parent.parent / "utils" / "templates" / "agents"


@dataclass(frozen=True)
class RoleDefinition:
    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]


def _parse_template(path: Path) -> RoleDefinition:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"No frontmatter in {path}")
    front, body = match.group(1), match.group(2).strip()

    def _get(key: str) -> str:
        m = re.search(rf"^{key}:\s*(.+)$", front, re.MULTILINE)
        return m.group(1).strip() if m else ""

    def _get_multiline(key: str) -> str:
        m = re.search(rf"^{key}:\s*>\n((?:  .+\n?)+)", front, re.MULTILINE)
        if m:
            return " ".join(line.strip() for line in m.group(1).splitlines()).strip()
        return _get(key)

    tools_str = _get("tools")
    return RoleDefinition(
        name=_get("name"),
        description=_get_multiline("description"),
        system_prompt=body,
        tools=tuple(t.strip() for t in tools_str.split(",") if t.strip()),
    )


ROLE_CATALOG: dict[str, RoleDefinition] = {
    role.name: role
    for role in (
        _parse_template(_TEMPLATES / f"{name}.md")
        for name in (
            # Worker roles (spawned as subagents)
            "team-lead",
            "data-expert",
            "feature-engineer",
            "ml-engineer",
            "domain-expert",
            "evaluator",
            "validator",
            "memory-keeper",
            # Coordinator roles (spawned directly by run_agent)
            "functional-coordinator",
            "two-pizza-agent",
            "platform-layer",
            "product-layer",
            "technical-review",
            "domain-review",
        )
    )
}

__all__ = ["ROLE_CATALOG", "RoleDefinition"]
