"""
Lightweight topology catalog — parses *.md frontmatter only, no SDK deps.

Separated from __init__.py so callers that only need TOPOLOGY_CATALOG
(e.g. claude_md.py → project_setup.py) do not transitively import the SDK.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TEMPLATES = Path(__file__).parent


@dataclass(frozen=True)
class TopologyDefinition:
    name: str
    style: str
    flow: str
    claude_md_section: str


def _parse(path: Path) -> TopologyDefinition:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"No frontmatter in {path}")
    front, body = match.group(1), match.group(2).strip()

    def _get(key: str) -> str:
        m = re.search(rf"^{key}:\s*(.+)$", front, re.MULTILINE)
        return m.group(1).strip() if m else ""

    return TopologyDefinition(
        name=_get("name"),
        style=_get("style"),
        flow=_get("flow"),
        claude_md_section=body,
    )


TOPOLOGY_CATALOG: dict[str, TopologyDefinition] = {
    t.name: t
    for t in (
        _parse(_TEMPLATES / f"{name}.md")
        for name in ("functional", "two-pizza", "platform", "autonomous", "matrix")
    )
}
