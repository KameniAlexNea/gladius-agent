"""
Role catalog — loads all role definitions from src/roles/*.md.

Each file has YAML frontmatter (name, role, session, description, tools, model,
maxTurns) and a body that becomes the system_prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TEMPLATES = Path(__file__).parent

ROLES = (
    "team-lead",
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "domain-expert",
    "evaluator",
    "validator",
    "memory-keeper",
    "full-stack-coordinator",
    # Topology-specific coordinator/specialist roles
    "functional-coordinator",
    "two-pizza-agent",
    "technical-review",
    "domain-review",
    "platform-layer",
    "product-layer",
)


@dataclass(frozen=True)
class RoleDefinition:
    name: str
    session: str       # "persistent" | "fresh"
    description: str
    tools: tuple[str, ...]
    model: str
    max_turns: int
    system_prompt: str


def _parse(path: Path) -> RoleDefinition:
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
    max_turns_str = _get("maxTurns")

    return RoleDefinition(
        name=_get("name"),
        session=_get("session"),
        description=_get_multiline("description"),
        tools=tuple(t.strip() for t in tools_str.split(",") if t.strip()),
        model=_get("model"),
        max_turns=int(max_turns_str) if max_turns_str.isdigit() else 0,
        system_prompt=body,
    )


ROLE_CATALOG: dict[str, RoleDefinition] = {
    r.name: r
    for r in (_parse(_TEMPLATES / f"{name}.md") for name in ROLES)
}

__all__ = ["ROLE_CATALOG", "RoleDefinition", "ROLES", "copy"]


import sys  # noqa: E402

_ROLES_DIR = Path(__file__).parent


def copy(dst: Path, spec: str | list, model: str, small_model: str, *, force: bool = False) -> None:
    """Copy role .md files into dst (.claude/agents/), substituting model placeholders."""
    dst.mkdir(parents=True, exist_ok=True)

    if spec == "all":
        candidates = sorted(_ROLES_DIR.glob("*.md"))
    else:
        names = [spec] if isinstance(spec, str) else list(spec)
        candidates = []
        for name in names:
            p = _ROLES_DIR / f"{name}.md"
            if p.is_file():
                candidates.append(p)
            else:
                print(f"  [warn] role not found: {name!r} — skipped", file=sys.stderr)

    for src in candidates:
        dest = dst / src.name
        if dest.exists() and not force:
            continue
        content = (
            src.read_text(encoding="utf-8")
            .replace("{{GLADIUS_MODEL}}", model)
            .replace("{{GLADIUS_SMALL_MODEL}}", small_model)
        )
        dest.write_text(content, encoding="utf-8")
        print(f"  agent  → .claude/agents/{src.name}")
