"""
Skill catalog — loads all custom skill definitions from src/skills/*/SKILL.md.

Each file has YAML frontmatter (name, description) and a body that agents
load at runtime via the Skill() tool call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent

_SKILL_NAMES = (
    "validation",
    "feature-engineering",
    "hpo",
    "ensembling",
)


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    body: str


def _parse(path: Path) -> SkillDefinition:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"No frontmatter in {path}")
    front, body = match.group(1), match.group(2).strip()

    def _get(key: str) -> str:
        # Single-line value: `name: foo`
        m = re.search(rf"^{key}:\s*(.+)$", front, re.MULTILINE)
        if not m:
            return ""
        value = m.group(1).strip()
        if value != ">":
            return value
        # Block scalar: `description: >\n  text...`
        bm = re.search(rf"^{key}:\s*>\n((?:[ \t]+.+\n?)+)", front, re.MULTILINE)
        if not bm:
            return ""
        return " ".join(line.strip() for line in bm.group(1).splitlines()).strip()

    return SkillDefinition(
        name=_get("name"),
        description=_get("description"),
        body=body,
    )


SKILL_CATALOG: dict[str, SkillDefinition] = {
    s.name: s
    for s in (
        _parse(_SKILLS_DIR / name / "SKILL.md")
        for name in _SKILL_NAMES
    )
}

__all__ = ["SKILL_CATALOG", "SkillDefinition"]
