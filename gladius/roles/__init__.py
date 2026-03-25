"""
Role catalog — loads all role definitions from src/roles/*.md.

Each file has YAML frontmatter (name, role, session, description, tools, model,
maxTurns) and a body that becomes the system_prompt.
"""

from __future__ import annotations

import re
import sys  # noqa: E402
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gladius import (
    RUNTIME_DATA_BRIEFING_RELATIVE_PATH,
    RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH,
    RUNTIME_RELATIVE_PATH,
    TEAM_LEAD_MEMORY_RELATIVE_PATH,
)

_TEMPLATES = Path(__file__).parent / "templates"

ROLES = (
    "scout",
    "team-lead",
    "data-expert",
    "feature-engineer",
    "ml-engineer",
    "domain-expert",
    "evaluator",
    "validator",
    "memory-keeper",
    "full-stack-coordinator",
    "platform-coordinator",
)


@dataclass(frozen=True)
class RoleDefinition:
    """
    RoleDefinition aka AgentDefinition
    """

    name: str
    session: str  # "persistent" | "fresh"
    max_turns: int

    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None
    skills: list[str] | None = None
    memory: Literal["user", "project", "local"] | None = None
    # Each entry is a server name (str) or an inline {name: config} dict.
    mcpServers: list[str | dict[str, Any]] | None = None  # noqa: N815


def _apply_path_placeholders(content: str) -> str:
    return (
        content.replace(
            "{{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}}",
            RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH,
        )
        .replace(
            "{{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}}",
            RUNTIME_DATA_BRIEFING_RELATIVE_PATH,
        )
        .replace("{{TEAM_LEAD_MEMORY_RELATIVE_PATH}}", TEAM_LEAD_MEMORY_RELATIVE_PATH)
        .replace("{{RUNTIME_RELATIVE_PATH}}", RUNTIME_RELATIVE_PATH)
    )


def _parse(path: Path) -> RoleDefinition:
    text = _apply_path_placeholders(path.read_text(encoding="utf-8"))
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
        prompt=body,
    )


def _strip_web_search(content: str) -> str:
    """Remove WebSearch from a role template (tools list and body text)."""
    # Remove from frontmatter tools line: ", WebSearch" or "WebSearch, "
    content = re.sub(r",\s*WebSearch\b", "", content)
    content = re.sub(r"\bWebSearch\s*,\s*", "", content)
    # Replace the Scan step that requires WebSearch (team-lead specific)
    content = re.sub(
        r"4\. \*\*Scan\*\* — use `WebSearch`.*?skill catalog\.",
        (
            "4. **Scan** — Search arXiv for recent winning approaches: "
            '`mcp__arxiv-mcp-server__search_papers({"query": "<competition type> machine learning SOTA", "max_results": 5})`. '
            "Fall back to the `literature-review` skill if the server is unavailable."
        ),
        content,
        flags=re.DOTALL,
    )
    return content


def copy(
    dst: Path,
    spec: str | list,
    model: str,
    small_model: str,
    *,
    force: bool = False,
    use_web_search: bool = False,
) -> None:
    """Copy role .md files into dst (.claude/agents/), substituting model placeholders."""
    dst.mkdir(parents=True, exist_ok=True)

    if spec == "all":
        candidates = sorted(_TEMPLATES.glob("*.md"))
    else:
        names = [spec] if isinstance(spec, str) else list(spec)
        candidates = []
        for name in names:
            p = _TEMPLATES / f"{name}.md"
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
        content = _apply_path_placeholders(content)
        if not use_web_search:
            content = _strip_web_search(content)
        dest.write_text(content, encoding="utf-8")
        print(f"  agent  → .claude/agents/{src.name}")


ROLE_CATALOG: dict[str, RoleDefinition] = {
    r.name: r for r in (_parse(_TEMPLATES / f"{name}.md") for name in ROLES)
}

__all__ = ["ROLE_CATALOG", "RoleDefinition", "ROLES", "copy"]
