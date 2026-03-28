"""
Role catalog — loads all role definitions from src/roles/*.md.

Each file has YAML frontmatter (name, role, session, description, tools, model,
maxTurns) and a body that becomes the system_prompt.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import AgentDefinition

from gladius.config import LAYOUT as _LAYOUT

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


@dataclass
class RoleDefinition(AgentDefinition):
    """
    RoleDefinition extends AgentDefinition with gladius-specific fields.
    """

    name: str = ""
    session: str = "fresh"  # "persistent" | "fresh"


def _apply_path_placeholders(content: str) -> str:
    return (
        content.replace(
            "{{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}}",
            _LAYOUT.runtime_experiment_state_relative_path,
        )
        .replace(
            "{{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}}",
            _LAYOUT.runtime_data_briefing_relative_path,
        )
        .replace("{{TEAM_LEAD_MEMORY_RELATIVE_PATH}}", _LAYOUT.team_lead_memory_relative_path)
        .replace("{{RUNTIME_RELATIVE_PATH}}", _LAYOUT.runtime_relative_path)
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

    def _get_list(key: str) -> list[str] | None:
        m = re.search(rf"^{key}:\n((?:[ \t]+-[ \t]+.+\n?)+)", front, re.MULTILINE)
        if m:
            return [
                re.sub(r"^[ \t]+-[ \t]+", "", line).strip()
                for line in m.group(1).splitlines()
                if line.strip()
            ]
        return None

    tools_str = _get("tools")
    max_turns_str = _get("maxTurns")

    return RoleDefinition(
        name=_get("name"),
        session=_get("session"),
        description=_get_multiline("description"),
        tools=[t.strip() for t in tools_str.split(",") if t.strip()],
        model=_get("model"),
        maxTurns=int(max_turns_str) if max_turns_str.isdigit() else None,
        skills=_get_list("skills"),
        mcpServers=_get_list("mcpServers"),
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
