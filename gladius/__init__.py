# Gladius Improved Architecture

from pathlib import Path


def team_lead_memory_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / ".claude" / "agent-memory" / "team-lead" / "MEMORY.md"