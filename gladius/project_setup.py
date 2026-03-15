"""Bootstrap a competition project directory from a YAML config."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

import gladius.claude_md as claude_md
import gladius.roles as roles
import gladius.skills as skills
import gladius.tools as tools
from gladius import team_lead_memory_path

_SRC = Path(__file__).parent
_TEMPLATES = _SRC

# ── Config loading ────────────────────────────────────────────────────────────

_VALID_PLATFORMS = {"kaggle", "zindi", "fake", "none"}
_VALID_TOPOLOGIES = {"functional", "two-pizza", "platform", "autonomous", "matrix"}
_VALID_DIRECTIONS = {"maximize", "minimize"}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")

    with p.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    for key in ("competition_id", "project_dir"):
        if not raw.get(key):
            raise ConfigError(f"Config missing required field: '{key}'")

    platform = str(raw.get("platform") or "none").strip().lower()
    if platform not in _VALID_PLATFORMS:
        raise ConfigError(f"platform must be one of {sorted(_VALID_PLATFORMS)}, got {platform!r}")
    raw["platform"] = platform

    has_metric = bool(raw.get("metric"))
    has_direction = bool(raw.get("direction"))
    if has_metric != has_direction:
        raise ConfigError("'metric' and 'direction' must both be provided or both omitted.")
    if has_direction:
        direction = str(raw["direction"]).strip().lower()
        if direction not in _VALID_DIRECTIONS:
            raise ConfigError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
        raw["direction"] = direction
    else:
        raw["metric"] = None
        raw["direction"] = None

    topology = str(raw.get("topology") or "functional").strip().lower()
    if topology not in _VALID_TOPOLOGIES:
        raise ConfigError(f"topology must be one of {sorted(_VALID_TOPOLOGIES)}, got {topology!r}")
    raw["topology"] = topology

    project_dir = Path(raw["project_dir"]).expanduser().resolve()
    raw["project_dir"] = str(project_dir)
    if raw.get("data_dir"):
        data_dir = Path(raw["data_dir"]).expanduser()
        raw["data_dir"] = str(data_dir if data_dir.is_absolute() else (project_dir / data_dir).resolve())
    else:
        raw["data_dir"] = str(project_dir / "data")

    raw.setdefault("model", os.environ.get("GLADIUS_MODEL", "claude-opus-4-5"))
    raw.setdefault("small_model", os.environ.get("GLADIUS_SMALL_MODEL", "inherit"))
    raw.setdefault("gladius_skills", "all")
    raw.setdefault("scientific_skills", True)
    raw.setdefault("scientific_skills_path", "")
    raw.setdefault("custom_skills_dir", "")
    raw.setdefault("roles", "all")
    mcp_cfg = raw.setdefault("mcp", {})
    mcp_cfg.setdefault("platform_server", True)
    mcp_cfg.setdefault("extra", {})
    raw.setdefault("force", False)

    raw.setdefault("default_mode", "acceptEdits")
    s = raw.setdefault("settings", {})
    s.setdefault("permissions_allow", [
        "Bash(uv *)",
        "Bash(python *)", "Bash(python3 *)",
        "Bash(ls *)", "Bash(cat *)", "Bash(head *)", "Bash(tail *)",
        "Bash(grep *)", "Bash(find *)", "Bash(wc *)", "Bash(sort *)", "Bash(unzip *)",
        "Bash(nohup *)", "Bash(ps *)", "Bash(kill *)", "Bash(wait *)",
        "Bash(mkdir *)", "Bash(cp *)", "Bash(mv *)", "Bash(touch *)", "Bash(chmod +x *)",
        "Bash(git status)", "Bash(git diff *)", "Bash(git log *)",
        "Bash(git add *)", "Bash(git commit *)",
        "Bash(echo *)", "Bash(which *)", "Bash(pwd)", "Bash(source *)",
        "Bash(* --help *)", "Bash(* --version)",
    ])
    s.setdefault("permissions_deny", [
        "Bash(rm *)",
        "Bash(sudo *)",
        "Bash(git push *)", "Bash(git reset --hard *)", "Bash(git clean *)",
        "Bash(pip *)", "Bash(pip3 *)",
        "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(~/.gnupg/**)", "Read(~/.env)",
    ])
    s.setdefault("additional_directories", [])

    return raw


# ── Setup steps ───────────────────────────────────────────────────────────────


def _dir_skeleton(root: Path) -> None:
    for sub in (
        root / ".claude" / "agents",
        root / ".claude" / "skills",
        root / ".claude" / "agent-memory",
        root / "src",
        root / "scripts",
        root / "artifacts",
        root / "submissions",
    ):
        sub.mkdir(parents=True, exist_ok=True)

    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("data/\nartifacts/\nsubmissions/\n*.db\n*.log\n.env\n", encoding="utf-8")


def _copy_hooks(root: Path, *, force: bool) -> None:
    hooks_src = _TEMPLATES / "hooks"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    for src in sorted(hooks_src.glob("*.sh")):
        dest = scripts_dir / src.name
        if dest.exists() and not force:
            continue
        shutil.copy2(src, dest)
        mode = dest.stat().st_mode
        dest.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  hook   → scripts/{src.name}")


def _write_settings(root: Path, cfg: dict) -> None:
    import json
    s = cfg["settings"]
    extra_dirs = list(s["additional_directories"] or [])
    if cfg["data_dir"] not in extra_dirs:
        extra_dirs = [cfg["data_dir"]] + extra_dirs
    settings = {
        "model": cfg["model"],
        "defaultMode": cfg["default_mode"],
        "env": {
            "COMPETITION_ID": cfg["competition_id"],
            "TARGET_METRIC": cfg["metric"] or "",
            "METRIC_DIRECTION": cfg["direction"] or "",
            "DATA_DIR": cfg["data_dir"],
            "TOPOLOGY": cfg["topology"],
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        },
        "permissions": {
            "allow": s["permissions_allow"],
            "deny": s["permissions_deny"],
        },
        "additionalDirectories": extra_dirs,
        "hooks": {
            "PostToolUse": [{"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "scripts/after_edit.sh"}]}],
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "scripts/validate_bash.sh"}]}],
        },
    }
    path = root / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print("  settings → .claude/settings.json")


def _seed_memory(root: Path, cfg: dict, *, force: bool) -> None:
    mem_file = team_lead_memory_path(root)
    if mem_file.exists() and not force:
        return
    template_name = "MEMORY-ml.md" if cfg["metric"] else "MEMORY-task.md"
    template = _TEMPLATES / "memory" / template_name
    if template.exists():
        mem_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, mem_file)
        print("  memory → .claude/agent-memory/team-lead/MEMORY.md")


# ── Main ──────────────────────────────────────────────────────────────────────


def setup(config_path: str | Path, *, force: bool = False) -> Path:
    cfg = load_config(config_path)
    if force:
        cfg["force"] = True

    root = Path(cfg["project_dir"])
    f = cfg["force"]

    print(f"\nSetting up: {root}")
    print(f"  competition: {cfg['competition_id']}  topology: {cfg['topology']}  platform: {cfg['platform']}")
    if cfg["metric"]:
        print(f"  metric: {cfg['metric']} ({cfg['direction']})")
    print()

    _dir_skeleton(root)
    roles.copy(root / ".claude" / "agents", cfg["roles"], cfg["model"], cfg["small_model"], force=f)
    skills.copy(root / ".claude" / "skills", cfg["gladius_skills"], force=f)
    if cfg["scientific_skills"]:
        skills.copy_scientific(root / ".claude" / "skills", cfg["scientific_skills_path"], force=f)
    skills.copy_custom(root / ".claude" / "skills", cfg["custom_skills_dir"])
    _copy_hooks(root, force=f)
    _write_settings(root, cfg)
    tools.write_mcp_json(root, cfg)
    claude_md.write_from_project(root, cfg)
    _seed_memory(root, cfg, force=f)

    print(f"\nDone. Project ready at: {root}\n")
    return root


# ── Runtime competition config (reads README.md frontmatter) ─────────────────


class CompetitionConfigError(ValueError):
    pass


def _parse_frontmatter(readme: Path) -> dict:
    """Parse YAML frontmatter from README.md."""
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise CompetitionConfigError(
            f"{readme}: must start with '---' to open the YAML frontmatter block."
        )
    end = text.find("\n---", 3)
    if end == -1:
        raise CompetitionConfigError(
            f"{readme}: frontmatter block never closed with '---'."
        )
    frontmatter_text = text[3:end]
    try:
        cfg = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise CompetitionConfigError(
            f"{readme}: invalid YAML frontmatter: {exc}"
        ) from exc
    if not isinstance(cfg, dict):
        raise CompetitionConfigError(f"{readme}: frontmatter must be a YAML mapping.")
    return {k: v if isinstance(v, (int, float, bool)) else str(v) for k, v in cfg.items()}


def load_competition_config(competition_dir: str) -> dict:
    """
    Parse README.md frontmatter in competition_dir.

    Returns dict with keys:
        competition_id, platform, metric (may be None), direction (may be None),
        data_dir (absolute path), topology, submission_threshold (may be None)

    Raises CompetitionConfigError if README.md is missing, has no frontmatter,
    is missing competition_id, or provides metric without direction (or vice-versa).
    """
    readme = Path(competition_dir) / "README.md"
    if not readme.exists():
        raise CompetitionConfigError(
            f"No README.md in {competition_dir!r}. "
            "Add one with a YAML frontmatter block:\n\n"
            "    ---\n"
            "    competition_id: my-competition\n"
            "    platform: kaggle\n"
            "    metric: auc_roc\n"
            "    direction: maximize\n"
            "    data_dir: data\n"
            "    ---\n"
        )

    cfg = _parse_frontmatter(readme)

    if not cfg.get("competition_id"):
        raise CompetitionConfigError(
            "README.md frontmatter missing required field: competition_id"
        )

    platform = cfg.get("platform") or "none"
    if platform not in ("kaggle", "zindi", "fake", "none"):
        raise CompetitionConfigError(
            f"platform must be kaggle | zindi | fake | none, got {platform!r}"
        )
    cfg["platform"] = platform

    has_metric = bool(cfg.get("metric"))
    has_direction = bool(cfg.get("direction"))
    if has_metric != has_direction:
        raise CompetitionConfigError(
            "'metric' and 'direction' must both be provided or both omitted."
        )
    if has_direction and cfg["direction"] not in ("maximize", "minimize"):
        raise CompetitionConfigError(
            f"direction must be maximize | minimize, got {cfg['direction']!r}"
        )

    cfg["metric"] = cfg.get("metric") or None
    cfg["direction"] = cfg.get("direction") or None

    _VALID_TOPOLOGIES = ("functional", "two-pizza", "platform", "autonomous", "matrix")
    topology = cfg.get("topology") or "functional"
    if topology not in _VALID_TOPOLOGIES:
        raise CompetitionConfigError(
            f"topology must be one of {' | '.join(_VALID_TOPOLOGIES)}, got {topology!r}"
        )
    cfg["topology"] = topology

    raw_threshold = cfg.get("submission_threshold")
    if raw_threshold is not None:
        try:
            cfg["submission_threshold"] = float(raw_threshold)
        except (ValueError, TypeError):
            raise CompetitionConfigError(
                f"submission_threshold must be a number, got {raw_threshold!r}"
            )
    else:
        cfg["submission_threshold"] = None

    p = Path(cfg.get("data_dir") or "data")
    if not p.is_absolute():
        p = Path(competition_dir) / p
    cfg["data_dir"] = str(p.resolve())

    return cfg
