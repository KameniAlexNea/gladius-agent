"""
Execution-phase project setup script.

Reads a YAML config file and bootstraps (or refreshes) a competition project
directory with:

  .claude/agents/       — role definitions from src/roles/
  .claude/skills/       — custom skills from src/skills/ + optional upstream
                          scientific skills from claude-scientific-skills/
  .claude/settings.json — model + hooks configuration
  .mcp.json             — MCP server registrations (platform tools from src/tools/)
  CLAUDE.md             — placeholder; src/claude_md.py overwrites it on first gladius run
  scripts/              — hook scripts (after_edit.sh, validate_bash.sh)
  artifacts/            — output directory (created empty)
  submissions/          — submission staging directory (created empty)

All writes are idempotent — existing files are **never overwritten** unless
flagged with `force: true` in the config or --force is passed on the CLI.
The only exception: CLAUDE.md is always overwritten (config may change).

Usage
-----
  # Bootstrap from a config file:
  python -m src.project_setup --config project.yaml

  # Force-refresh agents and skills (e.g. after editing src/roles/):
  python -m src.project_setup --config project.yaml --force

  # Scaffold a starter config next to this README:
  python -m src.project_setup init --out project.yaml

Config schema (project.yaml)
-----------------------------
See examples/project.yaml for an annotated reference.

Required fields:
  competition_id  str            — slug for logging / MCP calls
  project_dir     str            — target directory to set up (created if absent)

Optional fields (sensible defaults):
  platform        str            kaggle | zindi | fake | none   (default: none)
  metric          str|null       e.g. auc_roc — omit for open-ended tasks
  direction       str|null       maximize | minimize
  data_dir        str            path to data (default: <project_dir>/data)
  topology        str            functional | two-pizza | platform | autonomous | matrix
                                 (default: functional)
  model           str            Claude model slug (default: env GLADIUS_MODEL)
  small_model     str            Smaller model for cheap roles (default: inherit)

  gladius_skills  "all" | list   Skills from src/skills/ + gladius/utils/templates/skills/
                                 (default: all)
  scientific_skills  bool        Copy claude-scientific-skills catalog (default: true)
  scientific_skills_path  str    Override path to the scientific-skills directory

  custom_skills_dir  str         Path to a user-owned folder of additional skills.
                                 Every subdirectory with a SKILL.md is copied.
                                 Overrides any gladius skill of the same name.

  roles           "all" | list   Roles to copy from src/roles/    (default: all)

  mcp
    platform_server   bool       Register platform MCP server      (default: true)
    extra             dict       Additional mcpServers entries to merge

  force           bool           Overwrite existing files          (default: false)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

# ── Repo-relative helpers ─────────────────────────────────────────────────────

_SRC = Path(__file__).parent                              # gladius-agent/src/
_REPO = _SRC.parent                                       # gladius-agent/
_TEMPLATES = _REPO / "gladius" / "utils" / "templates"   # existing template tree
_SRC_ROLES = _SRC / "roles"                              # src/roles/*.md
_SRC_SKILLS = _SRC / "skills"                             # src/skills/<name>/SKILL.md


def _resolve_scientific_skills(override: str) -> Path:
    """Return the path to the scientific-skills directory."""
    if override:
        return Path(override).expanduser().resolve()
    env_val = os.environ.get("GLADIUS_SCIENTIFIC_SKILLS_PATH", "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    return _REPO / "claude-scientific-skills" / "scientific-skills"


# ── YAML config loading ───────────────────────────────────────────────────────

_VALID_PLATFORMS = {"kaggle", "zindi", "fake", "none"}
_VALID_TOPOLOGIES = {"functional", "two-pizza", "platform", "autonomous", "matrix"}
_VALID_DIRECTIONS = {"maximize", "minimize"}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse and validate the YAML project config. Returns a normalised dict."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")

    with p.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # ── required ──────────────────────────────────────────────────────────────
    for key in ("competition_id", "project_dir"):
        if not raw.get(key):
            raise ConfigError(f"Config missing required field: '{key}'")

    # ── platform ──────────────────────────────────────────────────────────────
    platform = str(raw.get("platform") or "none").strip().lower()
    if platform not in _VALID_PLATFORMS:
        raise ConfigError(
            f"platform must be one of {sorted(_VALID_PLATFORMS)}, got {platform!r}"
        )
    raw["platform"] = platform

    # ── metric / direction ────────────────────────────────────────────────────
    has_metric = bool(raw.get("metric"))
    has_direction = bool(raw.get("direction"))
    if has_metric != has_direction:
        raise ConfigError(
            "'metric' and 'direction' must both be provided or both omitted."
        )
    if has_direction:
        direction = str(raw["direction"]).strip().lower()
        if direction not in _VALID_DIRECTIONS:
            raise ConfigError(
                f"direction must be 'maximize' or 'minimize', got {direction!r}"
            )
        raw["direction"] = direction
    else:
        raw["metric"] = None
        raw["direction"] = None

    # ── topology ──────────────────────────────────────────────────────────────
    topology = str(raw.get("topology") or "functional").strip().lower()
    if topology not in _VALID_TOPOLOGIES:
        raise ConfigError(
            f"topology must be one of {sorted(_VALID_TOPOLOGIES)}, got {topology!r}"
        )
    raw["topology"] = topology

    # ── data_dir ──────────────────────────────────────────────────────────────
    project_dir = Path(raw["project_dir"]).expanduser().resolve()
    raw["project_dir"] = str(project_dir)
    if raw.get("data_dir"):
        data_dir = Path(raw["data_dir"]).expanduser()
        raw["data_dir"] = str(
            data_dir if data_dir.is_absolute() else (project_dir / data_dir).resolve()
        )
    else:
        raw["data_dir"] = str(project_dir / "data")

    # ── model ─────────────────────────────────────────────────────────────────
    raw.setdefault("model", os.environ.get("GLADIUS_MODEL", "claude-opus-4-5"))
    raw.setdefault("small_model", os.environ.get("GLADIUS_SMALL_MODEL", "inherit"))

    # ── skills / roles ────────────────────────────────────────────────────────
    raw.setdefault("gladius_skills", "all")
    raw.setdefault("scientific_skills", True)
    raw.setdefault("scientific_skills_path", "")
    raw.setdefault("custom_skills_dir", "")
    raw.setdefault("roles", "all")

    # ── mcp ───────────────────────────────────────────────────────────────────
    mcp_cfg = raw.setdefault("mcp", {})
    mcp_cfg.setdefault("platform_server", True)
    mcp_cfg.setdefault("extra", {})

    # ── misc ──────────────────────────────────────────────────────────────────
    raw.setdefault("force", False)

    return raw


# ── Per-step helpers ──────────────────────────────────────────────────────────


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_tree(src: Path, dst: Path, *, force: bool) -> bool:
    """Copy a directory tree. Returns True if actually copied, False if skipped."""
    if dst.exists() and not force:
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def _write_file(path: Path, content: str, *, force: bool) -> None:
    """Write a text file. Skips if already exists and not force."""
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Step 1 — directory skeleton ───────────────────────────────────────────────


def _setup_dir_skeleton(root: Path) -> None:
    """Create project directory and standard subdirectories."""
    for subdir in (
        root / ".claude" / "agents",
        root / ".claude" / "skills",
        root / ".claude" / "agent-memory" / "team-lead",
        root / "src",
        root / "scripts",
        root / "artifacts",
        root / "submissions",
    ):
        subdir.mkdir(parents=True, exist_ok=True)

    # .gitignore for data / artifacts / secrets
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "data/\nartifacts/\nsubmissions/\n*.db\n*.log\n.env\n",
            encoding="utf-8",
        )


# ── Step 2 — copy role agents ─────────────────────────────────────────────────


def _copy_roles(root: Path, roles_spec: str | list, model: str, small_model: str, *, force: bool) -> None:
    """Copy role .md files from src/roles/ into .claude/agents/.

    Substitutes {{GLADIUS_MODEL}} and {{GLADIUS_SMALL_MODEL}} at copy time.
    Falls back to gladius/utils/templates/agents/ for any role not found in src/roles/.
    """
    agents_dir = root / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Determine which role files to copy
    if roles_spec == "all":
        candidates = sorted(_SRC_ROLES.glob("*.md"))
    else:
        names = [roles_spec] if isinstance(roles_spec, str) else list(roles_spec)
        candidates = []
        for name in names:
            p = _SRC_ROLES / f"{name}.md"
            if not p.is_file():
                p = _TEMPLATES / "agents" / f"{name}.md"
            if p.is_file():
                candidates.append(p)
            else:
                print(f"  [warn] role not found: {name!r} — skipped", file=sys.stderr)

    for src in candidates:
        dest = agents_dir / src.name
        if dest.exists() and not force:
            continue
        content = (
            src.read_text(encoding="utf-8")
            .replace("{{GLADIUS_MODEL}}", model)
            .replace("{{GLADIUS_SMALL_MODEL}}", small_model)
        )
        dest.write_text(content, encoding="utf-8")
        print(f"  agent  → .claude/agents/{src.name}")


# ── Step 3 — copy skills ──────────────────────────────────────────────────────


def _copy_gladius_skills(root: Path, skills_spec: str | list, *, force: bool) -> None:
    """Copy gladius-provided skills into .claude/skills/.

    Sources (in priority order — first write wins for a given skill name):
      1. src/skills/                          (framework skills: validation, hpo, ensembling, …)
      2. gladius/utils/templates/skills/      (template skills: ml-setup, code-review, …)
    """
    skills_dir = root / ".claude" / "skills"
    tpl_skills = _TEMPLATES / "skills"

    # Collect candidates from both sources; src/skills/ takes precedence
    seen: dict[str, Path] = {}
    for src in sorted(tpl_skills.iterdir()):
        if src.is_dir():
            seen.setdefault(src.name, src)
    for src in sorted(_SRC_SKILLS.iterdir()):
        if src.is_dir() and not src.name.startswith("_"):
            seen[src.name] = src   # override template with framework version

    if skills_spec == "all":
        skill_dirs = list(seen.values())
    else:
        names = [skills_spec] if isinstance(skills_spec, str) else list(skills_spec)
        skill_dirs = []
        for name in names:
            if name in seen:
                skill_dirs.append(seen[name])
            else:
                print(f"  [warn] gladius skill not found: {name!r} — skipped", file=sys.stderr)

    for src in skill_dirs:
        dest = skills_dir / src.name
        if _copy_tree(src, dest, force=force):
            print(f"  skill  → .claude/skills/{src.name}/")


def _copy_custom_skills(root: Path, custom_dir: str, *, force: bool) -> None:
    """Copy user-provided skills from custom_skills_dir into .claude/skills/.

    Every immediate subdirectory that contains a SKILL.md is treated as a skill.
    Custom skills override any gladius skill of the same name.
    """
    if not custom_dir:
        return
    src_root = Path(custom_dir).expanduser().resolve()
    if not src_root.is_dir():
        print(f"  [warn] custom_skills_dir not found: {src_root} — skipped", file=sys.stderr)
        return

    skills_dir = root / ".claude" / "skills"
    for src in sorted(src_root.iterdir()):
        if src.is_dir() and (src / "SKILL.md").is_file():
            dest = skills_dir / src.name
            _copy_tree(src, dest, force=True)   # custom always wins
            print(f"  custom → .claude/skills/{src.name}/")


def _copy_scientific_skills(root: Path, sci_path: str, *, force: bool) -> None:
    """Copy the claude-scientific-skills catalog into .claude/skills/ (idempotent)."""
    catalog = _resolve_scientific_skills(sci_path)
    if not catalog.is_dir():
        print(
            f"  [warn] scientific-skills catalog not found at {catalog}\n"
            "         Run: git submodule update --init\n"
            "         Or set scientific_skills_path in your config.",
            file=sys.stderr,
        )
        return

    skills_dir = root / ".claude" / "skills"
    copied = 0
    for skill_dir in sorted(catalog.iterdir()):
        if not skill_dir.is_dir():
            continue
        dest = skills_dir / skill_dir.name
        if (dest / "SKILL.md").exists() and not force:
            continue
        _copy_tree(skill_dir, dest, force=force)
        copied += 1

    print(f"  scientific-skills → .claude/skills/  ({copied} skills)")


# ── Step 4 — hook scripts ─────────────────────────────────────────────────────


def _copy_hooks(root: Path, *, force: bool) -> None:
    """Copy hook scripts from templates and make them executable."""
    hooks_src = _TEMPLATES / "hooks"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    for src in sorted(hooks_src.glob("*.sh")):
        dest = scripts_dir / src.name
        if dest.exists() and not force:
            continue
        shutil.copy2(src, dest)
        _make_executable(dest)
        print(f"  hook   → scripts/{src.name}")


# ── Step 5 — .claude/settings.json ───────────────────────────────────────────


def _write_settings(root: Path, cfg: dict) -> None:
    """Always overwrite settings.json — model / env can change."""
    settings = {
        "model": cfg["model"],
        "env": {
            "COMPETITION_ID": cfg["competition_id"],
            "TARGET_METRIC": cfg["metric"] or "",
            "METRIC_DIRECTION": cfg["direction"] or "",
            "DATA_DIR": cfg["data_dir"],
            "TOPOLOGY": cfg["topology"],
        },
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Edit|Write",
                    "hooks": [{"type": "command", "command": "scripts/after_edit.sh"}],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "scripts/validate_bash.sh"}],
                }
            ],
        },
    }
    path = root / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print("  settings → .claude/settings.json")


# ── Step 6 — .mcp.json ────────────────────────────────────────────────────────


def _write_mcp_json(root: Path, cfg: dict) -> None:
    """Always overwrite .mcp.json — skill dir path and platform may change."""
    skills_dir = str(root / ".claude" / "skills")
    mcp: dict = {
        "mcpServers": {
            "skills-on-demand": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "skills_on_demand.server"],
                "env": {"SKILLS_DIR": skills_dir},
            }
        }
    }

    platform = cfg["platform"]
    if cfg["mcp"]["platform_server"] and platform not in ("none", ""):
        _PLATFORM_MODULE = {
            "kaggle": "src.tools.kaggle_tools",
            "zindi":  "src.tools.zindi_tools",
            "fake":   "src.tools.fake_platform_tools",
        }
        _PLATFORM_SERVER = {
            "kaggle": "kaggle_server",
            "zindi": "zindi_server",
            "fake": "fake_server",
        }
        mod = _PLATFORM_MODULE.get(platform)
        srv = _PLATFORM_SERVER.get(platform)
        if mod and srv:
            mcp["mcpServers"][f"{platform}-tools"] = {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    f"from {mod} import {srv}; import asyncio; asyncio.run({srv}.run())",
                ],
                "env": {},
            }

    # Merge any extra MCP servers from config
    extra = cfg["mcp"].get("extra") or {}
    for name, server_cfg in extra.items():
        mcp["mcpServers"][name] = server_cfg

    path = root / ".mcp.json"
    path.write_text(json.dumps(mcp, indent=2) + "\n", encoding="utf-8")
    print("  mcp    → .mcp.json")


# ── Step 7 — CLAUDE.md ────────────────────────────────────────────────────────


def _write_claude_md(root: Path, cfg: dict) -> None:
    """Write a placeholder CLAUDE.md. Gladius overwrites it with live state on first run."""
    path = root / "CLAUDE.md"
    path.write_text(
        f"# {cfg['competition_id']}\n\n"
        "> This file will be populated by gladius on the first run.\n\n"
        f"- Platform: `{cfg['platform']}`\n"
        f"- Data: `{cfg['data_dir']}`\n"
        f"- Topology: `{cfg['topology']}`\n",
        encoding="utf-8",
    )
    print("  claude → CLAUDE.md")



# ── Step 8 — team-lead memory seed ───────────────────────────────────────────


def _seed_memory(root: Path, cfg: dict, *, force: bool) -> None:
    """Seed team-lead MEMORY.md from the appropriate template (idempotent)."""
    mem_file = root / ".claude" / "agent-memory" / "team-lead" / "MEMORY.md"
    if mem_file.exists() and not force:
        return

    is_ml = bool(cfg["metric"])
    template_name = "MEMORY-ml.md" if is_ml else "MEMORY-task.md"
    template = _TEMPLATES / "memory" / template_name
    if not template.exists():
        return

    mem_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, mem_file)
    print(f"  memory → .claude/agent-memory/team-lead/MEMORY.md")


# ── Main orchestration ────────────────────────────────────────────────────────


def setup(config_path: str | Path, *, force: bool = False) -> Path:
    """
    Run the full project setup from a YAML config file.

    Returns the resolved project_dir Path.
    """
    cfg = load_config(config_path)
    if force:
        cfg["force"] = True

    root = Path(cfg["project_dir"])
    f = cfg["force"]

    print(f"\nSetting up: {root}")
    print(f"  competition: {cfg['competition_id']}")
    print(f"  topology:    {cfg['topology']}")
    print(f"  platform:    {cfg['platform']}")
    if cfg["metric"]:
        print(f"  metric:      {cfg['metric']} ({cfg['direction']})")
    print()

    _setup_dir_skeleton(root)
    _copy_roles(root, cfg["roles"], cfg["model"], cfg["small_model"], force=f)

    # Skills: gladius-provided first, then scientific catalog, then user custom (always wins)
    _copy_gladius_skills(root, cfg["gladius_skills"], force=f)
    if cfg["scientific_skills"]:
        _copy_scientific_skills(root, cfg["scientific_skills_path"], force=f)
    _copy_custom_skills(root, cfg["custom_skills_dir"], force=f)

    _copy_hooks(root, force=f)
    _write_settings(root, cfg)
    _write_mcp_json(root, cfg)
    _write_claude_md(root, cfg)
    _seed_memory(root, cfg, force=f)

    print(f"\nDone. Project ready at: {root}\n")
    return root


# ── 'init' subcommand — scaffold a starter config ─────────────────────────────


_STARTER_CONFIG = """\
# Gladius project configuration
# Run: python -m src.project_setup --config project.yaml

# ── Required ─────────────────────────────────────────────────────────────────
competition_id: my-competition          # slug for logging / platform API
project_dir: /path/to/project           # directory to set up (created if absent)

# ── Competition type ──────────────────────────────────────────────────────────
platform: none                          # kaggle | zindi | fake | none
metric: auc_roc                         # omit entirely for open-ended tasks
direction: maximize                     # maximize | minimize

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir: data                          # relative to project_dir, or absolute

# ── Agent topology ────────────────────────────────────────────────────────────
topology: functional                    # functional | two-pizza | platform | autonomous | matrix

# ── Model ─────────────────────────────────────────────────────────────────────
model: claude-opus-4-5                  # or set env GLADIUS_MODEL
small_model: inherit                    # or set env GLADIUS_SMALL_MODEL

# ── Skills ────────────────────────────────────────────────────────────────────
gladius_skills: all                     # "all" or list: skills that ship with gladius
scientific_skills: true                 # copy 170+ upstream scientific skills
scientific_skills_path: ""              # override; defaults to env GLADIUS_SCIENTIFIC_SKILLS_PATH
custom_skills_dir: ""                   # YOUR folder of extra skills (each subdir with SKILL.md is copied)

# ── Roles ─────────────────────────────────────────────────────────────────────
roles: all                              # "all" or list: [team-lead, ml-engineer, ...]

# ── MCP servers ───────────────────────────────────────────────────────────────
mcp:
  platform_server: true                 # register platform MCP server (if platform != none)
  extra: {}                             # additional mcpServers entries to merge
  # example extra server:
  # extra:
  #   my-server:
  #     type: stdio
  #     command: python
  #     args: [-m, my_mcp_server]
  #     env: {}

# ── Misc ──────────────────────────────────────────────────────────────────────
force: false                            # true = overwrite existing files
"""


def _cmd_init(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"File already exists: {out}  (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    out.write_text(_STARTER_CONFIG, encoding="utf-8")
    print(f"Starter config written to: {out}")


def _cmd_setup(args: argparse.Namespace) -> None:
    try:
        setup(args.config, force=args.force)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.project_setup",
        description="Bootstrap a Gladius competition project from a YAML config.",
    )
    sub = parser.add_subparsers(dest="command")

    # Default action: setup (no subcommand needed for backwards compat)
    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        help="Path to project YAML config file.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        default=False,
        help="Overwrite existing files (agents, skills, hooks).",
    )

    # 'init' subcommand
    init_parser = sub.add_parser("init", help="Scaffold a starter project.yaml config.")
    init_parser.add_argument(
        "--out", "-o",
        default="project.yaml",
        metavar="FILE",
        help="Output path for the starter config (default: project.yaml).",
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        default=False,
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        _cmd_init(args)
    elif args.config:
        _cmd_setup(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
