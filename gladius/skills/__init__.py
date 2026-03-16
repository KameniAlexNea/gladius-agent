from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_SRC_SKILLS = Path(__file__).parent


def _copy_tree(src: Path, dst: Path, *, force: bool) -> bool:
    if dst.exists() and not force:
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def copy(dst: Path, spec: str | list, *, force: bool = False) -> None:
    """Copy skills from src/skills/ into dst."""
    skill_dirs = [
        p
        for p in sorted(_SRC_SKILLS.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]

    if spec != "all":
        names = [spec] if isinstance(spec, str) else list(spec)
        by_name = {p.name: p for p in skill_dirs}
        skill_dirs = []
        for n in names:
            if n in by_name:
                skill_dirs.append(by_name[n])
            else:
                print(f"  [warn] skill not found: {n!r} — skipped", file=sys.stderr)

    for src in skill_dirs:
        if _copy_tree(src, dst / src.name, force=force):
            print(f"  skill  → .claude/skills/{src.name}/")


def copy_scientific(dst: Path, override_path: str = "", *, force: bool = False) -> None:
    """Copy the claude-scientific-skills catalog into dst."""
    if override_path:
        catalog = Path(override_path).expanduser().resolve()
    else:
        env_val = os.environ.get("GLADIUS_SCIENTIFIC_SKILLS_PATH", "").strip()
        catalog = (
            Path(env_val).expanduser().resolve()
            if env_val
            else (
                Path(__file__).parent.parent.parent
                / "claude-scientific-skills"
                / "scientific-skills"
            )
        )

    if not catalog.is_dir():
        print(
            f"  [warn] scientific-skills not found at {catalog}\n"
            "         Run: git submodule update --init",
            file=sys.stderr,
        )
        return

    copied = 0
    for skill_dir in sorted(catalog.iterdir()):
        if not skill_dir.is_dir():
            continue
        dest = dst / skill_dir.name
        if (dest / "SKILL.md").exists() and not force:
            continue
        _copy_tree(skill_dir, dest, force=force)
        copied += 1
    print(f"  scientific-skills → .claude/skills/  ({copied} skills)")


def copy_custom(dst: Path, custom_dir: str) -> None:
    """Copy user-provided skills from custom_dir into dst (always overrides)."""
    if not custom_dir:
        return
    src_root = Path(custom_dir).expanduser().resolve()
    if not src_root.is_dir():
        print(
            f"  [warn] custom_skills_dir not found: {src_root} — skipped",
            file=sys.stderr,
        )
        return
    for src in sorted(src_root.iterdir()):
        if src.is_dir() and (src / "SKILL.md").is_file():
            _copy_tree(src, dst / src.name, force=True)
            print(f"  custom → .claude/skills/{src.name}/")
