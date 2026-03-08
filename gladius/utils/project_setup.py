"""
Project directory setup utilities.

Called by the orchestrator to:
1. Bootstrap a competition project directory with Claude Code native config
   (.claude/agents/, .claude/skills/, .claude/settings.json, hook scripts).
2. Write/refresh CLAUDE.md — the shared context document loaded automatically
   by all agents (Claude Code loads it from the working directory at session
   start; agents also receive it as part of their context).

All writes are idempotent for bootstrap files (skip if already exist), but
CLAUDE.md is always overwritten because it carries live competition state.

Templates live in gladius/utils/templates/ — plain text files, no Python
strings. Modify them there; this module is logic only.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gladius.state import CompetitionState

_TEMPLATES = Path(__file__).parent / "templates"


# ── CLAUDE.md ─────────────────────────────────────────────────────────────────


def write_claude_md(state: "CompetitionState", project_dir: str) -> None:
    """
    Overwrite CLAUDE.md in project_dir with the current competition state.

    This is called at the start of every iteration so every agent sees
    fresh context without needing it injected into every prompt.
    """
    p = Path(project_dir) / "CLAUDE.md"

    best_oof = (
        f"{state.best_oof_score:.6f}"
        if state.best_oof_score is not None
        else "none yet"
    )
    best_lb = (
        f"{state.best_submission_score:.6f}"
        if state.best_submission_score is not None
        else "none yet"
    )
    direction_str = (
        "↑ higher is better"
        if state.metric_direction == "maximize"
        else ("↓ lower is better" if state.metric_direction == "minimize" else "n/a")
    )

    # Top-5 experiments, newest first
    exps = list(reversed(state.experiments[-5:]))
    exp_lines = "_(none yet)_"
    if exps:
        rows = []
        for e in exps:
            if state.target_metric:
                s = e.get("oof_score", "?")
                score_col = str(s)
            else:
                q = e.get("quality_score")
                score_col = f"{q}/100" if q is not None else "?"
            n = e.get("notes", "")[:100]
            it = e.get("iteration", "?")
            files = ", ".join(Path(f).name for f in e.get("solution_files", []))
            rows.append(f"| iter {it} | {score_col} | {files} | {n} |")
        score_header = "OOF Score" if state.target_metric else "Quality"
        exp_lines = (
            f"| Iteration | {score_header} | Files | Notes |\n| --- | --- | --- | --- |\n"
        ) + "\n".join(rows)

    # Failed approaches to avoid
    failed_lines = "_(none)_"
    if state.failed_runs:
        failed_lines = "\n".join(
            f"- iter {f.get('iteration', '?')}: {f.get('error', '?')[:80]}"
            for f in state.failed_runs[-5:]
        )

    # Stagnation detection — warn planner if last 3 experiments barely moved
    stagnation_block = ""
    if state.target_metric:
        scored = [e for e in state.experiments if e.get("oof_score") is not None]
        score_key = "oof_score"
        threshold_label = state.target_metric
        stagnation_threshold = 0.001
    else:
        scored = [e for e in state.experiments if e.get("quality_score") is not None]
        score_key = "quality_score"
        threshold_label = "quality"
        stagnation_threshold = 3.0
    if len(scored) >= 3:
        last3 = [e[score_key] for e in scored[-3:]]
        span = max(last3) - min(last3)
        if span < stagnation_threshold:
            stagnation_block = f"""
## ⚠️ STAGNATION WARNING

> The last **{len(last3)} experiments** moved the {threshold_label} score by only
> **{span:.4f}** (threshold: {stagnation_threshold}). Incremental tweaks are not working.
>
> **Planner: stop tuning. Go back to first principles.**
> - Re-examine the task description and deliverables.
> - Try a completely different approach or architecture.
> - WebSearch for breakthrough techniques specific to this task type.
"""

    # Performance section varies by task type
    if state.target_metric:
        perf_section = f"""\
## Current Best

| Metric | Score |
| --- | --- |
| Best OOF ({state.target_metric}) | **{best_oof}** |
| Best leaderboard | **{best_lb}** |
| Submissions today | {state.submission_count} / {state.max_submissions_per_day} |
"""
        metric_row = f"| Target metric | `{state.target_metric}` ({direction_str}) |"
        data_section = f"""\
## Data Files

```bash
ls {state.data_dir}
```

Standard files to expect:
- `train.csv` — training set (target column present)
- `test.csv` — test set (no target column)
- `sample_submission.csv` — submission format template
"""
        submission_section = """\
## Submission Rules

1. Load `sample_submission.csv` to get the exact submission format.
2. Your submission must match its columns and row count exactly.
3. Save as a CSV in the project directory.
4. Report the path in `submission_file` in your output.
"""
        skills_section = """\
## Available Skills

Invoke skills with the `Skill` tool — the output is returned inline, do NOT use TaskOutput.

| Skill name | When to invoke |
| --- | --- |
| `code-review` | **Required before reporting results** — catches leakage & metric bugs |
| `ml-pipeline` | Writing CV / feature / submission code — patterns, baselines, metric formulas |
| `ml-project-structure` | **Invoke first** — set up `src/` package layout before writing any solution code |
| `adversarial-validation` | Detect train/test distribution shift; find leaking features |
| `feature-engineering` | Systematic feature recipes, SHAP importance, feature pruning |
| `hpo` | Bayesian hyperparameter search with Optuna (after baseline is solid) |
| `ensembling` | OOF blending, rank averaging, stacking, hill-climbing selection |
| `research` | Find SOTA techniques on ArXiv + Kaggle forums for this task type |
| `polars`            | Fast DataFrame ops for large datasets (Arrow backend, lazy eval) |
| `transformers`      | HuggingFace Transformers for NLP/vision competitions |
| `pytorch-lightning` | Structured DL training loops, multi-GPU, callbacks |
| `timesfm`           | Google TimesFM zero-shot time-series forecasting |
| `submit-check` | Validate submission CSV format before uploading |
| `jupyter-mcp` | When you want to work in a Jupyter notebook — starts Jupyter + MCP server |
| `git-workflow` | After each working solution |
| `uv-venv` | Installing packages and running scripts |
"""
    else:
        best_quality = (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
        perf_section = f"""\
## Current Progress

| Metric | Value |
| --- | --- |
| Best quality score (0-100) | **{best_quality}** |
| Deliverables submitted | {state.submission_count} |
"""
        metric_row = "| Task type | open-ended (self-assessed quality 0-100) |"
        data_section = ""
        submission_section = """\
## Deliverable Rules

1. Read README.md thoroughly to understand what deliverable is required.
2. Build and verify the deliverable works end-to-end.
3. Package it as described in README.md (zip, binary, URL file, etc.).
4. Report the path in `submission_file` in your output.
5. Self-assess quality 0-100: rate completeness and correctness against README requirements.
"""
        skills_section = """\
## Available Skills

Invoke skills with the `Skill` tool — the output is returned inline, do NOT use TaskOutput.

| Skill name | When to invoke |
| --- | --- |
| `code-review` | **Required before reporting results** — catches crashes & missing features |
| `task-review` | **Required before reporting results** — quality self-assessment checklist |
| `git-workflow` | After each working iteration |
| `uv-venv` | Installing packages and running scripts |
"""

    content = f"""\
# Competition/Task: {state.competition_id}

> **This file is auto-generated by Gladius and refreshed every iteration.**
> Read it first before doing anything else.

## Settings

| Field | Value |
| --- | --- |
| ID | `{state.competition_id}` |
{metric_row}
| Data directory | `{state.data_dir}` |
| Output directory | `{state.output_dir}` |
| Iteration | {state.iteration} / {state.max_iterations} |
| Phase | {state.phase} |

{perf_section}
## Recent Experiments

{exp_lines}

## Failed Approaches (avoid repeating)

{failed_lines}
{stagnation_block}
{data_section}
{submission_section}
{skills_section}
## Agent Memory

If you are the **planner**, read your agent memory at
`{Path(project_dir).resolve()}/.claude/agent-memory/planner/MEMORY.md` before exploring.
Update it with insights after each exploration session.

If you are an **implementer**, focus only on executing the given plan.

## Package Management

> **This project uses `uv` — there is no `pip` in the venv.**
> **Always install packages with:** `uv add <package>`
> Never use `pip install`, `pip3 install`, or `python -m pip install`.

```bash
uv add optuna catboost lightgbm   # install packages
uv run python solution.py         # run a script inside the venv
```
"""
    p.write_text(content, encoding="utf-8")


# ── Bootstrap project .claude/ structure ──────────────────────────────────────


def setup_project_dir(
    state: "CompetitionState", project_dir: str, platform: str = "none"
) -> None:
    """
    One-time (idempotent) setup of Claude Code native config in project_dir.

    Creates:
    - .claude/agents/planner.md
    - .claude/agents/implementer.md
    - .claude/skills/<name>/SKILL.md  (one per skill)
    - .claude/settings.json           (hooks + env)
    - .mcp.json                       (MCP server registrations for CLI use)
    - scripts/after_edit.sh           (PostToolUse hook: compile Python on edit)
    - scripts/validate_bash.sh        (PreToolUse hook: block dangerous rm -rf)
    """
    root = Path(project_dir)
    is_ml = bool(state.target_metric)

    _write_agent(root, "planner")
    _write_agent(root, "implementer")

    if is_ml:
        _copy_skill(root, "ml-pipeline")
        _copy_skill(root, "ml-project-structure")
        _copy_skill(root, "submit-check")
        _copy_skill(root, "jupyter-mcp")
        _copy_skill(root, "adversarial-validation")
        _copy_skill(root, "feature-engineering")
        _copy_skill(root, "hpo")
        _copy_skill(root, "ensembling")
        _copy_skill(root, "research")
        _copy_skill(root, "polars")
        _copy_skill(root, "transformers")
        _copy_skill(root, "pytorch-lightning")
        _copy_skill(root, "timesfm")
        _copy_skill(root, "code-review-ml", dest_name="code-review")
        _copy_skill(root, "git-workflow-ml", dest_name="git-workflow")
    else:
        _copy_skill(root, "task-review")
        _copy_skill(root, "code-review-task", dest_name="code-review")
        _copy_skill(root, "git-workflow-task", dest_name="git-workflow")

    _copy_skill(root, "uv-venv")
    _write_claude_settings(root, state)
    _write_mcp_json(root, platform=platform)
    _copy_hook(root, "after_edit.sh")
    _copy_hook(root, "validate_bash.sh")
    _make_memory_dir(root, is_ml)


# ── Template helpers ──────────────────────────────────────────────────────────


def _write_agent(root: Path, name: str) -> None:
    """Always overwrite agent definitions (model env var may change)."""
    path = root / ".claude" / "agents" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    template = (_TEMPLATES / "agents" / f"{name}.md").read_text(encoding="utf-8")
    content = template.replace(
        "{{GLADIUS_MODEL}}",
        os.environ.get("GLADIUS_MODEL", "GLADIUS_MODEL_NOT_SET"),
    )
    path.write_text(content, encoding="utf-8")


def _copy_skill(root: Path, template_name: str, *, dest_name: str | None = None) -> None:
    """Copy a skill template into the competition's .claude/skills/ tree (idempotent)."""
    skill_name = dest_name or template_name
    path = root / ".claude" / "skills" / skill_name / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (_TEMPLATES / "skills" / f"{template_name}.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _copy_hook(root: Path, filename: str) -> None:
    """Copy a hook script into scripts/ and make it executable (idempotent)."""
    path = root / "scripts" / filename
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (_TEMPLATES / "hooks" / filename).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _make_executable(path)


# ── .claude/settings.json ─────────────────────────────────────────────────────


def _write_claude_settings(root: Path, state: "CompetitionState") -> None:
    path = root / ".claude" / "settings.json"
    # Always overwrite to keep env vars (competition ID etc.) up to date.
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "model": os.environ.get("GLADIUS_MODEL") or "GLADIUS_MODEL_NOT_SET",
        "env": {
            "COMPETITION_ID": state.competition_id,
            "TARGET_METRIC": state.target_metric or "",
            "METRIC_DIRECTION": state.metric_direction or "",
            "DATA_DIR": state.data_dir,
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
                    "hooks": [
                        {"type": "command", "command": "scripts/validate_bash.sh"}
                    ],
                }
            ],
        },
    }
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


# ── .mcp.json (for Claude Code CLI use) ──────────────────────────────────────


def _write_mcp_json(root: Path, platform: str = "none") -> None:
    path = root / ".mcp.json"
    import sys

    mcp_config: dict = {"mcpServers": {}}

    if platform not in ("none", ""):
        server_module = {
            "kaggle": "gladius.tools.kaggle_tools",
            "zindi": "gladius.tools.zindi_tools",
            "fake": "gladius.tools.fake_platform_tools",
        }.get(platform)
        server_name = {
            "kaggle": "kaggle_server",
            "zindi": "zindi_server",
            "fake": "fake_server",
        }.get(platform)
        if server_module and server_name:
            mcp_config["mcpServers"][f"{platform}-tools"] = {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    (
                        f"from {server_module} import {server_name}; "
                        f"import asyncio; asyncio.run({server_name}.run())"
                    ),
                ],
                "env": {},
            }

    path.write_text(json.dumps(mcp_config, indent=2) + "\n", encoding="utf-8")


# ── Agent memory directory ────────────────────────────────────────────────────


def _make_memory_dir(root: Path, is_ml: bool) -> None:
    """Pre-create the planner's memory directory so it exists on first run."""
    mem_dir = root / ".claude" / "agent-memory" / "planner"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_file = mem_dir / "MEMORY.md"
    if not mem_file.exists():
        template_name = "MEMORY-ml.md" if is_ml else "MEMORY-task.md"
        mem_file.write_text(
            (_TEMPLATES / "memory" / template_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


# ── Utility ───────────────────────────────────────────────────────────────────


def _make_executable(path: Path) -> None:
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
