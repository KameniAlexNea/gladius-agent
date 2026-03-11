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
import shutil
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from gladius.state import CompetitionState

_TEMPLATES = Path(__file__).parent / "templates"


# Path to the claude-scientific-skills directory (173+ upstream skills).
# Resolution order:
#   1. GLADIUS_SCIENTIFIC_SKILLS_PATH env var (absolute or relative path)
#   2. Default: claude-scientific-skills/scientific-skills/ submodule next to this repo
# Set GLADIUS_SCIENTIFIC_SKILLS_PATH in your .env to override.
def _resolve_scientific_skills_path() -> Path:
    env_val = os.environ.get("GLADIUS_SCIENTIFIC_SKILLS_PATH", "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    return (
        Path(__file__).parent.parent.parent
        / "claude-scientific-skills"
        / "scientific-skills"
    )


_SCIENTIFIC_SKILLS = _resolve_scientific_skills_path()


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
        threshold_str = (
            f"{state.submission_threshold:.6f}"
            if getattr(state, "submission_threshold", None) is not None
            else "not set (use WebSearch to check leaderboard before first submit)"
        )
        perf_section = f"""\
## Current Best

| Metric | Score |
| --- | --- |
| Best OOF ({state.target_metric}) | **{best_oof}** |
| Best leaderboard | **{best_lb}** |
| Submissions today | {state.submission_count} / {state.max_submissions_per_day} |
| **Minimum submission threshold** | **{threshold_str}** |

> **Do not build a submission unless your OOF score beats the minimum threshold.**
> If the threshold is "not set", `WebSearch` the competition leaderboard to find
> the current top/median score and set that as your personal bar.
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

1. **Gate:** Only build a submission once your OOF score beats the `Minimum submission threshold` shown above.
   - If threshold is "not set", `WebSearch` the leaderboard first and use the current median score as your bar.
2. Load `sample_submission.csv` to get the exact submission format.
3. Your submission must match its columns and row count exactly.
4. Save as a CSV in the project directory.
5. Report the path in `submission_file` in your output.
"""
        skills_section = """\
## Available Skills

> **No task starts without loading a skill. This is a hard requirement.**

### Search then load

```
mcp__skills-on-demand__search_skills({"query": "<task description>", "top_k": 5})
Skill({"skill": "<skill-name>"})
```

### Key ML skills from the catalog

| Domain | Skills |
| --- | --- |
| **Tabular ML** | `scikit-learn`, `polars`, `shap`, `statsmodels`, `statistical-analysis` |
| **Deep learning** | `pytorch-lightning`, `transformers`, `torch-geometric`, `stable-baselines3` |
| **Forecasting** | `timesfm-forecasting`, `aeon` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly`, `scientific-visualization` |
| **Data** | `exploratory-data-analysis`, `dask`, `lamindb`, `zarr-python` |
| **Life sciences** | `biopython`, `scanpy`, `anndata`, `scvi-tools`, `deeptools`, `gget` |
| **Chem / Drug** | `rdkit`, `deepchem`, `datamol`, `molfeat`, `matchms` |
| **Structural bio** | `alphafold-database`, `esm`, `pdb-database`, `uniprot-database` |
| **Genomics** | `pydeseq2`, `scikit-bio`, `scvelo`, `pathml`, `histolab` |
| **Databases** | `pubmed-database`, `biorxiv-database`, `openalex-database`, `chembl-database`, `ensembl-database` |
| **Research** | `perplexity-search`, `hypothesis-generation`, `literature-review`, `scientific-brainstorming` |
| **Clinical** | `clinical-decision-support`, `clinicaltrials-database`, `clinvar-database` |
| **Finance** | `alpha-vantage`, `fred-economic-data`, `edgartools` |
| **Compute** | `modal`, `networkx`, `sympy`, `geopandas` |
| **Writing** | `scientific-writing`, `latex-posters`, `infographics` |

> Always search — do not guess. The catalog grows continuously.
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

> **No task starts without loading a skill. This is a hard requirement.**

### Search then load

```
mcp__skills-on-demand__search_skills({"query": "<task description>", "top_k": 5})
Skill({"skill": "<skill-name>"})
```

### Key skills from the catalog

| Domain | Skills |
| --- | --- |
| **Tabular / Analysis** | `scikit-learn`, `polars`, `shap`, `statsmodels`, `statistical-analysis` |
| **Deep learning** | `pytorch-lightning`, `transformers`, `torch-geometric`, `stable-baselines3` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly`, `scientific-visualization` |
| **Data** | `exploratory-data-analysis`, `dask`, `lamindb`, `zarr-python` |
| **Life sciences** | `biopython`, `scanpy`, `anndata`, `scvi-tools`, `deeptools`, `gget` |
| **Chem / Drug** | `rdkit`, `deepchem`, `datamol`, `molfeat`, `matchms` |
| **Structural bio** | `alphafold-database`, `esm`, `pdb-database`, `uniprot-database` |
| **Databases** | `pubmed-database`, `biorxiv-database`, `openalex-database`, `chembl-database` |
| **Research** | `perplexity-search`, `hypothesis-generation`, `literature-review`, `scientific-brainstorming` |
| **Clinical** | `clinical-decision-support`, `clinicaltrials-database`, `clinvar-database` |
| **Finance** | `alpha-vantage`, `fred-economic-data`, `edgartools` |
| **Writing** | `scientific-writing`, `latex-posters`, `infographics`, `docx` |

> Always search — do not guess. The catalog grows continuously.
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

Your memory lives at:
`{Path(project_dir).resolve()}/.claude/agent-memory/MEMORY.md`

Read it at the start of every session. Write key findings, scores, and next steps
at the end of each session. This is how you remember across runs.

## Project Structure

Organise all code under `src/` as importable modules:

```
src/
├── data.py        # data loading, cleaning, feature engineering
├── eda.py         # exploratory analysis and visualisations
├── train.py       # model training + cross-validation (entry point)
├── evaluate.py    # metrics, OOF scoring, error analysis
└── submission.py  # build and validate the submission CSV
```

- One responsibility per module — no monolithic scripts.
- `train.py` is the main entry point: it imports from `data`, calls CV, scores, and delegates submission building to `submission.py`.
- Never import between modules in a cycle; keep `data` as a leaf dependency.

## Package Management

If `uv` has not been initialised yet in this directory, run the full setup first:

```bash
uv init          # creates pyproject.toml (skip if already exists)
uv venv          # creates .venv/
source .venv/bin/activate   # activate the environment
```

Then install packages and run code:

```bash
uv add lightgbm catboost optuna   # add dependencies (writes to pyproject.toml)
uv run python src/train.py        # run inside the venv
```

> **Never use `pip install`, `pip3 install`, or `python -m pip install`.**
> Always use `uv add` to install and `uv run python` to execute.

## Long-Running Scripts

For training scripts that take more than a few seconds, use `nohup` and track the PID directly.
**Never use background task IDs (`TaskOutput`, `TaskStop`).**

```bash
# Launch and capture PID
nohup uv run python src/train.py > train.log 2>&1 & echo $!

# Check if still running
ps -p <PID> -o pid,stat,etime,cmd --no-headers 2>/dev/null || echo "done"

# Read progress
tail -n 50 train.log

# Wait for completion
while kill -0 <PID> 2>/dev/null; do sleep 30; done && echo "finished"
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
    - .claude/skills/<name>/SKILL.md  (one per skill)
    - .claude/settings.json           (hooks + env)
    - .mcp.json                       (MCP server registrations for CLI use)
    - scripts/after_edit.sh           (PostToolUse hook: compile Python on edit)
    - scripts/validate_bash.sh        (PreToolUse hook: block dangerous rm -rf)
    """
    root = Path(project_dir)
    is_ml = bool(state.target_metric)

    _copy_all_scientific_skills(root)
    _write_claude_settings(root, state)
    _write_mcp_json(root, platform=platform)
    _copy_hook(root, "after_edit.sh")
    _copy_hook(root, "validate_bash.sh")
    _make_memory_dir(root, is_ml)


# ── Template helpers ──────────────────────────────────────────────────────────


def _copy_all_scientific_skills(root: Path) -> None:
    """Copy every skill from the claude-scientific-skills submodule (idempotent).

    Emits a warning and skips if the submodule hasn't been initialised or if
    GLADIUS_SCIENTIFIC_SKILLS_PATH points to a non-existent directory.
    """
    if not _SCIENTIFIC_SKILLS.is_dir():
        logger.warning(
            "claude-scientific-skills not found at '%s' — 170+ scientific skills will "
            "be skipped.\n"
            "To fix, run one of:\n"
            "  git submodule update --init          # if you cloned this repo\n"
            "  git clone https://github.com/K-Dense-AI/claude-scientific-skills.git\n"
            "Or set GLADIUS_SCIENTIFIC_SKILLS_PATH=/path/to/scientific-skills in your "
            ".env file.",
            _SCIENTIFIC_SKILLS,
        )
        return
    skills_dest = root / ".claude" / "skills"
    skills_dest.mkdir(parents=True, exist_ok=True)
    for skill_dir in sorted(_SCIENTIFIC_SKILLS.iterdir()):
        if not skill_dir.is_dir():
            continue
        dest = skills_dest / skill_dir.name
        if (dest / "SKILL.md").exists():
            continue
        shutil.copytree(skill_dir, dest, dirs_exist_ok=True)


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

    skills_dir = str(root / ".claude" / "skills")
    mcp_config: dict = {
        "mcpServers": {
            "skills-on-demand": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "skills_on_demand.server"],
                "env": {"SKILLS_DIR": skills_dir},
            }
        }
    }

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
    """Pre-create Gladius memory directory so it exists on first run."""
    mem_dir = root / ".claude" / "agent-memory"
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
