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

    # Stagnation detection — warn team lead if last 3 experiments barely moved
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
> **Team lead: stop tuning. Go back to first principles.**
> - Re-examine the task description and deliverables.
> - Try a completely different approach or architecture.
> - WebSearch for breakthrough techniques specific to this task type.
"""

    # Performance section varies by task type
    if state.target_metric:
        threshold_val = state.submission_threshold
        threshold_str = f"{threshold_val:.6f}" if threshold_val is not None else "not set"
        threshold_note = (
            f"> ⛔ **Do not build a submission unless your OOF score beats {threshold_str}.**"
            if threshold_val is not None
            else "> ⚠️ Threshold not set — `WebSearch` the leaderboard and use the current median score as your bar."
        )
        perf_section = f"""\
## Current Best

| Metric | Score |
| --- | --- |
| Best OOF ({state.target_metric}) | **{best_oof}** |
| Best leaderboard | **{best_lb}** |
| Submissions today | {state.submission_count} / {state.max_submissions_per_day} |
| **Minimum submission threshold** | **{threshold_str}** |

{threshold_note}
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
        submission_section = f"""\
## Submission Rules

1. **Gate:** Only build a submission once your OOF score beats the `Minimum submission threshold` shown above.
   - If threshold is "not set", `WebSearch` the leaderboard first and use the current median score as your bar.
2. Load `sample_submission.csv` to get the exact submission format.
3. Your submission must match its columns and row count exactly.
4. Save to `submissions/submission.csv`.
5. Report the path in `submission_file` in your output.
"""
        skills_section = """\
## Available Skills

> **No task starts without loading a skill. This is a hard requirement.**
> Search → load → execute is the mandatory workflow for every agent action.

### Discover skills on demand

```
mcp__skills-on-demand__search_skills({"query": "...", "top_k": 5})
mcp__skills-on-demand__list_skills({})
```

### Workflow

1. `mcp__skills-on-demand__search_skills({"query": "<task description>", "top_k": 5})`
2. Read the best match: `.claude/skills/<name>/SKILL.md`
3. Follow the skill's patterns and execute

### Core ML skills

| Skill name | When to invoke |
| --- | --- |
| `ml-setup` | **Invoke first** — project layout, CV patterns, baselines, metric formulas |
| `code-review` | **Required before reporting results** — leakage, metric errors, submission bugs |
| `adversarial-validation` | Train/test distribution shift; leaking features |
| `feature-engineering` | Systematic feature recipes, SHAP importance, feature pruning |
| `hpo` | Bayesian hyperparameter search with Optuna (after baseline is solid) |
| `ensembling` | OOF blending, rank averaging, stacking, hill-climbing |
| `submit-check` | Validate submission CSV format |
| `git-workflow` | After each working solution |
| `uv-venv` | Installing packages and running scripts |

### Scientific skills catalog (170+ skills across 17+ domains)

Search these domains by keyword to find the right skill:

| Domain | Example skills |
| --- | --- |
| **ML / AI** | polars, lightgbm, xgboost, transformers, pytorch-lightning, timm, shap, scikit-learn, optuna, deepchem |
| **Bioinformatics** | biopython, bioservices, ensembl, ncbi-entrez, gget, deeptools, scanpy, anndata, geniml |
| **Cheminformatics** | rdkit, datamol, chembl, binding-db, hmdb, drugbank, brenda, glycoengineering |
| **Proteomics / Structural** | alphafold, esm, diffdock, protein-engineering |
| **Multi-omics** | arboreto, cobrapy, cellxgene-census, gtex, geo-database |
| **Clinical / Healthcare AI** | clinical-decision-support, clinical-reports, clinicaltrials, clinvar, cbioportal, fda-database |
| **Medical imaging** | histolab, imaging-data-commons |
| **Materials science** | materials-project, ase |
| **Physics / Astronomy** | astropy, fluidsim |
| **Data analysis** | dask, exploratory-data-analysis, geopandas, infographics |
| **Research** | biorxiv-database, perplexity-search, hypothesis-generation, hypogenic, citation-management |
| **Lab automation** | benchling-integration, ginkgo-cloud-lab, adaptyv |
| **Financial** | alpha-vantage, fred-economic-data, edgartools, hedgefundmonitor |
| **Geospatial** | geopandas, geomaster |
| **Storage / Compute** | dnanexus-integration, parallel-web, datacommons-client |
| **Quantum** | cirq |
| **Scientific communication** | docx, infographics |

> 250+ databases, 60+ optimized package skills, 15+ integration skills.
> Always search — do not guess. The library grows continuously.
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
> Search → load → execute is the mandatory workflow for every agent action.

### Discover skills on demand

```
mcp__skills-on-demand__search_skills({"query": "...", "top_k": 5})
mcp__skills-on-demand__list_skills({})
```

### Workflow

1. `mcp__skills-on-demand__search_skills({"query": "<task description>", "top_k": 5})`
2. Read the best match: `.claude/skills/<name>/SKILL.md`
3. Follow the skill's patterns and execute

### Core task skills

| Skill name | When to invoke |
| --- | --- |
| `code-review` | **Required before reporting results** — correctness, completeness, quality 0–100 |
| `git-workflow` | After each working iteration |
| `uv-venv` | Installing packages and running scripts |

### Scientific skills catalog (170+ skills across 17+ domains)

Search these domains by keyword to find the right skill:

| Domain | Example skills |
| --- | --- |
| **ML / AI** | polars, lightgbm, xgboost, transformers, pytorch-lightning, timm, shap, scikit-learn, optuna, deepchem |
| **Bioinformatics** | biopython, bioservices, ensembl, ncbi-entrez, gget, deeptools, scanpy, anndata, geniml |
| **Cheminformatics** | rdkit, datamol, chembl, binding-db, hmdb, drugbank, brenda, glycoengineering |
| **Proteomics / Structural** | alphafold, esm, diffdock, protein-engineering |
| **Multi-omics** | arboreto, cobrapy, cellxgene-census, gtex, geo-database |
| **Clinical / Healthcare AI** | clinical-decision-support, clinical-reports, clinicaltrials, clinvar, cbioportal, fda-database |
| **Medical imaging** | histolab, imaging-data-commons |
| **Materials science** | materials-project, ase |
| **Physics / Astronomy** | astropy, fluidsim |
| **Data analysis** | dask, exploratory-data-analysis, geopandas, infographics |
| **Research** | biorxiv-database, perplexity-search, hypothesis-generation, hypogenic, citation-management |
| **Lab automation** | benchling-integration, ginkgo-cloud-lab, adaptyv |
| **Financial** | alpha-vantage, fred-economic-data, edgartools, hedgefundmonitor |
| **Geospatial** | geopandas, geomaster |
| **Storage / Compute** | dnanexus-integration, parallel-web, datacommons-client |
| **Quantum** | cirq |
| **Scientific communication** | docx, infographics |

> 250+ databases, 60+ optimized package skills, 15+ integration skills.
> Always search — do not guess. The library grows continuously.
"""

    _TOPOLOGY_DESCRIPTIONS: dict[str, str] = {
        "functional": """\
**Functional topology** — Apple-style deep-expertise pipeline.
Control flow is a linear sequence where each agent is a "Directly Responsible Individual" (DRI) for their phase. Quality at the source is the priority; the `ml-engineer` expects perfect data from the `data-expert`.

```mermaid
graph LR
    TL[team-lead] --> DE[data-expert]
    DE --> FE[feature-engineer]
    FE --> ME[ml-engineer]
    ME --> EV[evaluator]
    EV --> V[validator]
    V --> MK[memory-keeper]
```

| Role | Responsibility |
| --- | --- |
| **team-lead** | Visionary alignment. Sets the "Product" (Experiment) roadmap. |
| **data-expert** | Data Integrity. Ensures the foundation is bulletproof. |
| **feature-engineer** | Technical Craft. Designs the most representative signals. |
| **ml-engineer** | Performance Engineering. Optimizes model architecture and CV. |
| **evaluator** | QA Testing. Validates the logs and artifact existence. |
| **validator** | Final Sign-off. Acts as the impartial judge for submission quality. |
| **memory-keeper** | Institutional Knowledge. Documents the "why" behind every success/failure. |

**Handoff contract:** Every executing role MUST write its result to `.claude/EXPERIMENT_STATE.json` as its final action. The topology reads this file to gate progression — a missing or malformed entry halts the pipeline.""",

        "two-pizza": """\
**Two-Pizza topology** — Amazon-style cross-functional ownership.
Instead of a relay race, a `full-stack-coordinator` owns the entire execution of a "Single-Threaded Goal." They delegate to specialists but remain responsible for the "Customer Outcome" (the score).

```mermaid
graph TD
    TL[team-lead] --> FSC[full-stack-coordinator]
    FSC --> DE[data-expert]
    FSC --> ME[ml-engineer]
    FSC --> V[validator]
    V --> MK[memory-keeper]
```

| Role | Responsibility |
| --- | --- |
| **team-lead** | Principle Enforcement. Ensures the experiment follows the 16 Leadership Principles. |
| **full-stack-coordinator** | Delivery Lead. Writes the core code, orchestrates runs, and fixes bugs in real-time. |
| **data-expert** | On-demand support for schema/EDA bottlenecks. |
| **ml-engineer** | On-demand support for hyperparameter tuning and CV structure. |
| **validator** | "Bar Raiser." Impartially checks if the result is better than the current best. |
| **memory-keeper** | Post-Mortem. Writes the "6-pager" summary into `MEMORY.md`. |""",

        "platform": """\
**Platform topology** — Google-style "Product vs Infra" layers.
The `platform-layer` builds the tools (scaffolding, data cleaning, evaluation suites) while the `product-layer` builds the model. This allows for rapid iteration on the model without rewriting boilerplate.

```mermaid
graph TD
    TL[team-lead] --> PL[Platform Layer: data-expert + evaluator]
    PL --> PR[Product Layer: feature-engineer + ml-engineer]
    PR --> V[validator]
    V --> MK[memory-keeper]
```

| Role | Responsibility |
| --- | --- |
| **team-lead** | OKR Setting. Defines the objective for the current "Quarter" (Iteration). |
| **platform-layer** | Tooling. `data-expert` builds the `src/` scaffold; `evaluator` builds the scoring harness. |
| **product-layer** | Implementation. `feature-engineer` and `ml-engineer` iterate within the platform's constraints. |
| **validator** | Launch Review. Checks if the experiment meets the "Product Requirements" (Score). |
| **memory-keeper** | Documentation. Updates the "Internal Wiki" (`MEMORY.md`). |""",

        "autonomous": """\
**Autonomous topology** — Meta-style parallel independent teams.
The `team-lead` generates N distinct experiment plans. N mini-teams run concurrently in their own isolated environments. The `validator` selects the "Evolutionary Winner."

```mermaid
graph TD
    TL[team-lead] --> P1[Mini-Team A]
    TL --> P2[Mini-Team B]
    TL --> P3[Mini-Team C]
    P1 & P2 & P3 --> V[validator]
    V --> MK[memory-keeper]
```

| Role | Responsibility |
| --- | --- |
| **team-lead** | Portfolio Manager. Diversifies the approach by generating multiple plans. |
| **mini-teams** | Rapid Prototyping. Each team (data-expert + feature-engineer + ml-engineer) builds a full pipeline for their specific plan. |
| **validator** | Selection Pressure. Picks the best OOF to promote to the leaderboard. |
| **memory-keeper** | Synthesis. Aggregates findings from all parallel branches into one cohesive history. |

**N is controlled by `--parallel N` CLI flag (default: 1).**""",

        "matrix": """\
**Matrix topology** — Microsoft-style dual-authority approval.
The `ml-engineer` executes the plan, but their work must be approved by *both* the technical lead (`team-lead`) and the science lead (`domain-expert`). This ensures no leakage and high scientific rigor.

```mermaid
graph TD
    TL[team-lead] --> ME[ml-engineer]
    DE[domain-expert] --> ME
    ME --> Approval{Dual Review}
    Approval --> V[validator]
    V --> MK[memory-keeper]
```

| Role | Responsibility |
| --- | --- |
| **team-lead** | Technical Authority. Ensures the code and logic are sound. |
| **domain-expert** | Scientific Authority. Checks for data leakage, business logic errors, and domain validity. |
| **ml-engineer** | Implementation Partner. Works across both leads to build a valid solution. |
| **evaluator / validator** | Compliance. Ensures the result meets all organizational standards. |
| **memory-keeper** | Knowledge Transfer. Shares the collaborative learnings across the "Ecosystem." |""",
    }
    topology_desc = _TOPOLOGY_DESCRIPTIONS.get(
        state.topology,
        f"**{state.topology}** — no description available.",
    )

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
| Topology | `{state.topology}` |

## Management Topology

{topology_desc}

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

If you are the **team lead**, read your agent memory at
`{Path(project_dir).resolve()}/.claude/agent-memory/team-lead/MEMORY.md` before planning.
Update it with insights after each iteration.

## Project Structure

Organise all code under `src/` and `scripts/` as importable modules:

```
src/
├── __init__.py
├── config.py      # constants: paths, metric, target column
├── data.py        # load_train(), load_test() helpers
├── features.py    # feature engineering transforms
└── models.py      # model factory + CV loop
scripts/
├── train.py       # entry point: fit, save artifacts/oof.npy, print OOF score
└── evaluate.py    # reload artifacts/oof.npy and report metric
```

- `scripts/train.py` MUST print: `OOF <metric>: <value>` — the evaluator parses this.
- Save OOF predictions to `artifacts/oof.npy` (and `artifacts/oof_classes.npy` for multiclass).
- Use `pathlib` throughout; no hardcoded absolute paths.
- Set `random_state=42` everywhere for reproducibility.

## Package Management

> **This project uses `uv` — there is no `pip` in the venv.**
> **Always install packages with:** `uv add <package>`
> Never use `pip install`, `pip3 install`, or `python -m pip install`.

```bash
uv add optuna catboost lightgbm   # install packages
uv run python scripts/train.py    # run a script inside the venv
```

## Long-Running Scripts

For training scripts that take more than a few seconds, use `nohup` and track the PID.
**Never use background task IDs (`TaskOutput`, `TaskStop`).**

```bash
# Launch and capture PID
nohup uv run python scripts/train.py > train.log 2>&1 & echo $!

# Check if still running
ps -p <PID> -o pid,stat,etime,cmd --no-headers 2>/dev/null || echo "done"

# Tail progress
tail -n 50 train.log

# Wait for finish
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
    - .claude/agents/<role>.md       (one per topology role)
    - .claude/skills/<name>/SKILL.md  (one per skill)
    - .claude/settings.json           (hooks + env)
    - .mcp.json                       (MCP server registrations for CLI use)
    - scripts/after_edit.sh           (PostToolUse hook: compile Python on edit)
    - scripts/validate_bash.sh        (PreToolUse hook: block dangerous rm -rf)
    """
    root = Path(project_dir)
    is_ml = bool(state.target_metric)

    _write_subagents(root)

    # Copy all 170+ upstream scientific skills first (idempotent).
    _copy_all_scientific_skills(root)

    if is_ml:
        _copy_skill(root, "ml-setup")
        _copy_skill(root, "submit-check")
        _copy_skill(root, "adversarial-validation")
        _copy_skill(root, "feature-engineering")
        _copy_skill(root, "hpo")
        _copy_skill(root, "ensembling")
    _copy_skill(root, "code-review")

    _copy_skill(root, "git-workflow")

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


def _write_subagents(root: Path) -> None:
    """Copy all agent templates into .claude/agents/ (idempotent).

    All *.md files in templates/agents/ are treated as topology role agents.
    New files are only written once — existing files are preserved so teams can
    customise them without being overwritten on every run.

    {{GLADIUS_SMALL_MODEL}} and {{GLADIUS_MODEL}} are substituted at copy time.
    Defaults to "inherit" / "GLADIUS_MODEL_NOT_SET" when env vars are unset.
    """
    agents_dir = root / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    small_model = os.environ.get("GLADIUS_SMALL_MODEL", "inherit")
    managed: set[str] = set()
    for src in sorted((_TEMPLATES / "agents").glob("*.md")):
        if src.stem in managed:
            continue
        dest = agents_dir / src.name
        if not dest.exists():
            content = src.read_text(encoding="utf-8").replace(
                "{{GLADIUS_SMALL_MODEL}}", small_model
            )
            dest.write_text(content, encoding="utf-8")


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


def _copy_skill(
    root: Path, template_name: str, *, dest_name: str | None = None
) -> None:
    """Copy a skill template into the competition's .claude/skills/ tree (idempotent).

    If the template is a directory (new multi-file format with SKILL.md + references/
    + scripts/ subdirectories), the whole tree is copied.  Falls back to the legacy
    single-file format (.md) for any skill not yet converted.
    """
    skill_name = dest_name or template_name
    dest = root / ".claude" / "skills" / skill_name
    if (dest / "SKILL.md").exists():
        return
    template_dir = _TEMPLATES / "skills" / template_name
    if template_dir.is_dir():
        shutil.copytree(template_dir, dest, dirs_exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text(
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
    """Pre-create the team lead's memory directory so it exists on first run."""
    mem_dir = root / ".claude" / "agent-memory" / "team-lead"
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
