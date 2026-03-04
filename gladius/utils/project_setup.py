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
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gladius.state import CompetitionState

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

    # Skills section — lists all available skill files so agents know to use them
    if state.target_metric:
        skills_section = """\
## Available Skills

Invoke skills with the `Skill` tool — the output is returned inline, do NOT use TaskOutput.

| Skill name | When to invoke |
| --- | --- |
| `code-review` | **Required before reporting results** — catches leakage & metric bugs |
| `ml-pipeline` | Writing CV / feature / submission code |
| `submit-check` | Validate submission CSV before uploading |
| `git-workflow` | After each working solution |
| `uv-venv` | Installing packages and running scripts |
"""
    else:
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
`.claude/agent-memory/planner/MEMORY.md` before exploring.
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
    - .claude/skills/ml-pipeline/SKILL.md
    - .claude/skills/submit-check/SKILL.md
    - .claude/settings.json  (hooks + env)
    - .mcp.json              (MCP server registrations for CLI use)
    - scripts/after_edit.sh  (PostToolUse hook: compile Python on edit)
    - scripts/validate_bash.sh (PreToolUse hook: block dangerous rm -rf)
    """
    root = Path(project_dir)

    _write_agent_planner(root)
    _write_agent_implementer(root)
    if state.target_metric:
        _write_skill_ml_pipeline(root)
    else:
        _write_skill_task_review(root)
    if state.target_metric:
        _write_skill_submit_check(root)
    _write_skill_git_workflow(root, bool(state.target_metric))
    _write_skill_uv_venv(root)
    _write_skill_code_review(root, bool(state.target_metric))
    _write_claude_settings(root, state)
    _write_mcp_json(root, platform=platform)
    _write_hook_after_edit(root)
    _write_hook_validate_bash(root)
    _make_memory_dir(root, bool(state.target_metric))


# ── Agent definitions ─────────────────────────────────────────────────────────


def _write_agent_planner(root: Path) -> None:
    # Always overwrite so that permission mode and content stay in sync.
    path = root / ".claude" / "agents" / "planner.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""\
---
name: planner
description: >
  Expert analyst for ML competitions and open-ended engineering tasks.
  Explores data or requirements, reviews experiment history, and decides
  the highest-impact next approach. Use proactively at the start of each
  iteration.
tools: Read, Glob, Grep, WebSearch, TodoWrite
model: {os.environ.get("GLADIUS_MODEL", "GLADIUS_MODEL_NOT_SET")}
memory: project
maxTurns: 40
permissionMode: plan
---

You are an expert analyst for ML competitions and open-ended engineering tasks.

**Start every session by:**
1. Reading `CLAUDE.md` in the current directory for task state.
2. Reading your agent memory at `.claude/agent-memory/planner/MEMORY.md`.
3. Exploring the data directory (ML) or existing deliverables (open tasks).

**Your job:**
- Understand what has already been tried (from CLAUDE.md experiments table).
- For ML: identify the highest-impact next model/feature/tuning approach.
- For open tasks: identify the next deliverable improvement or missing feature.
- Produce a concrete, ordered action plan the implementer can execute blindly.

**STRICT RULES — you are in READ-ONLY planning mode:**
- You NEVER run Bash commands.
- You NEVER write or edit ANY files — not MEMORY.md, not plan files, nothing.
- You NEVER spawn Task subagents.
- You NEVER write implementation code.
- Plans must be specific and self-contained — no "investigate X" steps.
- WebSearch for domain-specific techniques when you lack knowledge.
- Call ExitPlanMode when your plan is ready — that is the ONLY output channel.
- The orchestrator's summarizer handles MEMORY.md updates; you do NOT touch it.
""",
        encoding="utf-8",
    )


def _write_agent_implementer(root: Path) -> None:
    # Always overwrite so that permission mode and content stay in sync.
    path = root / ".claude" / "agents" / "implementer.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""\
---
name: implementer
description: >
  Expert engineer. Executes the planner's plan end-to-end: writes code,
  runs it, debugs errors, measures performance, and reports results.
  Works for both ML competitions and open-ended engineering tasks.
  Always gets a fresh context — does not retain knowledge between runs.
tools: Read, Write, Edit, Bash, Glob, Grep
model: {os.environ.get("GLADIUS_MODEL", "GLADIUS_MODEL_NOT_SET")}
maxTurns: 80
permissionMode: bypassPermissions
---

You are an expert engineer executing a task experiment.

**Task type detection:**
- Read `CLAUDE.md` — if `TARGET_METRIC` is set → ML competition.
- If no `TARGET_METRIC` → open-ended task (app, script, analysis, etc.).

**Start every session by:**
1. Reading `CLAUDE.md` in the current directory for task context.
2. Reading the plan provided to you in the task description.

**Your job (ML competition):**
- Implement the plan completely.
- Run the code, fix any errors you encounter, repeat until it works.
- Measure the target metric using OOF / cross-validation.
- Produce a `submission.csv` for the test set.
- Report the final metric score and all file paths you created.

**Your job (open-ended task):**
- Implement the plan completely per the requirements in `README.md`.
- Run the deliverable end-to-end and verify it works.
- Invoke the `task-review` skill using the Skill tool: `Skill({{"name": "task-review"}})`
  The skill output is returned directly — do NOT use TaskOutput.
  Use the checklist to self-assess quality 0–100 before reporting.
- Package the deliverable (zip / binary / URL file) as described in README.

**Rules:**
- You decide file names, libraries, and code structure. No constraints.
- Keep all created files; never delete previous solutions.
- **NEVER modify or overwrite `CLAUDE.md`** — it is managed exclusively by the orchestrator.
- **NEVER spawn Task subagents.**

**When you're done, before reporting results:**
- Invoke the `code-review` skill using the Skill tool: `Skill({{"name": "code-review"}})`
  The skill output is returned directly in the same turn — do NOT use TaskOutput to wait for it.
  Fix every CRITICAL item it reports before finalising.

**Report:**
- `status`: success | error | timeout | oom
- `oof_score`: OOF metric value (ML only; null for open tasks)
- `quality_score`: self-assessed quality 0–100 (always required)
- `solution_files`: list of all files you created or modified
- `submission_file`: path to submission CSV / deliverable artifact
- `notes`: brief summary of what you built and the score
- `error_message`: (only on failure) what went wrong
""",
        encoding="utf-8",
    )


# ── Skills ────────────────────────────────────────────────────────────────────


def _write_skill_ml_pipeline(root: Path) -> None:
    path = root / ".claude" / "skills" / "ml-pipeline" / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """\
---
name: ml-pipeline
description: >
  ML competition pipeline best practices. Auto-loaded when writing or
  debugging competition ML code.
user-invocable: false
---

## Validation

- **Classification**: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **Regression**: KFold(n_splits=5, shuffle=True, random_state=42)
- Fit on train folds, predict on validation fold, aggregate OOF predictions.
- Compute the competition metric on the full OOF array (not fold averages).

## Baselines

- Tabular: **LightGBM** or **XGBoost** first (fast, strong default).
- Deep learning: only after a solid gradient boosting baseline exists.
- Always record baseline score before any feature engineering.

## Feature Engineering

- Log every engineered feature in a comment explaining the hypothesis.
- Never fit target-based encodings on the full training set — always compute
  within each CV fold to avoid leakage.
- Drop ID columns and date columns (unless engineering lag features).

## Metrics

| Competition type | Metric | sklearn call |
| --- | --- | --- |
| Binary classification | AUC-ROC | `roc_auc_score(y_true, y_score)` |
| Multiclass | Log loss | `log_loss(y_true, y_proba)` |
| Regression | RMSE | `mean_squared_error(y_true, y_pred, squared=False)` |

Always `print(f"OOF {metric_name}: {score:.6f}")` so the score appears in logs.

## Submission Format

```python
import pandas as pd
sample_sub = pd.read_csv("data/sample_submission.csv")  # or data_dir
sub = sample_sub.copy()
sub[sub.columns[-1]] = test_predictions  # fill prediction column
assert len(sub) == len(sample_sub), "Row count mismatch!"
assert not sub.isnull().any().any(), "NaN in submission!"
sub.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(sub)} rows, columns: {list(sub.columns)}")
```

## File Naming

- Name solution scripts descriptively: `solution_lgbm_baseline.py`, `solution_xgb_v2.py`
- Keep ALL previous solution files — never delete older versions.
- Write a `run.sh` if the solution has multiple steps.
""",
        encoding="utf-8",
    )


def _write_skill_task_review(root: Path) -> None:
    path = root / ".claude" / "skills" / "task-review" / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """\
# Skill: Task Review & Quality Self-Assessment

## When to use this skill

Use this skill when you have completed a deliverable and need to:
1. Verify it meets the requirements in `README.md`
2. Self-assign a quality score (0–100) for reporting
3. Decide whether the deliverable is ready to package and submit

---

## Quality Score Guide (0–100)

| Score | Meaning |
| --- | --- |
| 90–100 | Exceeds requirements; polished, documented, tested |
| 70–89 | Meets all stated requirements; no major gaps |
| 50–69 | Meets most requirements; some gaps or rough edges |
| 30–49 | Partial implementation; core functionality works |
| 0–29 | Incomplete; significant requirements unmet |

---

## Review Checklist

Before reporting your quality score, verify each item:

### Functional Correctness
- [ ] Run the deliverable end-to-end (e.g., `uv run python app.py` or `./run.sh`)
- [ ] Outputs match what `README.md` asks for (format, content, location)
- [ ] No unhandled errors or crashes on the happy path

### Completeness
- [ ] All required features listed in `README.md` are implemented
- [ ] Any configuration/secrets described in README are handled
- [ ] Dependencies are declared (e.g., in `pyproject.toml`)

### Packaging
- [ ] All required files are present
- [ ] Deliverable can be reproduced from scratch with documented steps
- [ ] Submission artifact is saved with the path reported in `submission_file`

---

## Scoring Process

1. Read `README.md` and extract the explicit success criteria.
2. Run through the checklist above.
3. Assign a score 0–100 based on the guide.
4. Write 1–2 sentences justifying the score in your output `notes` field.

---

## Packaging the Deliverable

```bash
# Zip the output for submission
zip -r deliverable.zip output/ app.py requirements.txt README_submission.md

# Or record a URL / binary path in a text file
echo "https://..." > submission_url.txt
```

Report `submission_file` as the path to the zip / binary / URL file.
""",
        encoding="utf-8",
    )


def _write_skill_submit_check(root: Path) -> None:
    path = root / ".claude" / "skills" / "submit-check" / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """\
---
name: submit-check
description: Validate a submission CSV before platform upload
disable-model-invocation: true
---

Validate the submission file at $ARGUMENTS against the sample submission.

Steps:
1. Load `data/sample_submission.csv` (or find sample_submission.csv in the data dir)
2. Load the submission at `$ARGUMENTS`
3. Check that column names match exactly (same order and names)
4. Check that row counts match
5. Check for NaN or Inf in all numeric columns
6. For probability predictions: check all values are in [0, 1]
7. For regression: check values are in a sane range (warn if extreme outliers)

Output:
- `VALID` if all checks pass
- `INVALID: <reason>` listing specific issues
""",
        encoding="utf-8",
    )


def _write_skill_code_review(root: Path, is_ml: bool) -> None:
    path = root / ".claude" / "skills" / "code-review" / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_ml:
        content = """\
---
name: code-review
description: Review ML solution code before finalising — catch leakage, metric errors, and format bugs
---

Before finalising any solution script, review it against this checklist.
Fix every item marked CRITICAL before reporting results.

## CRITICAL — data leakage

- [ ] Target-based encodings (mean encoding, target encoding) are computed
      **inside** each CV fold, never on the full training set.
- [ ] Temporal features (lag, rolling stats) use only past data — no future leakage.
- [ ] No test-set rows accidentally appear in the training fold.
- [ ] StandardScaler / other transformers are fit on train fold only, then applied to val/test.
- [ ] `train_test_split` is NOT used instead of proper k-fold CV.

## CRITICAL — metric correctness

- [ ] The OOF metric is computed on out-of-fold predictions, not train predictions.
- [ ] The metric function matches the competition definition exactly
      (e.g. `average='macro'` vs `average='binary'` for F1).
- [ ] For probability metrics (AUC, log-loss): predictions are probabilities, not class labels.
- [ ] The metric direction (maximize/minimize) is respected when comparing scores.

## CRITICAL — submission format

- [ ] Submission CSV column names match `sample_submission.csv` exactly.
- [ ] Submission row count matches `sample_submission.csv` exactly.
- [ ] No NaN or Inf in prediction column.
- [ ] File is saved to the path reported in `submission_file`.

## Important — robustness

- [ ] No hard-coded file paths — use variables from CLAUDE.md context.
- [ ] Random seeds set for reproducibility (`random_state=42`, `np.random.seed(42)`).
- [ ] OOF score is printed as `OOF {metric}: {score:.6f}` so it appears in logs.
- [ ] Script runs end-to-end without manual intervention.

## Style

- [ ] Each feature engineering step has a comment explaining the hypothesis.
- [ ] File name is descriptive: `solution_lgbm_v2.py`, not `solution.py`.
"""
    else:
        content = """\
---
name: code-review
description: Review deliverable code before finalising — catch crashes, missing features, and packaging issues
---

Before finalising any deliverable, review it against this checklist.
Fix every item marked CRITICAL before reporting results.

## CRITICAL — functional correctness

- [ ] Run the deliverable end-to-end: `uv run python app.py` or `./run.sh` — no crashes.
- [ ] Every feature listed in `README.md` is implemented and reachable.
- [ ] Edge cases handled: empty input, invalid input, missing files.
- [ ] No hardcoded paths — all paths are relative or configurable.

## CRITICAL — completeness

- [ ] All required files are present (app, config, dependencies, README).
- [ ] Dependencies declared in `pyproject.toml` (not just installed ad-hoc).
- [ ] The deliverable can be reproduced from scratch by a fresh `uv sync && uv run ...`.

## CRITICAL — packaging

- [ ] Submission artifact exists at the path reported in `submission_file`.
- [ ] Artifact contains everything needed to run or evaluate the deliverable.
- [ ] `README.md` (or equivalent) explains how to install and run.

## Important — robustness

- [ ] No unhandled exceptions on the happy path.
- [ ] Environment variables or config files used for secrets — not hardcoded.
- [ ] Script/app runs without user interaction (unless README explicitly requires it).

## Style

- [ ] Code is readable: functions have docstrings, logic is commented where non-obvious.
- [ ] File names are descriptive and consistent.
"""
    path.write_text(content, encoding="utf-8")


def _write_skill_git_workflow(root: Path, is_ml: bool) -> None:
    path = root / ".claude" / "skills" / "git-workflow" / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_ml:
        commit_example = (
            'git commit -m "iter-{N}: {approach_summary} — OOF {metric}={score:.6f}"'
        )
        msg_format = "`iter-{N}: <one-sentence approach> — OOF {metric}={score:.6f}`"
        trigger = "After implementing a solution that runs without errors and produces an OOF score,"
    else:
        commit_example = (
            'git commit -m "iter-{N}: {change_summary} — quality {score}/100"'
        )
        msg_format = "`iter-{N}: <one-sentence change> — quality {score}/100`"
        trigger = (
            "After implementing a deliverable that runs end-to-end without errors,"
        )
    path.write_text(
        f"""\
---
name: git-workflow
description: Commit every working iteration with a descriptive message
---

{trigger}
stage and commit it locally:

```bash
git add -A
{commit_example}
```

Guidelines:
- Commit only after the deliverable runs end-to-end without errors.
- Use `git add -A` to stage everything (source files, artifacts, run scripts).
- Message format: {msg_format}
- Never force-push. Never rebase during a run.
- If a run fails, do NOT commit. Just proceed to the next iteration.
- Check `git status` before committing to avoid committing unintended files.
- `.gladius/` and `data/` should already be in `.gitignore` — verify if unsure.

Tip: use `git log --oneline -10` to review recent iteration history.
""",
        encoding="utf-8",
    )


def _write_skill_uv_venv(root: Path) -> None:
    path = root / ".claude" / "skills" / "uv-venv" / "SKILL.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """\
---
name: uv-venv
description: Install Python packages and run scripts using uv
---

This project uses **uv** for fast Python package management.
Never use `pip install` directly — always use `uv add` or `uv run`.

## Installing packages

```bash
# Add a runtime dependency (updates pyproject.toml + uv.lock)
uv add lightgbm scikit-learn pandas numpy

# Add a dev/optional dependency
uv add --dev pytest

# Sync the environment (install all locked deps)
uv sync
```

## Running scripts

```bash
# Run a script inside the venv without activating it
uv run python script.py

# Run with extra args
uv run python script.py --arg value

# Run a one-liner
uv run python -c "import numpy; print(numpy.__version__)"
```

## Checking installed packages

```bash
uv pip list
uv pip show lightgbm
```

## Notes
- `uv` is already installed. The venv is in `.venv/`.
- Prefer `uv add` over editing `pyproject.toml` manually.
- `uv.lock` should be committed to git for reproducibility.
- If a package needs a non-PyPI source (e.g. GPU build), use
  `uv add --index https://... package_name`.
""",
        encoding="utf-8",
    )


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
        # Platform-specific MCP tools are injected per-call by the validation agent;
        # register the server here for any Claude CLI interactive use.
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


# ── Hook scripts ──────────────────────────────────────────────────────────────


def _write_hook_after_edit(root: Path) -> None:
    path = root / "scripts" / "after_edit.sh"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """\
#!/usr/bin/env bash
# PostToolUse hook — runs after Edit or Write tool calls.
# Compiles any modified Python file immediately so Claude sees syntax errors
# in the same turn rather than discovering them at runtime.
#
# Exit code 2 = send error back to Claude and block (Claude will fix it).
# Exit code 0 = success, continue normally.

INPUT=$(cat)

# Extract file path from tool_input (Edit sets 'path', Write sets 'path')
FILE_PATH=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input', {})
    print(ti.get('file_path', ''))  # SDK Write and Edit both use 'file_path'
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

if [[ "$FILE_PATH" == *.py ]] && [[ -f "$FILE_PATH" ]]; then
    ERRORS=$(python3 -m py_compile "$FILE_PATH" 2>&1)
    if [[ -n "$ERRORS" ]]; then
        echo "Syntax error detected in $FILE_PATH — fix before continuing:" >&2
        echo "$ERRORS" >&2
        exit 2
    fi
fi

exit 0
""",
        encoding="utf-8",
    )
    _make_executable(path)


def _write_hook_validate_bash(root: Path) -> None:
    path = root / "scripts" / "validate_bash.sh"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """\
#!/usr/bin/env bash
# PreToolUse hook — runs before every Bash tool call.
# Blocks commands that could destroy data outside the project directory.
#
# Exit code 2 = block the command and send the error message to Claude.
# Exit code 0 = allow the command.

INPUT=$(cat)

COMMAND=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

# Block recursive delete of absolute paths (outside project)
if echo "$COMMAND" | grep -qE 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f[[:space:]]+/'; then
    echo "Blocked: 'rm -rf /' style command is not allowed. Use relative paths only." >&2
    exit 2
fi

# Block recursive delete of home directory
if echo "$COMMAND" | grep -qE 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f[[:space:]]+~'; then
    echo "Blocked: 'rm -rf ~' is not allowed." >&2
    exit 2
fi

# Block any bash modification of CLAUDE.md (e.g. cat >> CLAUDE.md, tee CLAUDE.md)
if echo "$COMMAND" | grep -qE 'CLAUDE\\.md'; then
    if echo "$COMMAND" | grep -qE '(>>|>|tee|sed -i|awk.*>|perl.*-i|patch|truncate)'; then
        echo "Blocked: modifying CLAUDE.md via Bash is not allowed. CLAUDE.md is managed by the orchestrator." >&2
        exit 2
    fi
fi

exit 0
""",
        encoding="utf-8",
    )
    _make_executable(path)


# ── Agent memory directory ────────────────────────────────────────────────────


def _make_memory_dir(root: Path, is_ml: bool) -> None:
    """Pre-create the planner's memory directory so it exists on first run."""
    mem_dir = root / ".claude" / "agent-memory" / "planner"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_file = mem_dir / "MEMORY.md"
    if not mem_file.exists():
        if is_ml:
            score_col = "OOF"
            insights_label = "Key Data Insights"
            insights_hint = "_(Add notes here as you explore the dataset)_"
            first_direction = "1. Establish a baseline (LightGBM or XGBoost) before any feature engineering."
        else:
            score_col = "Quality"
            insights_label = "Key Task Insights"
            insights_hint = "_(Add notes here as you explore the requirements and existing deliverables)_"
            first_direction = "1. Read README.md end-to-end and list all explicit success criteria before building anything."
        mem_file.write_text(
            f"""\
# Planner Memory

> Auto-updated by summarizer. Last iteration: 0

## {insights_label}

{insights_hint}

## What Works  ✅

_(Record successful approaches with approximate score delta and iteration number)_

## What Fails / Dead Ends  ❌

_(Record approaches that hurt score, timed out, or errored — include the reason)_

## Patterns & Hypotheses  💡

_(Open hypotheses not yet tested, or patterns observed)_

## Experiment Score History

| iter | {score_col} | approach | notes |
| --- | --- | --- | --- |

## Suggested Next Directions

{first_direction}
""",
            encoding="utf-8",
        )


# ── Utility ───────────────────────────────────────────────────────────────────


def _make_executable(path: Path) -> None:
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
