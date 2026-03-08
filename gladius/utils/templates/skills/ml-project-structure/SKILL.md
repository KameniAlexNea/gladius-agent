---
name: ml-project-structure
description: >
  Canonical ML competition project layout using a src/ package. Invoke FIRST,
  before writing any solution code. Sets up the directory tree, pyproject.toml,
  and module templates so the codebase stays organised across many iterations.
---

# ML Project Structure

## Overview

Every competition solution is a mini-package. Breaking code into small, focused modules lets you test, reuse, and replace each part independently. Never put everything in one monolithic script.

**Invoke this skill FIRST — before writing any other code.**

## ⚠️ CRITICAL: Working Directory

**All commands in this skill must be run from the competition root** (`<competition-root>/`), not from inside `.claude/`, `.gladius/`, `data/`, or any subdirectory.

Before anything, confirm you are in the right place:
```bash
pwd   # must print the competition root (same directory that contains CLAUDE.md)
ls CLAUDE.md  # must exist here
```

If `CLAUDE.md` is not in the current directory, `cd` to where it is first.

## When to Use This Skill

- Starting the first iteration — run `scripts/init.sh` **once** from the competition root.
- Resuming after a crash — verify the `src/` layout is intact.
- Before writing any model or feature code — the package must exist first.

## Core Capabilities

### 1. Canonical Project Layout

```
<competition-root>/          ← you must be here when running any command
├── CLAUDE.md
├── pyproject.toml           ← managed by uv
├── uv.lock
├── .venv/
│
├── src/                     ← all importable project code
│   ├── __init__.py
│   ├── config.py            ← DATA_DIR, SEED, N_FOLDS, TARGET, OUTPUT_DIR
│   ├── data.py              ← load_train(), load_test(), get_cv()
│   ├── features.py          ← add_features(df) → df (never mutates in-place)
│   ├── models.py            ← make_model(), fit_predict()
│   ├── metrics.py           ← competition_metric(y_true, y_pred) → float
│   └── submission.py        ← make_submission(test_df, preds) → saves CSV
│
├── scripts/                 ← thin CLI entry-points that import from src/
│   └── train.py             ← full CV run; prints OOF score; saves submission
│
├── notebooks/               ← exploratory work only (never import from here)
│   └── eda.ipynb
│
├── artifacts/               ← saved models, OOF arrays, encoder pickles
│   └── .gitkeep
│
└── submissions/             ← submission CSVs ready to upload
    └── .gitkeep
```

See `references/layout.md` for detailed module responsibilities and template code.

### 2. Initialise the Project

Run `scripts/init.sh` from the **competition root**. It is idempotent — safe to re-run.

```bash
bash scripts/init.sh
```

This creates the directory tree, writes an empty `src/__init__.py`, and creates placeholder `artifacts/.gitkeep` and `submissions/.gitkeep`.

### 3. Module Templates

Read `references/layout.md` for template code for each module:
- `src/config.py` — paths, constants, CV settings
- `src/data.py` — load functions, CV splitter
- `src/features.py` — feature engineering entry-point (never mutates in-place)
- `src/models.py` — model factory, `fit_predict()` helper
- `src/metrics.py` — competition metric function
- `src/submission.py` — build and validate submission CSV

### 4. Module Rules

1. **No logic in scripts** — `scripts/train.py` is 20–40 lines; all logic lives in `src/`.
2. **Functions return new objects** — never modify input DataFrames in-place.
3. **One responsibility per module** — `features.py` never trains models.
4. **Fix all random seeds** — use `src/config.SEED` everywhere.
5. **Save artifacts** — OOF arrays and models go in `artifacts/`; blending works without re-training.

## Quick Workflow

1. `bash scripts/init.sh` (from competition root)
2. Edit `src/config.py` — set `DATA_DIR`, `TARGET`, `N_FOLDS`.
3. Edit `src/data.py` — verify `load_train()` and `load_test()` return the right DataFrames.
4. Copy `ml-pipeline`'s `cv_train.py` into `scripts/train.py` and update imports.
5. `uv run python scripts/train.py`

## Resources

### scripts/
- `init.sh` — idempotent project initialisation (run from competition root)

### references/
- `layout.md` — full module responsibilities with template code for each file
