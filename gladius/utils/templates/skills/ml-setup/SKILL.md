---
name: ml-setup
description: >
  ML competition setup: canonical src/ project layout + CV pipeline patterns.
  Invoke FIRST before writing any solution code. Sets up directory tree,
  module templates, and provides CV/metric/baseline patterns for the full run.
---

# ML Setup

Two things every competition needs from the start:
1. **Project layout** — a `src/` package that stays organised across iterations
2. **Pipeline patterns** — correct CV, metrics, and baseline templates

---

## ⚠️ CRITICAL: Working Directory

All shell commands must be run from the **competition root** — the directory that contains `CLAUDE.md`.

```bash
pwd          # must print the competition root
ls CLAUDE.md # must exist here
```

If `CLAUDE.md` is not here, `cd` to where it is first.

---

## Part 1 — Project Layout

### Directory Tree

```
<competition-root>/
├── CLAUDE.md
├── pyproject.toml        ← managed by uv
├── uv.lock
├── .venv/
│
├── src/                  ← all importable project code
│   ├── __init__.py
│   ├── config.py         ← DATA_DIR, SEED, N_FOLDS, TARGET, OUTPUT_DIR
│   ├── data.py           ← load_train(), load_test(), get_cv()
│   ├── features.py       ← add_features(df) → df (never mutates in-place)
│   ├── models.py         ← make_model(), fit_predict()
│   ├── metrics.py        ← competition_metric(y_true, y_pred) → float
│   └── submission.py     ← make_submission(test_df, preds) → saves CSV
│
├── scripts/              ← thin CLI entry-points (20–40 lines, all logic in src/)
│   └── train.py          ← full CV run; prints OOF score; saves submission
│
├── notebooks/            ← exploratory only (never imported by scripts)
│   └── eda.ipynb
│
├── artifacts/            ← saved models, OOF arrays, encoder pickles
│   └── .gitkeep
│
└── submissions/          ← submission CSVs ready to upload
    └── .gitkeep
```

See `references/layout.md` for module template code.

### Initialise the Project

Run once from the competition root (idempotent — safe to re-run):

```bash
bash scripts/init.sh
```

### Module Rules

1. **No logic in scripts** — `scripts/train.py` is 20–40 lines; all logic in `src/`.
2. **Functions return new objects** — never modify input DataFrames in-place.
3. **One responsibility per module** — `features.py` never trains models.
4. **Fix all random seeds** — use `src/config.SEED` everywhere.
5. **Save artifacts** — OOF arrays and models go in `artifacts/`; enables blending without re-training.

---

## Part 2 — Pipeline Patterns

### Cross-Validation Setup

| Competition type | Splitter |
| --- | --- |
| Classification | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Regression | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| Time-series | `TimeSeriesSplit(n_splits=5)` — **never shuffle** |

Always:
- Fit on train folds, predict on the validation fold only.
- Accumulate OOF predictions into one array of length `len(train)`.
- Compute the final metric on the **full OOF array** (not fold-by-fold averages).

### Competition Metrics

See `references/metrics.md` for the full table: sklearn calls, direction, and common mistakes.

### Baseline Models

- **Tabular:** LightGBM first; add XGBoost / CatBoost for diversity.
- **NLP:** Invoke the `transformers` skill for HuggingFace fine-tuning.
- **Time-series:** Invoke the `timesfm-forecasting` skill for zero-shot; lag + LightGBM for supervised.

### Full CV Training Script

See `scripts/cv_train.py` for a complete template: load → CV loop → aggregate OOF → score → save submission.

---

## Quick Workflow

1. `bash scripts/init.sh` from competition root (creates `src/`, `scripts/`, dirs)
2. Edit `src/config.py` — set `TARGET`, `DATA_DIR`, `N_FOLDS`
3. Edit `src/metrics.py` — paste the competition metric
4. Copy `scripts/cv_train.py` as `scripts/train.py` and run it
5. Iterate: add features in `src/features.py`, tune in `src/models.py`

---

## Resources

### scripts/
- `init.sh` — idempotent project tree bootstrap
- `cv_train.py` — complete CV training template with LightGBM

### references/
- `layout.md` — module template code for config, data, features, models, metrics, submission
- `metrics.md` — competition metric reference table
