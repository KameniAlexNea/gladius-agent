---
name: ml-pipeline
description: >
  ML competition pipeline patterns: cross-validation setup, baseline models,
  submission format building, and metric formulas. Auto-loaded when writing
  any competition ML code. Covers tabular, NLP, and time-series baselines.
---

# ML Pipeline

## Overview

A competition ML pipeline has four fixed stages: CV setup → baseline training → feature loop → submission building. This skill provides correct patterns for each stage so you spend time on features, not boilerplate bugs.

## When to Use This Skill

- Writing the first solution script for a competition.
- Finding the correct sklearn call for a competition metric.
- Debugging OOF vs leaderboard discrepancies (likely a CV or submission bug).

## Core Capabilities

### 1. Cross-Validation Setup

**Classification (imbalanced-safe):** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

**Regression:** `KFold(n_splits=5, shuffle=True, random_state=42)`

**Time-series (no shuffle):** `TimeSeriesSplit(n_splits=5)` — never shuffle time-ordered data.

Always:
- Fit on train folds, predict on the validation fold only.
- Accumulate OOF predictions into a single array of length `len(train)`.
- Compute the final metric on the **full OOF array** (not fold-by-fold averages).

### 2. Competition Metrics

Read `references/metrics.md` for the full table of metrics, their sklearn calls, and common mistakes.

### 3. Baseline Models

- **Tabular:** LightGBM first; add XGBoost / CatBoost for diversity.
- **NLP:** Invoke `transformers` skill for HuggingFace fine-tuning.
- **Time-series:** Invoke `timesfm` skill for zero-shot; lag + LightGBM for supervised.

### 4. Full CV Training Script

See `scripts/cv_train.py` for a complete end-to-end template: load data → CV loop → aggregate OOF → compute score → save submission.

## Quick Workflow

1. Load `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.
2. Copy `scripts/cv_train.py` as your starting point.
3. Replace `TARGET`, `DATA_DIR`, and the metric function.
4. Run — it prints `OOF {metric}: {score:.6f}` and saves `submission.csv`.

## Resources

### scripts/
- `cv_train.py` — complete CV training template with LightGBM

### references/
- `metrics.md` — competition metric table, sklearn calls, common mistakes
