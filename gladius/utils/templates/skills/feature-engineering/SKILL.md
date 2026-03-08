---
name: feature-engineering
description: >
  Systematic feature generation recipes, SHAP-based importance measurement,
  and feature pruning for ML competitions. Use after a baseline is established.
  Covers numerical, categorical, temporal, and aggregation features — all
  with leakage prevention built in.
---

# Feature Engineering

## Overview

Feature engineering is the highest-ROI activity in tabular competitions after a solid CV baseline exists. This skill provides ready-to-use recipes for every feature type, a SHAP-based workflow for measuring importance, and a disciplined pruning process. For detailed SHAP documentation, also invoke the `shap` skill.

## When to Use This Skill

- Baseline OOF score is established (LightGBM / XGBoost, raw features).
- You want to systematically improve features before HPO.
- OOF score has plateaued and you need orthogonal improvements.
- You need to identify which features are driving model decisions.

## Core Capabilities

### 1. Leakage Prevention (ALWAYS READ FIRST)

Read `references/safety_rules.md` before writing any feature code. Data leakage is the most common bug in competition solutions.

### 2. Numerical Feature Recipes

See `scripts/numerical_features.py` for log transforms, binning, Z-score outlier flags, and pairwise interactions.

### 3. Categorical Feature Recipes

See `scripts/categorical_features.py` for frequency encoding, target encoding (fold-safe), rare-group collapsing, ordinal encoding, and combination features.

**Critical:** Target encoding MUST be computed inside each CV fold. The script includes an example fold-safe implementation.

### 4. Temporal / DateTime Recipes

See `scripts/temporal_features.py` for cyclic encoding, lag features, rolling statistics, and time-since-reference features. **Sort by entity + time before computing lags.**

### 5. Aggregation Features (Group Stats)

See `scripts/temporal_features.py::make_group_aggs` for per-group mean/std/min/max/count. Always fit on train only, then map to val/test.

### 6. SHAP Importance & Feature Pruning

See `scripts/importance.py` for the full SHAP workflow (TreeExplainer, permutation importance) and the incremental pruning loop.

**Pruning rule:** Drop a feature if mean |SHAP| < 0.001 AND adding it does not improve OOF score by ≥ 0.0005.

## Quick Workflow

1. Read `references/safety_rules.md`.
2. Write candidate features using recipes from the `scripts/` files.
3. Run `scripts/importance.py` to compute SHAP values and rank features.
4. Run the pruning loop — keep only features that improve OOF by ≥ 0.0005.
5. Commit the feature set with a comment explaining each engineered feature's hypothesis.

## Resources

### references/
- `safety_rules.md` — leakage prevention, must-reads before any feature code

### scripts/
- `numerical_features.py` — log transforms, binning, outlier flags, interactions
- `categorical_features.py` — frequency/target/ordinal encoding, combo features
- `temporal_features.py` — datetime decomposition, lag/rolling features, group aggregations
- `importance.py` — SHAP tree explainer, permutation importance, incremental pruning loop
