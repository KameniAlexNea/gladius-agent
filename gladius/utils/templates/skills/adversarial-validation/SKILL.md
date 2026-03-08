---
name: adversarial-validation
description: >
  Detect distribution shift between train and test sets using a classifier.
  Run at the start of every competition and whenever OOF score diverges from
  the leaderboard. Identifies leaking features and provides sample weights to
  compensate for structural shift.
---

# Adversarial Validation

## Overview

Train a binary classifier to distinguish train rows from test rows. If AUC > 0.55, the distributions differ — features that drive the classifier are likely to hurt your CV-LB correlation. Use sample weighting or feature removal to correct shift before investing in HPO.

## When to Use This Skill

- Start of every competition — establish baseline train/test similarity.
- OOF is high but leaderboard score lags (LB-OOF gap > 0.01).
- After adding a batch of new features — verify they don't introduce shift.
- Before computing lag/aggregation features — confirm time splits are safe.

## Core Capabilities

### 1. Run the Adversarial Classifier

Full working script — see `scripts/run.py`.

**Quick inline version:**
```python
from adversarial-validation.scripts.run import run_adversarial_validation
auc, top_feats = run_adversarial_validation("data", TARGET_COL)
```

### 2. Interpret the AUC

| AUC | Verdict | Action |
| --- | --- | --- |
| 0.50–0.55 | ✅ No shift | Proceed normally |
| 0.55–0.65 | ⚠️ Mild shift | Check top features; monitor LB-OOF gap |
| 0.65–0.80 | ❌ Moderate shift | Drop or transform top leaking features |
| 0.80–1.00 | 🚨 Severe shift | Likely ID/time leak — investigate immediately |

### 3. Identify Leaking Features

The adversarial classifier's feature importances reveal which features differ most between train and test. Common culprits:
- **Row ID / index columns** — always unique across splits; drop them.
- **Timestamp columns** — train is older, test is newer; engineer carefully.
- **Count/aggregation features** computed on different time windows in train vs test.

### 4. Adversarial Sample Weighting

When shift is structural and unavoidable, weight training samples by `P(is_test | features)` so your model emphasises train examples that "look like" test. See `scripts/run.py` for the weighting code.

## Quick Workflow

1. Invoke this skill at the start of every competition run.
2. Run `scripts/run.py` — it prints the AUC and top-20 leaking features.
3. If AUC > 0.55, drop the top features listed and re-run until AUC < 0.60.
4. If dropping hurts CV too much, apply adversarial sample weights instead.

## Resources

### scripts/
- `run.py` — complete adversarial validation + feature importance + sample weights
