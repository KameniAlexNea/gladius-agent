---
name: validation
description: >
  Comprehensive validation before reporting results. Covers three areas:
  (1) code quality — data leakage, CV contamination, metric formula errors;
  (2) submission format — column names, row count, NaN/Inf, value ranges;
  (3) distribution shift — adversarial train/test AUC, leaking feature removal,
  adversarial sample weighting. Always invoke before finalising any iteration.
---

# Validation

## Overview

Three checks in one skill. Run in order before reporting any iteration result.
Catches the most common sources of wasted leaderboard submissions: data leakage,
CV metric errors, submission format bugs, and distribution shift.

## When to Use This Skill

- Before reporting any OOF score or submitting predictions.
- Whenever OOF looks suspiciously high or the LB-OOF gap widens.
- After writing any new feature engineering code.
- At competition start — run adversarial AUC baseline once.

## Core Capabilities

1. **Code & ML Quality Review** — data leakage, CV contamination, metric formula errors, robustness. Full checklist in `references/checklist.md`.
2. **Submission File Validation** — column names, row count, NaN/Inf, value ranges vs `sample_submission.csv`.
3. **Adversarial Validation** — train/test distribution shift via binary classifier: AUC, leaking feature ranking, sample weights. Standalone script: `scripts/adversarial_validation.py`.

---

## 1. Code & ML Quality Review

Fix every CRITICAL item before reporting results.

### CRITICAL — data leakage

- [ ] Target-based encodings (mean encoding, target encoding) computed **inside** each CV fold — never on full train.
- [ ] Temporal features (lag, rolling stats) use only past data — no future leakage.
- [ ] No test rows accidentally appear in any training fold.
- [ ] StandardScaler / other transformers fit on train fold only, applied to val/test.
- [ ] `train_test_split` is NOT used instead of proper k-fold CV.

### CRITICAL — metric correctness

- [ ] OOF metric computed on the full OOF array, not fold-by-fold averages.
- [ ] Metric function matches the competition definition exactly (`average='macro'` vs `'binary'` etc.).
- [ ] For probability metrics (AUC, log-loss): predictions are probabilities, not class labels.
- [ ] Metric direction (maximize/minimize) respected when comparing scores.

### CRITICAL — submission format

- [ ] Submission CSV column names match `sample_submission.csv` exactly.
- [ ] Submission row count matches `sample_submission.csv` exactly.
- [ ] No NaN or Inf in prediction column.
- [ ] File saved to the path reported in `submission_file`.

### Important — robustness

- [ ] No hard-coded file paths — use variables from CLAUDE.md context.
- [ ] Random seeds set for reproducibility (`random_state=42`, `np.random.seed(42)`).
- [ ] OOF score printed as `OOF {metric}: {score:.6f}` so it appears in logs.
- [ ] Script runs end-to-end without manual intervention.

---

## 2. Submission File Validation

Validate the submission CSV before platform upload.

Steps:
1. Load `data/sample_submission.csv` (or find sample_submission.csv in the data dir).
2. Load the submission at `artifacts/submission.csv`.
3. Check that column names match exactly (same order and names).
4. Check that row counts match.
5. Check for NaN or Inf in all numeric columns.
6. For probability predictions: check all values are in [0, 1].
7. For regression: check values are in a sane range (warn if extreme outliers).

Output:
- `VALID` if all checks pass.
- `INVALID: <reason>` listing specific issues.

---

## 3. Adversarial Validation (Distribution Shift)

Run the standalone script — it handles data loading, CV, AUC reporting, top feature ranking, and sample weight generation:

```bash
uv run python scripts/adversarial_validation.py --data data/ --target <target_col>
# Outputs: AUC, top-20 leaking features, artifacts/adversarial_weights.npy
```

Or import the function directly:

```python
from scripts.adversarial_validation import run_adversarial_validation
result = run_adversarial_validation("data/", target_col="target")
# result = {"auc": 0.58, "verdict": "...", "top_features": {...}}
```

### When to run

- Start of every competition — establish baseline train/test similarity.
- OOF is high but leaderboard score lags (LB-OOF gap > 0.01).
- After adding a batch of new features — verify they don't introduce shift.

### AUC interpretation

| AUC | Verdict | Action |
| --- | --- | --- |
| 0.50–0.55 | ✅ No shift | Proceed normally |
| 0.55–0.65 | ⚠️ Mild shift | Check top features; monitor LB-OOF gap |
| 0.65–0.80 | ❌ Moderate shift | Drop or transform top leaking features |
| 0.80–1.00 | 🚨 Severe shift | Likely ID/time leak — investigate immediately |

### Common leaking feature culprits

- **Row ID / index columns** — always unique across splits; drop them.
- **Timestamp columns** — train is older, test is newer; engineer carefully.
- **Count/aggregation features** computed on different time windows in train vs test.

### Using adversarial sample weights

When shift is structural and unavoidable, pass the saved weights to your model:

```python
weights = np.load("artifacts/adversarial_weights.npy")
model.fit(X_train, y_train, sample_weight=weights)
```

---

## Quick workflow (each iteration)

1. Run **Section 1** (code review checklist) on any newly written code.
2. Run **Section 2** (submission file check) before every submission.
3. Run **Section 3** (adversarial AUC) once at competition start; re-run when LB-OOF gap widens.
