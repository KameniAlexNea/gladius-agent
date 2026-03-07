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
