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
