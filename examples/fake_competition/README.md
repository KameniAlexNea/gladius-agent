---
competition_id: fake_binary
platform: fake
metric: auc_roc
direction: maximize
data_dir: data
---

# Customer Churn Prediction Challenge

## Overview

A telecom operator wants to identify customers likely to cancel their subscription
in the next 30 days so that the retention team can intervene proactively.
Your task is to build a binary classifier that predicts churn probability for
each customer in the test set.

## Task

Binary classification — predict the probability that a customer will churn.

## Evaluation

Submissions are scored using **Area Under the ROC Curve (AUC-ROC)**.
Higher is better. A random predictor scores ≈ 0.50.

The submission file must contain exactly two columns:

```
customer_id,target
12345,0.82
12346,0.14
...
```

## Data

| File | Description |
|---|---|
| `train.csv` | 800 customers with known churn outcome (`target` = 0 or 1) |
| `test.csv` | 200 customers — predict churn probability for each |
| `sample_submission.csv` | Correct format with a 0.5 baseline |

### Features

20 anonymised numerical features (`feature_0` … `feature_19`) derived from:
- Usage patterns (call volume, data consumption, support contacts)
- Account tenure and contract type
- Billing history (late payments, plan changes)

The target column is `target` (1 = churned, 0 = retained).
Class balance: approximately 30 % positive (churned).

## Notes

- No missing values in this dataset.
- Features are on different scales — normalisation may help linear models.
- Some features are correlated; feature selection or regularisation is advised.
- Tree-based models (LightGBM, XGBoost) typically outperform linear baselines here.
