# Competition Metrics Reference

## Metric Table

| Competition type | Metric name | sklearn call | Direction |
| --- | --- | --- | --- |
| Binary classification | AUC-ROC | `roc_auc_score(y_true, y_score)` | maximize |
| Binary classification | Log loss | `log_loss(y_true, y_proba)` | minimize |
| Binary classification | F1 (binary) | `f1_score(y_true, y_pred, average='binary')` | maximize |
| Multiclass classification | Log loss | `log_loss(y_true, y_proba_matrix)` | minimize |
| Multiclass classification | F1 macro | `f1_score(y_true, y_pred, average='macro')` | maximize |
| Multiclass classification | Weighted F1 | `f1_score(y_true, y_pred, average='weighted')` | maximize |
| Multiclass classification | Accuracy | `accuracy_score(y_true, y_pred)` | maximize |
| Regression | RMSE | `mean_squared_error(y_true, y_pred, squared=False)` | minimize |
| Regression | MAE | `mean_absolute_error(y_true, y_pred)` | minimize |
| Regression | RMSLE | `np.sqrt(mean_squared_log_error(y_true, y_pred))` | minimize |
| Regression | R² | `r2_score(y_true, y_pred)` | maximize |
| Ranking | MAP@K | — see below | maximize |
| Ranking | NDCG | `ndcg_score(y_true, y_score)` | maximize |

## Common Mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Using train predictions instead of OOF | OOF = 1.0 or near 1.0 | Ensure OOF array is filled only from validation folds |
| Wrong `average=` for F1 | Score inconsistent with leaderboard | Check competition description: binary vs macro vs weighted |
| Predicting class labels for AUC | `ValueError` from sklearn | Use `predict_proba()[:, 1]`, not `predict()` |
| RMSLE with negative predictions | `ValueError` | Clip preds to ≥ 0 before RMSLE |
| Computing metric per fold then averaging | Inflated score | Concatenate all OOF preds first, then compute once |

## Logging Convention

Always print OOF score in this format so it appears searchable in logs:

```python
print(f"OOF {metric_name}: {score:.6f}")
```

## MAP@K Implementation

```python
def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
```
