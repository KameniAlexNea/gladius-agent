# HPO Best Practices

## When to Run HPO

- ✅ After a solid feature set is established (first build the kitchen, then tune the oven).
- ✅ The chosen architecture has proven competitive vs all alternatives you tried.
- ✅ You have iteration budget for 50–200 trials without worrying about wall-clock time.
- ❌ **Do NOT** run HPO on a weak feature set — you will overfit to noise.
- ❌ **Do NOT** tune multiple architectures in parallel — pick the winner first.

## Speed Tricks

| Trick | Impact |
| --- | --- |
| Use `N_FOLDS=3` during HPO | 1.67× faster without much noise |
| `MedianPruner(n_warmup_steps=2)` | Kills bad trials after only 2 folds |
| Set high `n_estimators` + use `early_stopping` | Automatically find the right depth |
| `storage="sqlite:///hpo.db"` + `load_if_exists=True` | Resume interrupted studies |
| Reduce dataset to 50k rows during HPO | Run full data only for final retraining |

## After HPO Completes

1. Copy `study.best_params` into your main training script.
2. Retrain with full `N_FOLDS=5` using best params.
3. Verify OOF improves vs the pre-HPO baseline — if not, HPO overfit the 3-fold CV.
4. Save the new OOF score and report it.

## Interpreting HPO Results

- If best trial barely improves vs default params: the model is already near-optimal; focus on features.
- If `learning_rate` converges near 0.3: try a lower range (0.01–0.1) — fast LR often indicates underfitting.
- If `n_estimators` maxes out: widen the range upward (e.g. 3000–5000) and re-run.

## Resuming an Interrupted Study

```python
study = optuna.load_study(study_name="lgbm_hpo", storage="sqlite:///hpo.db")
study.optimize(objective, n_trials=50)  # adds 50 more trials
```
