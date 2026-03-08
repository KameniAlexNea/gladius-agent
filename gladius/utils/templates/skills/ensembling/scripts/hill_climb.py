"""
Greedy hill-climbing ensemble selection.

Greedily adds models when each addition improves OOF score.
Allows a model to be added multiple times (implicit weighting).
Most resistant to overfitting the validation set — use for large pools.

Usage:
    weights = hill_climb(oof_preds, y_train, roc_auc_score, n_rounds=200)
    final_tst = sum(test_preds[n] * w for n, w in weights.items())
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def hill_climb(
    oof_preds: dict,
    y_true: np.ndarray,
    metric_fn=roc_auc_score,
    n_rounds: int = 200,
    maximize: bool = True,
) -> dict[str, float]:
    names = list(oof_preds.keys())
    oof_arr = np.column_stack([oof_preds[n] for n in names])

    best_score = -np.inf if maximize else np.inf
    chosen: list[str] = []

    for _ in range(n_rounds):
        best_add = None
        for i, name in enumerate(names):
            indices = [names.index(c) for c in chosen] + [i]
            candidate = oof_arr[:, indices].mean(axis=1)
            s = metric_fn(y_true, candidate)
            if (maximize and s > best_score) or (not maximize and s < best_score):
                best_score = s
                best_add = name
        if best_add is None:
            break
        chosen.append(best_add)

    counts = {n: chosen.count(n) for n in set(chosen)}
    total = sum(counts.values())
    weights = {n: c / total for n, c in counts.items()}

    print(
        f"Hill-climb OOF: {best_score:.6f}  ({len(chosen)} additions, {len(counts)} unique models)"
    )
    for n, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {n}: {w:.4f}")

    return weights
