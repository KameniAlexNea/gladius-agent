# ML Project Layout — Module Responsibilities & Templates

## Directory Tree

```
<competition-root>/
├── CLAUDE.md               ← auto-generated, do not edit
├── pyproject.toml          ← managed by uv (uv init if absent)
├── uv.lock
├── .venv/
│
├── src/                    ← all importable project code lives here
│   ├── __init__.py         ← empty, makes src a package
│   ├── config.py           ← paths, seeds, CV folds, constants
│   ├── data.py             ← loading, splitting, CV iterator
│   ├── features.py         ← feature engineering functions
│   ├── models.py           ← model factory / training / predict helpers
│   ├── metrics.py          ← competition metric functions
│   ├── submission.py       ← build + validate submission CSV
│   └── (postprocess.py)    ← optional: clipping, blending, rank transforms
│
├── scripts/                ← thin CLI entry-points that import from src/
│   ├── train.py            ← full CV run → saves OOF + test preds
│   ├── tune.py             ← (optional) Optuna HPO → best params to config
│   └── predict.py          ← (optional) load artifacts → write submission CSV
│
├── notebooks/              ← exploratory work only (never imported by scripts)
│   └── eda.ipynb
│
├── artifacts/              ← saved models, OOF arrays, encoder pickles
│   └── .gitkeep
│
└── submissions/            ← submission CSVs ready to upload
    └── .gitkeep
```

---

## `src/config.py`

Central source of truth for all paths and hyperparameters.

```python
from pathlib import Path

DATA_DIR    = Path("data")          # adjust to actual data_dir from CLAUDE.md
OUTPUT_DIR  = Path("submissions")
ARTIFACTS   = Path("artifacts")
SEED        = 42
N_FOLDS     = 5
TARGET      = "target"              # replace with actual target column name
```

---

## `src/data.py`

All file I/O and CV splitting.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from src.config import DATA_DIR, N_FOLDS, SEED, TARGET


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv")


def get_cv(stratified: bool = True):
    if stratified:
        return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
```

---

## `src/features.py`

Returns a new DataFrame with added features. **Never mutates the input.**

```python
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return df with engineered features appended.
    Never mutates the input — always work on a copy.
    Document each feature with a comment explaining the hypothesis.
    """
    df = df.copy()
    # Example: log transform of skewed numeric feature
    # df["price_log"] = np.log1p(df["price"])   # hypothesis: prices are log-normal
    return df
```

---

## `src/models.py`

Model factory and a `fit_predict()` helper that encapsulates the training loop.

```python
import lightgbm as lgb
import numpy as np
from src.config import SEED


def make_lgbm(params: dict | None = None) -> lgb.LGBMClassifier:
    defaults = dict(
        n_estimators=1000, learning_rate=0.05,
        num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    return lgb.LGBMClassifier(**(defaults | (params or {})))


def fit_predict(model, X_train, y_train, X_val, y_val, X_test):
    """Train model and return (oof_predictions, test_predictions)."""
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof  = model.predict_proba(X_val)[:, 1]    # adjust for regression / multiclass
    test = model.predict_proba(X_test)[:, 1]
    return oof, test
```

---

## `src/metrics.py`

Wraps the competition metric so it can be swapped without touching training code.

```python
import numpy as np


def competition_metric(y_true, y_pred) -> float:
    """Replace with the actual competition metric from CLAUDE.md."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)
```

---

## `src/submission.py`

Builds and validates the submission CSV.

```python
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, OUTPUT_DIR


def make_submission(test: pd.DataFrame, preds, filename: str = "submission.csv") -> str:
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = sample_sub.copy()
    sub[sub.columns[-1]] = preds
    assert len(sub) == len(sample_sub), f"Row count mismatch: {len(sub)} vs {len(sample_sub)}"
    assert not sub.isnull().any().any(), "NaN in submission!"
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = str(OUTPUT_DIR / filename)
    sub.to_csv(path, index=False)
    print(f"Submission saved: {path}  ({len(sub)} rows)")
    return path
```

---

## `scripts/train.py` (thin entry-point)

20–40 lines. All logic in `src/`.

```python
"""Full CV training run. Run from competition root: uv run python scripts/train.py"""
import numpy as np
from src.config import ARTIFACTS, N_FOLDS, TARGET
from src.data import load_train, load_test, get_cv
from src.features import add_features
from src.metrics import competition_metric
from src.models import make_lgbm, fit_predict
from src.submission import make_submission

train = add_features(load_train())
test  = add_features(load_test())

X      = train.drop(columns=[TARGET])
y      = train[TARGET]
X_test = test[[c for c in X.columns if c in test.columns]]

oof   = np.zeros(len(train))
preds = np.zeros(len(test))

for fold, (tr, va) in enumerate(get_cv().split(X, y)):
    m_oof, m_test = fit_predict(make_lgbm(), X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X_test)
    oof[va] = m_oof
    preds  += m_test / N_FOLDS
    print(f"  Fold {fold+1}: {competition_metric(y.iloc[va], m_oof):.6f}")

score = competition_metric(y, oof)
print(f"OOF score: {score:.6f}")

ARTIFACTS.mkdir(exist_ok=True)
np.save(ARTIFACTS / "oof.npy", oof)
np.save(ARTIFACTS / "test_preds.npy", preds)
make_submission(test, preds, f"submission_lgbm_v1.csv")
```
