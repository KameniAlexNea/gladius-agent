---
name: ml-project-structure
description: >
  Canonical ML competition project layout. Invoke before writing the first
  solution file to set up a structured, reusable codebase instead of a
  single monolithic script.
---

## Guiding principle

**Every competition solution is a mini-package.**  
Break code into small, focused modules so each can be tested, reused, and
replaced independently. Never put everything in one script.

## Standard project layout

```
<competition-root>/
├── pyproject.toml          # managed by uv (uv init if absent)
├── uv.lock
├── .venv/
│
├── src/                    # all importable project code lives here
│   ├── config.py           # paths, seeds, CV folds, constants
│   ├── data.py             # loading, splitting, CV iterator
│   ├── features.py         # feature engineering functions
│   ├── models.py           # model factory / training / predict helpers
│   ├── metrics.py          # competition metric functions
│   ├── postprocess.py      # clipping, blending, rank transforms
│   └── submission.py       # build + validate submission CSV
│
├── notebooks/              # exploratory work only (not imported)
│   └── eda.ipynb
│
├── scripts/                # thin CLI entry-points that import from src/
│   ├── train.py            # full CV run → saves OOF + test preds
│   ├── tune.py             # Optuna HPO → best params to config
│   └── predict.py          # load artifacts → write submission CSV
│
├── artifacts/              # saved models, OOF arrays, encoder pickles
│   └── .gitkeep
│
└── submissions/            # submission CSVs ready to upload
    └── .gitkeep
```

## Module responsibilities

### `src/config.py`
```python
from pathlib import Path

DATA_DIR   = Path("../data")        # adjust to actual data path
OUTPUT_DIR = Path("submissions")
ARTIFACTS  = Path("artifacts")
SEED       = 42
N_FOLDS    = 5
TARGET     = "target"               # actual target column name
```

### `src/data.py`
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

### `src/features.py`
```python
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with engineered features appended. Never mutates in-place."""
    df = df.copy()
    # add features here — document each one with a comment
    return df
```

### `src/models.py`
```python
import lightgbm as lgb
import numpy as np
from src.config import SEED

def make_lgbm(params: dict | None = None) -> lgb.LGBMClassifier:
    defaults = dict(n_estimators=1000, learning_rate=0.05,
                    num_leaves=31, random_state=SEED, n_jobs=-1)
    return lgb.LGBMClassifier(**(defaults | (params or {})))

def fit_predict(model, X_train, y_train, X_val, X_test):
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val := None)],  # replace y_val
              callbacks=[lgb.early_stopping(50, verbose=False)])
    oof  = model.predict_proba(X_val)[:, 1]
    test = model.predict_proba(X_test)[:, 1]
    return oof, test
```

### `src/metrics.py`
```python
import numpy as np

def competition_metric(y_true, y_pred) -> float:
    """Replace with the actual competition metric."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)
```

### `scripts/train.py` (thin entry-point)
```python
"""Full CV training run."""
import numpy as np
import pandas as pd
from src.config import ARTIFACTS, N_FOLDS, TARGET
from src.data import load_train, load_test, get_cv
from src.features import add_features
from src.metrics import competition_metric
from src.models import make_lgbm, fit_predict
from src.submission import make_submission

train = add_features(load_train())
test  = add_features(load_test())

X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test[X.columns]

oof  = np.zeros(len(train))
preds = np.zeros(len(test))

for fold, (tr, va) in enumerate(get_cv().split(X, y)):
    model = make_lgbm()
    oof[va], fold_preds = fit_predict(model, X.iloc[tr], y.iloc[tr],
                                      X.iloc[va], X_test)
    preds += fold_preds / N_FOLDS

score = competition_metric(y, oof)
print(f"OOF score: {score:.6f}")

np.save(ARTIFACTS / "oof.npy", oof)
np.save(ARTIFACTS / "test_preds.npy", preds)
make_submission(test, preds)
```

## Rules

1. **No logic in scripts** — scripts are 20–40 lines, all logic lives in `src/`.
2. **Functions, not notebooks** — notebooks are for exploration only; never import from them.
3. **No global state mutations** — functions return new objects, never modify inputs.
4. **One responsibility per module** — features.py never trains models.
5. **Everything reproducible** — fix all random seeds via `src/config.SEED`.
6. **Artifacts saved, not recomputed** — save OOF arrays and models in `artifacts/`
   so you can blend without re-training.

## Initialising the project

```bash
# If pyproject.toml doesn't exist yet
uv init --name solution --python 3.11

# Create and activate venv
uv venv && source .venv/bin/activate

# Install core deps
uv add lightgbm scikit-learn pandas numpy

# Make src importable without install
# (set pythonpath or add to pyproject.toml)
export PYTHONPATH="$PWD"
```

Add to `pyproject.toml` so scripts can import `src` without `PYTHONPATH`:
```toml
[tool.setuptools]
packages = ["src"]
```
Or simply run all scripts from the project root where `src/` is visible.
