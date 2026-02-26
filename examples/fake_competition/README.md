# Fake Competition — local end-to-end test

A self-contained binary-classification problem you can use to test Gladius
without a real Kaggle or Zindi account.

## Setup

```bash
# 1. Install dependencies (from the repo root)
pip install -e ".[dev]"

# 2. Generate data
cd examples/fake_competition
python generate_data.py
```

This writes four files inside `data/`:

| File | Description |
|---|---|
| `train.csv` | 800 rows · 20 features · `target` column |
| `test.csv` | 200 rows · 20 features · no target |
| `sample_submission.csv` | Trivial 0.5 baseline (AUC ≈ 0.50) |
| `.answers.csv` | Hidden test labels used by the fake scorer |

## Run the agent

```bash
# From the repo root
gladius \
  --competition fake_binary \
  --platform fake \
  --data-dir examples/fake_competition/data \
  --project-dir examples/fake_competition \
  --metric auc_roc \
  --direction maximize \
  --iterations 5

# Or without installing the package:
python -m gladius.orchestrator \
  --competition fake_binary \
  --platform fake \
  --data-dir examples/fake_competition/data \
  --project-dir examples/fake_competition \
  --metric auc_roc \
  --direction maximize \
  --iterations 5
```

The fake platform (`platform=fake`) behaves like Kaggle/Zindi but:

- **Submission** → scores the CSV against `.answers.csv` using AUC-ROC and records the result locally in `.fake_platform/history.json`. No internet required.
- **Leaderboard** → returns a seeded fake leaderboard plus your best score.
- **Status** → always shows unlimited remaining submissions.

## Environment variables

The fake platform reads:

| Variable | Default | Description |
|---|---|---|
| `FAKE_ANSWERS_PATH` | `data/.answers.csv` | Path to hidden answer key |
| `FAKE_PLATFORM_DIR` | `.fake_platform/` | Where history is stored |

## What to expect

| Model | Expected AUC |
|---|---|
| Trivial 0.5 baseline | ≈ 0.50 |
| Logistic Regression | ≈ 0.87 |
| Gradient Boosting | ≈ 0.92 |

The agent should iterate from a weak baseline up toward the XGBoost/LightGBM range.
