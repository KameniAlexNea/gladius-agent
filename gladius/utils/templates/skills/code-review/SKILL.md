---
name: code-review
description: >
  Review code and deliverables before finalising results. For ML competitions:
  catches data leakage, metric errors, and submission format bugs. For
  open-ended tasks: verifies functional correctness, completeness, packaging,
  and assigns a quality score 0–100. Always invoke before reporting results.
---

# Code Review & Quality Check

Fix every CRITICAL item before reporting results.

---

## ML Competition Review

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

### Style

- [ ] Each feature engineering step has a comment explaining the hypothesis.
- [ ] File name is descriptive: `solution_lgbm_v2.py`, not `solution.py`.

---

## Open-Ended Task Review

### CRITICAL — functional correctness

- [ ] Run the deliverable end-to-end (`uv run python app.py` or `./run.sh`) — no crashes.
- [ ] Every feature listed in `README.md` is implemented and reachable.
- [ ] Edge cases handled: empty input, invalid input, missing files.
- [ ] No hardcoded paths — all paths are relative or configurable.

### CRITICAL — completeness

- [ ] All required files are present (app, config, dependencies, README).
- [ ] Dependencies declared in `pyproject.toml` (not just installed ad-hoc).
- [ ] Deliverable reproducible from scratch: `uv sync && uv run ...`.

### CRITICAL — packaging

- [ ] Submission artifact exists at the path reported in `submission_file`.
- [ ] Artifact contains everything needed to run or evaluate the deliverable.
- [ ] `README.md` explains how to install and run.

### Important — robustness

- [ ] No unhandled exceptions on the happy path.
- [ ] Environment variables or config files used for secrets — not hardcoded.
- [ ] Script/app runs without user interaction (unless README explicitly requires it).

### Quality Score (0–100)

| Score | Meaning |
| --- | --- |
| 90–100 | Exceeds requirements; polished, documented, tested |
| 70–89 | Meets all stated requirements; no major gaps |
| 50–69 | Meets most requirements; some gaps or rough edges |
| 30–49 | Partial implementation; core functionality works |
| 0–29 | Incomplete; significant requirements unmet |

Read `README.md`, run through the checklist, assign a score, and write 1–2 sentences justifying it in the `notes` output field.

### Packaging the Deliverable

```bash
zip -r deliverable.zip output/ app.py pyproject.toml README.md
# or record a URL / binary path:
echo "https://..." > submission_url.txt
```

Report `submission_file` as the path to the zip / binary / URL file.
