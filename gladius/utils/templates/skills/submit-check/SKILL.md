---
name: submit-check
description: Validate a submission CSV before platform upload
disable-model-invocation: true
---

Validate the submission file at $ARGUMENTS against the sample submission.

Steps:
1. Load `data/sample_submission.csv` (or find sample_submission.csv in the data dir)
2. Load the submission at `$ARGUMENTS`
3. Check that column names match exactly (same order and names)
4. Check that row counts match
5. Check for NaN or Inf in all numeric columns
6. For probability predictions: check all values are in [0, 1]
7. For regression: check values are in a sane range (warn if extreme outliers)

Output:
- `VALID` if all checks pass
- `INVALID: <reason>` listing specific issues
