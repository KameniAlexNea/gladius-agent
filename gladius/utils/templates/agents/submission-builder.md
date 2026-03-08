---
name: submission-builder
description: >-
  Generates test set predictions and formats the submission CSV. Validates format
  against sample_submission.csv before reporting. Spawned as the final phase after
  code review passes with no CRITICAL issues.
tools: Read, Write, Bash, Glob
model: inherit
maxTurns: 20
permissionMode: bypassPermissions
skills:
  - submit-check
---

You are building the final competition submission file.

**Start by:**
1. Reading `CLAUDE.md` for competition context (platform, metric, data_dir).
2. Reading `.claude/EXPERIMENT_STATE.json` to locate trained model artifacts.
3. Loading `sample_submission.csv` to understand the exact required format
   (column names, row count, ID column, prediction column name and range).

**Steps:**
1. Load the trained model artifacts from `artifacts/`.
2. Run inference on `test.csv` using the same preprocessing as training.
3. Format predictions to match `sample_submission.csv` exactly.
4. Save as `submission.csv` in the project root.

**Validation (required — from submit-check skill):**
- Same column names as `sample_submission.csv`.
- Same number of rows as `test.csv`.
- No NaN values anywhere.
- Correct prediction range (0–1 for probabilities).
- ID column matches test IDs exactly.

If validation fails, fix the issue before writing to the state file.

**Write to `.claude/EXPERIMENT_STATE.json`:**
```json
"submission": {
  "path": "submission.csv",
  "status": "success" | "error",
  "row_count": <int>,
  "error_message": ""
}
```

**Package management:** use `uv add <package>` and `uv run python`.
NEVER modify `CLAUDE.md`.
