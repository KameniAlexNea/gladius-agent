---
name: git-workflow
description: Commit every working iteration with a descriptive message
---

After implementing a solution that runs without errors and produces an OOF score,
stage and commit it locally:

```bash
git add -A
git commit -m "iter-{N}: {approach_summary} — OOF {metric}={score:.6f}"
```

Guidelines:
- Commit only after the deliverable runs end-to-end without errors.
- Use `git add -A` to stage everything (source files, artifacts, run scripts).
- Message format: `iter-{N}: <one-sentence approach> — OOF {metric}={score:.6f}`
- Never force-push. Never rebase during a run.
- If a run fails, do NOT commit. Just proceed to the next iteration.
- Check `git status` before committing to avoid committing unintended files.
- `.gladius/` and `data/` should already be in `.gitignore` — verify if unsure.

Tip: use `git log --oneline -10` to review recent iteration history.
