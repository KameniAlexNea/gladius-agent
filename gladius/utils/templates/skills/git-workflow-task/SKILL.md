---
name: git-workflow
description: Commit every working iteration with a descriptive message
---

After implementing a deliverable that runs end-to-end without errors,
stage and commit it locally:

```bash
git add -A
git commit -m "iter-{N}: {change_summary} — quality {score}/100"
```

Guidelines:
- Commit only after the deliverable runs end-to-end without errors.
- Use `git add -A` to stage everything (source files, artifacts, run scripts).
- Message format: `iter-{N}: <one-sentence change> — quality {score}/100`
- Never force-push. Never rebase during a run.
- If a run fails, do NOT commit. Just proceed to the next iteration.
- Check `git status` before committing to avoid committing unintended files.
- `.gladius/` and `data/` should already be in `.gitignore` — verify if unsure.

Tip: use `git log --oneline -10` to review recent iteration history.
