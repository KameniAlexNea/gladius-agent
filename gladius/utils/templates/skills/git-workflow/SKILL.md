---
name: git-workflow
description: Commit every working iteration with a descriptive message
---

After implementing a solution or deliverable that runs end-to-end without errors,
stage and commit it locally.

## ML competition

```bash
git add -A
git commit -m "iter-{N}: {approach_summary} — OOF {metric}={score:.6f}"
```

## Open-ended task

```bash
git add -A
git commit -m "iter-{N}: {change_summary} — quality {score}/100"
```

## Guidelines

- Commit only after the deliverable runs end-to-end without errors.
- Use `git add -A` to stage everything (source files, artifacts, run scripts).
- Never force-push. Never rebase during a run.
- If a run fails, do NOT commit — proceed to the next iteration.
- Check `git status` before committing to avoid staging unintended files.
- `.gladius/` and `data/` should already be in `.gitignore` — verify if unsure.

Tip: `git log --oneline -10` to review recent iteration history.
