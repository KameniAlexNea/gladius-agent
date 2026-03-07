---
name: code-review
description: Review deliverable code before finalising — catch crashes, missing features, and packaging issues
---

Before finalising any deliverable, review it against this checklist.
Fix every item marked CRITICAL before reporting results.

## CRITICAL — functional correctness

- [ ] Run the deliverable end-to-end: `uv run python app.py` or `./run.sh` — no crashes.
- [ ] Every feature listed in `README.md` is implemented and reachable.
- [ ] Edge cases handled: empty input, invalid input, missing files.
- [ ] No hardcoded paths — all paths are relative or configurable.

## CRITICAL — completeness

- [ ] All required files are present (app, config, dependencies, README).
- [ ] Dependencies declared in `pyproject.toml` (not just installed ad-hoc).
- [ ] The deliverable can be reproduced from scratch by a fresh `uv sync && uv run ...`.

## CRITICAL — packaging

- [ ] Submission artifact exists at the path reported in `submission_file`.
- [ ] Artifact contains everything needed to run or evaluate the deliverable.
- [ ] `README.md` (or equivalent) explains how to install and run.

## Important — robustness

- [ ] No unhandled exceptions on the happy path.
- [ ] Environment variables or config files used for secrets — not hardcoded.
- [ ] Script/app runs without user interaction (unless README explicitly requires it).

## Style

- [ ] Code is readable: functions have docstrings, logic is commented where non-obvious.
- [ ] File names are descriptive and consistent.
