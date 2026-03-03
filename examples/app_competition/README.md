# Example: App Competition (No Metric)

This example shows how to use Gladius for an open-ended task where there is
no numeric leaderboard metric — the agent self-assesses quality (0–100).

---

## competition.yaml

```yaml
competition_id: build-a-todo-app
data_dir: ./data
output_dir: ./output
max_iterations: 5
# No metric / direction → open-ended mode
```

Notice: **no `metric`**, **no `direction`**, **no `platform`** fields.
Gladius detects open mode automatically and switches every agent to
quality-based self-assessment.

---

## task description (put this in the project's README.md)

```markdown
# Task: Build a CLI To-Do App

Build a command-line to-do application in Python that:

1. Stores tasks in a local SQLite database (`todos.db`).
2. Supports these commands:
   - `add <title>` — add a task
   - `list` — print all tasks with completion status
   - `done <id>` — mark task as complete
   - `delete <id>` — remove a task
3. Persists data across runs (re-running `list` shows previous tasks).
4. Has a simple README_submission.md explaining how to run it.

## Deliverable

Zip `todo_app.py`, `README_submission.md`, and any supporting files into
`todo_app.zip` and save it as the `submission_file`.
```

---

## How the agent loop works (open mode)

| Phase | What happens |
| --- | --- |
| Planning | Planner reads README, produces a concrete implementation plan |
| Implementing | Implementer builds + runs the app, self-rates quality 0–100 |
| Validation | Validator checks against README requirements, assigns final quality |
| Submission | `platform: none` → artifact path is logged locally, no upload |
| Improvement | Next iteration tries to raise the quality score by ≥ 2 points |

---

## Running it

```bash
gladius run --config examples/app_competition/competition.yaml
```
