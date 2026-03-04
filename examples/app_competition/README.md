---
competition_id: build-a-todo-app
platform: none
---

# Task: Build a CLI To-Do App

Build a command-line to-do application in Python that:

1. Stores tasks in a local SQLite database (`todos.db`).
2. Supports these commands:
   - `add <title>` — add a new task
   - `list` — print all tasks with ID, title, and completion status
   - `done <id>` — mark a task as complete
   - `delete <id>` — remove a task permanently
3. Persists data across runs (`todos.db` survives process restarts).
4. Handles invalid input gracefully (unknown command, bad ID, etc.).

## Deliverable

Package `todo_app.py` and a brief `README_submission.md` (explaining how to
install and run) into `todo_app.zip`. Save the zip as the `submission_file`.

## Self-assessment criteria (quality 0–100)

| Score | Meaning |
| --- | --- |
| 90–100 | All 4 commands work, persists correctly, error handling present, documented |
| 70–89 | All commands work, persists, minor issues |
| 50–69 | Most commands work, some edge cases fail |
| < 50  | Core functionality missing or crashes |
