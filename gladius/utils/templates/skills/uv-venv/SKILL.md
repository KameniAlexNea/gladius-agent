---
name: uv-venv
description: Create a virtualenv, install Python packages, and run scripts using uv
---

This project uses **uv** for fast Python package management.
Never use `pip install` directly — always use `uv add` or `uv run`.

## 1. Create and activate the virtual environment (do this first)

```bash
# Create the venv in .venv/ (safe to re-run — no-op if it already exists)
uv venv

# Activate it for the current shell session
source .venv/bin/activate
```

> Always activate the venv before running any script or installing packages.
> You will see `(.venv)` in your prompt when activated.

## 2. Installing packages

```bash
# Add a runtime dependency (updates pyproject.toml + uv.lock)
uv add lightgbm scikit-learn pandas numpy

# Add a dev/optional dependency
uv add --dev pytest

# Sync the environment (install all locked deps, e.g. after git pull)
uv sync
```

## 3. Running scripts

```bash
# Run a script inside the activated venv
python script.py

# Or, without activating, use uv run
uv run python script.py --arg value

# Run a one-liner
python -c "import numpy; print(numpy.__version__)"
```

## Checking installed packages

```bash
uv pip list
uv pip show lightgbm
```

## Notes
- `uv` is already installed globally. The venv lives in `.venv/`.
- Prefer `uv add` over editing `pyproject.toml` manually.
- `uv.lock` should be committed to git for reproducibility.
- If a package needs a non-PyPI source (e.g. GPU build), use
  `uv add --index https://... package_name`.
