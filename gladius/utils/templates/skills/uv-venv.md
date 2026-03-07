---
name: uv-venv
description: Install Python packages and run scripts using uv
---

This project uses **uv** for fast Python package management.
Never use `pip install` directly — always use `uv add` or `uv run`.

## Installing packages

```bash
# Add a runtime dependency (updates pyproject.toml + uv.lock)
uv add lightgbm scikit-learn pandas numpy

# Add a dev/optional dependency
uv add --dev pytest

# Sync the environment (install all locked deps)
uv sync
```

## Running scripts

```bash
# Run a script inside the venv without activating it
uv run python script.py

# Run with extra args
uv run python script.py --arg value

# Run a one-liner
uv run python -c "import numpy; print(numpy.__version__)"
```

## Checking installed packages

```bash
uv pip list
uv pip show lightgbm
```

## Notes
- `uv` is already installed. The venv is in `.venv/`.
- Prefer `uv add` over editing `pyproject.toml` manually.
- `uv.lock` should be committed to git for reproducibility.
- If a package needs a non-PyPI source (e.g. GPU build), use
  `uv add --index https://... package_name`.
