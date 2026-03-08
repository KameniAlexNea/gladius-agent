#!/usr/bin/env bash
# init.sh — Initialise the competition project structure.
#
# MUST be run from the competition root (the directory containing CLAUDE.md).
# Usage: bash scripts/init.sh

set -e

if [ ! -f "CLAUDE.md" ]; then
  echo "ERROR: CLAUDE.md not found in current directory."
  echo "You must run this script from the competition root, not from a subdirectory."
  echo "Current directory: $(pwd)"
  exit 1
fi

echo "Initialising project in: $(pwd)"

# ── Project directories ────────────────────────────────────────────────────────
mkdir -p src scripts notebooks artifacts submissions

# ── src package ───────────────────────────────────────────────────────────────
touch src/__init__.py
touch artifacts/.gitkeep
touch submissions/.gitkeep

# ── pyproject.toml (if it doesn't exist yet) ──────────────────────────────────
if [ ! -f "pyproject.toml" ]; then
  uv init --name solution --python 3.11 --no-package
  echo "Created pyproject.toml"
fi

# ── Ensure src is importable without PYTHONPATH ──────────────────────────────
# Add src to pyproject.toml packages if not already there
if ! grep -q "packages" pyproject.toml 2>/dev/null; then
  cat >> pyproject.toml << 'TOML'

[tool.setuptools]
packages = ["src"]
TOML
  echo "Added src package to pyproject.toml"
fi

# ── Create and activate venv ──────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  uv venv
  echo "Created .venv"
fi

echo ""
echo "Project structure initialised successfully."
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. Edit src/config.py (set DATA_DIR, TARGET)"
echo "  3. uv add lightgbm scikit-learn pandas numpy"
echo "  4. uv run python scripts/train.py"
