"""
Gladius CLI — set up a competition project and launch the agent.

  gladius CONFIG [--max-turns N]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gladius",
        description="Self-directed ML competition agent.",
        epilog="Example: gladius examples/project.yaml",
    )
    parser.add_argument("config", metavar="CONFIG", help="Project config YAML file.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help="Hard cap on agent turns (default: unlimited).",
    )

    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        from gladius.project_setup import load_config, setup

        cfg = load_config(config_path)
    except Exception as exc:
        print(f"error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        setup(config_path)
    except Exception as exc:
        print(f"error during setup: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        from gladius.orchestrator import run_competition

        asyncio.run(
            run_competition(
                competition_dir=cfg["project_dir"],
                max_turns=args.max_turns,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
