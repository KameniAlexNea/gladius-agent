"""
Gladius CLI — set up a competition project and launch the agent.

  gladius CONFIG [--max-turns N]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from gladius.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
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
        logger.error(f"config file not found: {args.config}")
        sys.exit(1)

    try:
        from gladius.project_setup import load_config, setup

        cfg = load_config(config_path)
        configure_logging(cfg["project_dir"])
    except Exception as exc:
        logger.exception(f"error loading config: {exc}")
        sys.exit(1)

    try:
        setup(config_path)
    except Exception as exc:
        logger.exception(f"error during setup: {exc}")
        sys.exit(1)

    try:
        from gladius.orchestrator import run_competition

        asyncio.run(
            run_competition(
                competition_dir=cfg["project_dir"],
                max_turns=args.max_turns,
                max_iterations=cfg.get("max_iterations") or None,
                config_path=str(config_path),
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        logger.exception(f"error: {exc}")
        sys.exit(1)
