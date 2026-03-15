"""
Gladius CLI — single command that sets up and runs the competition agent.

  gladius CONFIG [options]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gladius",
        description="Fully autonomous multi-agent ML competition system.",
        epilog="Example: gladius examples/project.yaml -n 10",
    )
    parser.add_argument("config", metavar="CONFIG", help="Project config YAML file.")
    parser.add_argument("--iterations", "-n", type=int, default=None, metavar="N",
                        help="Maximum number of iterations (default: 20).")
    parser.add_argument("--mode", "-m", choices=["experimental", "personal-production"],
                        default="experimental", help="Run mode (default: experimental).")
    parser.add_argument("--parallel", "-p", type=int, default=None, metavar="N",
                        help="Number of parallel branches for autonomous topology (default: 1).")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Start fresh — ignore any existing state DB.")
    parser.add_argument("--no-submit", action="store_true", default=False,
                        help="Disable automatic submission even when the validator approves.")
    parser.add_argument("--max-seconds", type=int, default=None, metavar="S",
                        help="Hard time limit per iteration in seconds.")
    parser.add_argument("--max-agent-calls", type=int, default=None, metavar="N",
                        help="Maximum agent calls per iteration.")
    parser.add_argument("--max-failures", type=int, default=None, metavar="N",
                        help="Abort after this many cumulative failed runs.")

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

    # Always run setup first (idempotent unless force=True in config)
    try:
        setup(config_path)
    except Exception as exc:
        print(f"error during setup: {exc}", file=sys.stderr)
        sys.exit(1)

    competition_dir = cfg["project_dir"]

    try:
        from gladius.orchestrator import run_competition
        asyncio.run(
            run_competition(
                competition_dir=competition_dir,
                max_iterations=args.iterations if args.iterations is not None else 20,
                resume_from_db=not args.no_resume,
                auto_submit=not args.no_submit,
                n_parallel=args.parallel if args.parallel is not None else 1,
                mode=args.mode,
                max_iteration_seconds=args.max_seconds,
                max_agent_calls_per_iteration=args.max_agent_calls,
                max_failed_runs_total=args.max_failures,
                max_turns=cfg.get("max_turns") or {},
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
