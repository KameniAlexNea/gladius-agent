"""
Gladius CLI.

  gladius setup CONFIG   — bootstrap a competition project directory
  gladius run   CONFIG   — run the competition agent loop
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _add_setup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", metavar="CONFIG", help="Project config YAML file.")
    parser.add_argument("--force", "-f", action="store_true", default=False,
                        help="Overwrite existing files.")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config",
        metavar="CONFIG",
        help="Project config YAML file. The competition directory is read from "
             "the 'project_dir' key.",
    )
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


def _run_setup(args: argparse.Namespace) -> None:
    try:
        from gladius.project_setup import ConfigError, setup
        setup(args.config, force=args.force)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_competition(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        from gladius.project_setup import load_config
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    competition_dir = cfg["project_dir"]
    if not Path(competition_dir).is_dir():
        print(
            f"error: project_dir '{competition_dir}' does not exist. "
            "Run 'gladius setup' first.",
            file=sys.stderr,
        )
        sys.exit(1)

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


# ── Main entry point ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gladius",
        description="Fully autonomous multi-agent ML competition system.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_setup = sub.add_parser(
        "setup",
        help="Bootstrap a competition project directory from a config file.",
        epilog="Example: gladius setup examples/project.yaml",
    )
    _add_setup_args(p_setup)

    p_run = sub.add_parser(
        "run",
        help="Run the competition agent loop.",
        epilog="Example: gladius run examples/project.yaml -n 10",
    )
    _add_run_args(p_run)

    args = parser.parse_args(argv)

    if args.command == "setup":
        _run_setup(args)
    elif args.command == "run":
        _run_competition(args)
    else:
        parser.print_help()
        sys.exit(1)


# ── Legacy stand-alone entry points (kept for backward compatibility) ─────────


def setup_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gladius-setup",
        description="Bootstrap a Gladius competition project from a YAML config.",
        epilog="Example: gladius-setup examples/project.yaml --force",
    )
    _add_setup_args(parser)
    _run_setup(parser.parse_args(argv))


def run_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gladius-run",
        description="Run a Gladius competition agent loop from a project config file.",
        epilog="Example: gladius-run examples/project.yaml -n 10 --mode personal-production",
    )
    _add_run_args(parser)
    _run_competition(parser.parse_args(argv))
