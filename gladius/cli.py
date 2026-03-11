"""CLI entry point: argument parsing and the 'gladius status' command."""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ── gladius status ────────────────────────────────────────────────────────────


def print_status(competition_dir: str) -> None:
    """Print a summary by grepping the gladius.log file."""
    log_path = Path(competition_dir) / "gladius.log"
    if not log_path.exists():
        logger.info(f"No gladius.log found at {log_path} — competition has not been started.")
        return

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Pull out lines logged by the orchestrator (INFO/WARNING/ERROR level)
    interesting = [
        l for l in lines
        if re.search(r"\| (INFO|WARNING|ERROR|CRITICAL) +\|", l)
    ]
    logger.info("=" * 70)
    logger.info(f"  GLADIUS LOG SUMMARY — {competition_dir}")
    logger.info("=" * 70)
    for line in interesting[-60:]:
        # Strip loguru prefix for cleaner display
        msg = re.sub(r"^.*?\| (?:INFO|WARNING|ERROR|CRITICAL) +\| [^:]+:[^:]+:\d+ - ", "", line)
        logger.info(f"  {msg}")
    logger.info("=" * 70 + "\n")


# ── Argument parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gladius", description="Autonomous ML competition agent"
    )
    sub = p.add_subparsers(dest="command")

    # ── run (explicit subcommand) ─────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run the competition agent (default)")
    run_p.add_argument(
        "--competition-dir",
        required=True,
        help="Path to the competition directory (must contain README.md with frontmatter)",
    )
    run_p.add_argument("--iterations", type=int, default=20)
    run_p.add_argument(
        "--no-submit", action="store_true", help="Dry-run, skip platform submissions"
    )
    run_p.add_argument(
        "--mode",
        choices=["experimental", "personal-production"],
        default="experimental",
        help=(
            "Runtime profile: experimental (default) or personal-production "
            "with stricter guardrails"
        ),
    )
    run_p.add_argument(
        "--max-iteration-seconds",
        type=int,
        default=None,
        help="Optional hard runtime budget per iteration in seconds",
    )
    run_p.add_argument(
        "--max-failed-runs-total",
        type=int,
        default=None,
        help="Optional cap on cumulative failed implementer runs before forced stop",
    )

    # ── status ────────────────────────────────────────────────────────────────
    status_p = sub.add_parser(
        "status", help="Print a trace summary of a competition run from the DB"
    )
    status_p.add_argument(
        "--competition-dir",
        required=True,
        help="Path to the competition directory",
    )

    # Back-compat: gladius --competition-dir X  (no subcommand → treated as 'run')
    p.add_argument("--competition-dir", default=None)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--no-submit", action="store_true")
    p.add_argument(
        "--mode",
        choices=["experimental", "personal-production"],
        default="experimental",
    )
    p.add_argument("--max-iteration-seconds", type=int, default=None)
    p.add_argument("--max-failed-runs-total", type=int, default=None)

    return p


# ── Entry point ───────────────────────────────────────────────────────────────


async def _amain() -> None:
    from gladius.orchestrator import run_competition

    args = build_parser().parse_args()

    command = args.command
    if command is None and args.competition_dir:
        command = "run"

    if command == "status":
        print_status(args.competition_dir)
        return

    if command not in ("run", None) or not args.competition_dir:
        build_parser().print_help()
        return

    _env_file = Path(args.competition_dir) / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
        logger.debug(f"Loaded env from {_env_file}")

    await run_competition(
        competition_dir=args.competition_dir,
        max_iterations=args.iterations,
        auto_submit=not args.no_submit,
        mode=args.mode,
        max_iteration_seconds=args.max_iteration_seconds,
        max_failed_runs_total=args.max_failed_runs_total,
    )


def main() -> None:
    asyncio.run(_amain())
