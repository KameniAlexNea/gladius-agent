"""CLI entry point: argument parsing and the 'gladius status' command."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ── gladius status ────────────────────────────────────────────────────────────


def print_status(competition_dir: str) -> None:
    """Print a human-readable trace summary from the DB to stdout."""
    gladius_dir = Path(competition_dir) / ".gladius"
    db_path = gladius_dir / "state.db"
    if not db_path.exists():
        logger.info(f"No DB found at {db_path} — competition has not been started.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    comp = conn.execute("SELECT * FROM competition LIMIT 1").fetchone()
    curr = conn.execute("SELECT * FROM current_state WHERE id = 1").fetchone()
    if not comp or not curr:
        logger.info("DB exists but contains no state yet.")
        conn.close()
        return

    logger.info("\n" + "=" * 70)
    logger.info(f"  GLADIUS STATUS — {comp['competition_id']}")
    logger.info("=" * 70)
    logger.info(
        f"  Topology     : {comp.get('topology', 'functional')}  "
        f"(iter {curr['iteration']}/{comp['max_iterations']}  "
        f"done={'yes' if curr['done'] else 'no'})"
    )

    metric = comp["target_metric"]
    if metric:
        best = curr["best_oof_score"]
        best_lb = curr["best_submission_score"]
        logger.info(f"  Metric       : {metric} ({comp['metric_direction']})")
        logger.info(f"  Best OOF     : {f'{best:.6f}' if best is not None else 'none'}")
        logger.info(
            f"  Best LB      : {f'{best_lb:.6f}' if best_lb is not None else 'none'}"
        )
    else:
        best_q = curr["best_quality_score"]
        logger.info("  Task type    : open-ended (quality score)")
        logger.info(
            f"  Best quality : {f'{best_q}/100' if best_q is not None else 'none'}"
        )
    logger.info(
        f"  Submissions  : {curr['submission_count']}/{comp['max_submissions_per_day']} today"
    )
    if curr["last_stop_reason"]:
        logger.info(f"  Stop reason  : {curr['last_stop_reason']}")

    # ── experiments ──────────────────────────────────────────────────────────
    exps = conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    if exps:
        logger.info(f"\n  {'─' * 66}")
        logger.info(f"  EXPERIMENTS ({len(exps)} total)")
        logger.info(f"  {'─' * 66}")
        score_col = "quality_score" if not metric else "oof_score"
        for e in exps:
            score = e[score_col]
            score_str = (
                f"{score:.6f}"
                if (metric and score is not None)
                else (f"{score}/100" if (not metric and score is not None) else "n/a")
            )
            files = e["solution_files"] or ""
            logger.info(f"  iter {e['iteration']:02d}  {score_str:>12}  {files[:50]}")

    # ── plans ────────────────────────────────────────────────────────────────
    try:
        plans = conn.execute("SELECT * FROM plans ORDER BY iteration").fetchall()
        if plans:
            logger.info(f"\n  {'─' * 66}")
            logger.info(f"  PLANS ({len(plans)} total)")
            logger.info(f"  {'─' * 66}")
            for pl in plans:
                summary = (pl["approach_summary"] or "")[:65]
                logger.info(f"  iter {pl['iteration']:02d}  {summary}")
    except sqlite3.OperationalError:
        pass  # old DB without plans table

    # ── event log ────────────────────────────────────────────────────────────
    try:
        events = conn.execute(
            "SELECT * FROM event_log ORDER BY id DESC LIMIT 40"
        ).fetchall()
        if events:
            logger.info(f"\n  {'─' * 66}")
            logger.info("  RECENT EVENTS (newest first)")
            logger.info(f"  {'─' * 66}")
            for ev in events:
                ts = (ev["ts"] or "")[:19]
                detail = (ev["detail"] or "")[:55]
                logger.info(
                    f"  {ts}  iter={ev['iteration']:02d}  {ev['event']:<20} {detail}"
                )
    except sqlite3.OperationalError:
        pass  # old DB without event_log table

    # ── agent runs ───────────────────────────────────────────────────────────
    try:
        runs = conn.execute(
            "SELECT * FROM agent_runs ORDER BY id DESC LIMIT 20"
        ).fetchall()
        if runs:
            logger.info(f"\n  {'─' * 66}")
            logger.info("  RECENT AGENT RUNS (newest first)")
            logger.info(f"  {'─' * 66}")
            for r in runs:
                dur = f"{r['duration_ms'] / 1000:.1f}s" if r["duration_ms"] else "?"
                err = " ERROR" if r["is_error"] else ""
                notes = f"  [{r['notes']}]" if r["notes"] else ""
                logger.info(
                    f"  iter={r['iteration']:02d}  {r['phase']:<14} {r['agent_name']:<16}"
                    f"  {dur:>7}{err}{notes}"
                )
    except sqlite3.OperationalError:
        pass

    # ── errors ───────────────────────────────────────────────────────────────
    errs = conn.execute("SELECT * FROM error_log ORDER BY id DESC LIMIT 10").fetchall()
    if errs:
        logger.info(f"\n  {'─' * 66}")
        logger.info(f"  ERRORS / GUARDRAILS (last {min(len(errs), 10)})")
        logger.info(f"  {'─' * 66}")
        for e in errs:
            logger.info(
                f"  iter={e['iteration']}: {(e['error'] or '')[:80]}"
            )

    logger.info("=" * 70 + "\n")
    conn.close()


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
        "--no-resume", action="store_true", help="Start fresh, ignore saved state"
    )
    run_p.add_argument(
        "--no-submit", action="store_true", help="Dry-run, skip platform submissions"
    )
    run_p.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run N implementers in parallel with different approaches (default: 1)",
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
        "--max-agent-calls-per-iteration",
        type=int,
        default=None,
        help="Optional cap on total planner/implementer/validator/summarizer calls per iteration",
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
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument(
        "--mode",
        choices=["experimental", "personal-production"],
        default="experimental",
    )
    p.add_argument("--max-iteration-seconds", type=int, default=None)
    p.add_argument("--max-agent-calls-per-iteration", type=int, default=None)
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
        resume_from_db=not args.no_resume,
        auto_submit=not args.no_submit,
        n_parallel=args.parallel,
        mode=args.mode,
        max_iteration_seconds=args.max_iteration_seconds,
        max_agent_calls_per_iteration=args.max_agent_calls_per_iteration,
        max_failed_runs_total=args.max_failed_runs_total,
    )


def main() -> None:
    asyncio.run(_amain())
