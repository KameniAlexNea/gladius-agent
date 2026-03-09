"""CLI entry point: argument parsing and the 'gladius status' command."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ── gladius status ────────────────────────────────────────────────────────────

def print_status(competition_dir: str) -> None:
    """Print a human-readable trace summary from the DB to stdout."""
    gladius_dir = Path(competition_dir) / ".gladius"
    db_path = gladius_dir / "state.db"
    if not db_path.exists():
        print(f"No DB found at {db_path} — competition has not been started.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    comp = conn.execute("SELECT * FROM competition LIMIT 1").fetchone()
    curr = conn.execute("SELECT * FROM current_state WHERE id = 1").fetchone()
    if not comp or not curr:
        print("DB exists but contains no state yet.")
        conn.close()
        return

    print("\n" + "=" * 70)
    print(f"  GLADIUS STATUS — {comp['competition_id']}")
    print("=" * 70)
    print(
        f"  Phase        : {curr['phase']}  "
        f"(iter {curr['iteration']}/{comp['max_iterations']})"
    )

    metric = comp["target_metric"]
    if metric:
        best = curr["best_oof_score"]
        best_lb = curr["best_submission_score"]
        print(f"  Metric       : {metric} ({comp['metric_direction']})")
        print(f"  Best OOF     : {f'{best:.6f}' if best is not None else 'none'}")
        print(f"  Best LB      : {f'{best_lb:.6f}' if best_lb is not None else 'none'}")
    else:
        best_q = curr["best_quality_score"]
        print("  Task type    : open-ended (quality score)")
        print(f"  Best quality : {f'{best_q}/100' if best_q is not None else 'none'}")
    print(f"  Submissions  : {curr['submission_count']}/{comp['max_submissions_per_day']} today")
    if curr["last_stop_reason"]:
        print(f"  Stop reason  : {curr['last_stop_reason']}")

    # ── experiments ──────────────────────────────────────────────────────────
    exps = conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    if exps:
        print(f"\n  {'─'*66}")
        print(f"  EXPERIMENTS ({len(exps)} total)")
        print(f"  {'─'*66}")
        score_col = "quality_score" if not metric else "oof_score"
        for e in exps:
            score = e[score_col]
            score_str = (
                f"{score:.6f}" if (metric and score is not None)
                else (f"{score}/100" if (not metric and score is not None) else "n/a")
            )
            files = e["solution_files"] or ""
            print(f"  iter {e['iteration']:02d}  {score_str:>12}  {files[:50]}")

    # ── plans ────────────────────────────────────────────────────────────────
    try:
        plans = conn.execute("SELECT * FROM plans ORDER BY iteration").fetchall()
        if plans:
            print(f"\n  {'─'*66}")
            print(f"  PLANS ({len(plans)} total)")
            print(f"  {'─'*66}")
            for pl in plans:
                summary = (pl["approach_summary"] or "")[:65]
                print(f"  iter {pl['iteration']:02d}  {summary}")
    except sqlite3.OperationalError:
        pass  # old DB without plans table

    # ── event log ────────────────────────────────────────────────────────────
    try:
        events = conn.execute(
            "SELECT * FROM event_log ORDER BY id DESC LIMIT 40"
        ).fetchall()
        if events:
            print(f"\n  {'─'*66}")
            print("  RECENT EVENTS (newest first)")
            print(f"  {'─'*66}")
            for ev in events:
                ts = (ev["ts"] or "")[:19]
                detail = (ev["detail"] or "")[:55]
                print(
                    f"  {ts}  iter={ev['iteration']:02d}  "
                    f"{ev['event']:<20} {detail}"
                )
    except sqlite3.OperationalError:
        pass  # old DB without event_log table

    # ── agent runs ───────────────────────────────────────────────────────────
    try:
        runs = conn.execute(
            "SELECT * FROM agent_runs ORDER BY id DESC LIMIT 20"
        ).fetchall()
        if runs:
            print(f"\n  {'─'*66}")
            print("  RECENT AGENT RUNS (newest first)")
            print(f"  {'─'*66}")
            for r in runs:
                dur = f"{r['duration_ms'] / 1000:.1f}s" if r["duration_ms"] else "?"
                err = " ERROR" if r["is_error"] else ""
                notes = f"  [{r['notes']}]" if r["notes"] else ""
                print(
                    f"  iter={r['iteration']:02d}  {r['phase']:<14} {r['agent_name']:<16}"
                    f"  {dur:>7}{err}{notes}"
                )
    except sqlite3.OperationalError:
        pass

    # ── errors ───────────────────────────────────────────────────────────────
    errs = conn.execute(
        "SELECT * FROM error_log ORDER BY id DESC LIMIT 10"
    ).fetchall()
    if errs:
        print(f"\n  {'─'*66}")
        print(f"  ERRORS / GUARDRAILS (last {min(len(errs), 10)})")
        print(f"  {'─'*66}")
        for e in errs:
            print(f"  iter={e['iteration']}  phase={e['phase']}: {(e['error'] or '')[:80]}")

    print("=" * 70 + "\n")
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
        logger.debug("Loaded env from %s", _env_file)

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
