"""
Main competition loop.

Design rules:
  - Agents output structured JSON; the orchestrator acts on it.
  - Agents NEVER mutate state directly.
  - Two agents per iteration: planner then implementer.
  - Planner is resumed each iteration (accumulates competition understanding).
  - Implementer is fresh each iteration (focused on one plan).
  - State is saved to SQLite after every phase (crash-safe).

Usage:
    gladius --competition-dir examples/fake_competition
    gladius --competition-dir examples/fake_competition --iterations 10 --no-resume
    gladius --competition-dir examples/fake_competition --parallel 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from gladius.agents.implementer import run_implementer
from gladius.agents.planner import run_planner
from gladius.agents.summarizer import run_summarizer
from gladius.agents.validation import run_validation_agent
from gladius.state import CompetitionState, StateStore
from gladius.utils.competition_config import load_competition_config
from gladius.utils.project_setup import setup_project_dir, write_claude_md

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gladius.orchestrator")


# ── Platform submission helpers ───────────────────────────────────────────────
def _submit_to_kaggle(competition_id: str, submission_path: str, message: str) -> None:
    import subprocess

    r = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition_id,
            "-f",
            submission_path,
            "-m",
            message,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.warning(f"Kaggle submit stderr: {r.stderr.strip()}")
    else:
        logger.info(f"Kaggle submission accepted: {r.stdout.strip()}")


def _submit_to_zindi(competition_id: str, submission_path: str, message: str) -> None:
    import os

    try:
        from zindi.user import Zindian
    except ImportError:
        logger.error("zindi package not installed")
        return
    username = os.getenv("ZINDI_USERNAME") or os.getenv("USER_NAME")
    password = os.getenv("ZINDI_PASSWORD") or os.getenv("PASSWORD")
    if not username or not password:
        logger.error("Missing ZINDI_USERNAME / ZINDI_PASSWORD")
        return
    try:
        user = Zindian(username=username, fixed_password=password)
        user.select_a_challenge(
            fixed_index=int(os.getenv("ZINDI_CHALLENGE_INDEX", "0"))
        )
        if user.remaining_subimissions <= 0:
            logger.warning("Zindi: no remaining submissions today")
            return
        user.submit(filepaths=[submission_path], comments=[message])
        logger.info(
            f"Zindi submission accepted ({user.remaining_subimissions} remaining today)"
        )
    except Exception as exc:
        logger.error(f"Zindi submission error: {exc}")


def _submit_to_fake(competition_id: str, submission_path: str, message: str) -> None:
    try:
        from gladius.tools.fake_platform_tools import _score_submission

        score = _score_submission(submission_path)
        logger.info(f"[FAKE PLATFORM] Scored: {score:.6f}")
    except Exception as exc:
        logger.error(f"[FAKE PLATFORM] Scoring failed: {exc}")


def _submit(
    platform: str, competition_id: str, submission_path: str, message: str
) -> None:
    if platform == "zindi":
        _submit_to_zindi(competition_id, submission_path, message)
    elif platform == "fake":
        _submit_to_fake(competition_id, submission_path, message)
    else:
        _submit_to_kaggle(competition_id, submission_path, message)


# ── Improvement check ─────────────────────────────────────────────────────────
def _is_better(
    new_score: float, best_score: float, direction: str, threshold: float = 1e-4
) -> bool:
    if direction == "maximize":
        return new_score > best_score + threshold
    return new_score < best_score - threshold


# ── Main loop ─────────────────────────────────────────────────────────────────
async def run_competition(
    competition_dir: str,
    max_iterations: int = 20,
    resume_from_db: bool = True,
    auto_submit: bool = True,
    n_parallel: int = 1,
) -> CompetitionState:
    cfg = load_competition_config(competition_dir)
    competition_id = cfg["competition_id"]
    platform = cfg["platform"]
    data_dir = cfg["data_dir"]
    target_metric = cfg["metric"]
    metric_direction = cfg["direction"]
    project_dir = competition_dir
    gladius_dir = Path(project_dir) / ".gladius"
    gladius_dir.mkdir(parents=True, exist_ok=True)

    store = StateStore(str(gladius_dir / "state.db"))

    state: CompetitionState | None = store.load() if resume_from_db else None
    if state is None:
        logger.info("Initialising new competition state")
        state = CompetitionState(
            competition_id=competition_id,
            data_dir=str(Path(data_dir).resolve()),
            output_dir=str(gladius_dir.resolve()),
            target_metric=target_metric,
            metric_direction=metric_direction,
            max_iterations=max_iterations,
        )
    else:
        logger.info(
            f"Resuming from iteration {state.iteration}, phase={state.phase}, "
            f"best={state.best_oof_score:.6f}"
        )

    # Bootstrap project directory with Claude Code native config (.claude/,
    # skills, hooks, agent definitions, MEMORY.md). Idempotent — safe to call
    # on resume.
    logger.info("Setting up project directory")
    setup_project_dir(state, project_dir)

    while state.iteration < state.max_iterations and state.phase != "done":
        logger.info(
            f"[iter {state.iteration:02d}/{state.max_iterations}] "
            f"phase={state.phase}  best={state.best_oof_score:.6f}  "
            f"experiments={len(state.experiments)}"
        )

        # Refresh CLAUDE.md with current state so all agents see live context.
        write_claude_md(state, project_dir)

        try:
            # ── PLANNING ─────────────────────────────────────────────────────
            if state.phase == "planning":
                plan, session_id = await run_planner(
                    state,
                    data_dir,
                    project_dir,
                    platform=platform,
                    n_parallel=n_parallel,
                )
                state.current_plan = plan
                state.planner_session_id = session_id
                logger.info(f"Plan ready: {plan.get('approach_summary', '')[:120]}")
                state.phase = "implementing"

            # ── IMPLEMENTING ─────────────────────────────────────────────────
            elif state.phase == "implementing":
                # Gather plans: use planner’s multi-plan list when available
                # and n_parallel > 1, otherwise fall back to single plan.
                alt_plans: list[dict] = (
                    state.current_plan.get("plans", []) if state.current_plan else []
                )
                if n_parallel > 1 and len(alt_plans) > 1:
                    plans_to_run = alt_plans[:n_parallel]
                    logger.info(f"Running {len(plans_to_run)} parallel implementers")
                    results = await asyncio.gather(
                        *[run_implementer(p, state, project_dir) for p in plans_to_run],
                        return_exceptions=True,
                    )
                    # Flatten: keep successful results, log failures
                    successful: list[dict] = []
                    for i, r in enumerate(results):
                        if isinstance(r, Exception):
                            logger.warning(f"Parallel implementer {i} raised: {r}")
                        elif isinstance(r, dict) and r.get("status") == "success":
                            successful.append(r)
                        else:
                            err = (
                                r.get("error_message", "")
                                if isinstance(r, dict)
                                else str(r)
                            )
                            state.failed_runs.append(
                                {
                                    "iteration": state.iteration,
                                    "status": (
                                        r.get("status", "error")
                                        if isinstance(r, dict)
                                        else "error"
                                    ),
                                    "error": err,
                                    "approach": plans_to_run[i].get(
                                        "approach_summary", ""
                                    ),
                                }
                            )
                    if not successful:
                        logger.warning("All parallel implementers failed")
                        state.consecutive_errors += 1
                        state.iteration += 1
                        state.phase = "planning"
                        store.save(state)
                        continue
                    # Pick the best by OOF score
                    direction = state.metric_direction
                    result = max(
                        successful,
                        key=lambda r: (
                            r["oof_score"]
                            if direction == "maximize"
                            else -r["oof_score"]
                        ),
                    )
                    logger.info(
                        f"Best parallel result: OOF {result['oof_score']:.6f} "
                        f"(from {len(successful)}/{len(plans_to_run)} successful)"
                    )
                    # Record all successful runs as experiments
                    for r in successful:
                        state.experiments.append(
                            {
                                "iteration": state.iteration,
                                "oof_score": r["oof_score"],
                                "solution_files": r.get("solution_files", []),
                                "submission_file": r.get("submission_file", ""),
                                "notes": r.get("notes", ""),
                                "approach": "",
                            }
                        )
                else:
                    # ── Sequential single implementer ─────────────────────
                    result = await run_implementer(
                        state.current_plan, state, project_dir
                    )

                    if result["status"] != "success":
                        logger.warning(
                            f"Implementer {result['status']}: {result.get('error_message', '')}"
                        )
                        state.failed_runs.append(
                            {
                                "iteration": state.iteration,
                                "status": result["status"],
                                "error": result.get("error_message", ""),
                                "approach": (
                                    state.current_plan.get("approach_summary", "")
                                    if state.current_plan
                                    else ""
                                ),
                            }
                        )
                        state.consecutive_errors += 1
                        state.iteration += 1
                        state.phase = "planning"
                        store.save(state)
                        continue

                    # Single-run success — record experiment
                    state.experiments.append(
                        {
                            "iteration": state.iteration,
                            "oof_score": result["oof_score"],
                            "solution_files": result.get("solution_files", []),
                            "submission_file": result.get("submission_file", ""),
                            "notes": result.get("notes", ""),
                            "approach": (
                                state.current_plan.get("approach_summary", "")
                                if state.current_plan
                                else ""
                            ),
                        }
                    )

                # ── Common post-implementation logic (parallel + sequential) ─
                state.consecutive_errors = 0
                oof = result["oof_score"]
                logger.info(
                    f"Implementation done — OOF {state.target_metric}: {oof:.6f}"
                )

                if _is_better(oof, state.best_oof_score, state.metric_direction):
                    state.best_oof_score = oof
                    logger.info(f"New best OOF: {oof:.6f}")

                state.phase = "validation"

            # ── VALIDATION ───────────────────────────────────────────────────
            elif state.phase == "validation":
                latest = state.experiments[-1]
                oof_score = latest["oof_score"]
                submission_file = latest["submission_file"]

                validation = await run_validation_agent(
                    solution_path=", ".join(latest.get("solution_files", [])),
                    oof_score=oof_score,
                    submission_path=submission_file,
                    state=state,
                    project_dir=project_dir,
                )

                if (
                    validation["submit"]
                    and submission_file
                    and state.submission_count < state.max_submissions_per_day
                    and auto_submit
                ):
                    state.submission_count += 1
                    state.best_submission_path = submission_file
                    logger.info(f"Submitting [{platform}]: {submission_file}")
                    _submit(
                        platform=platform,
                        competition_id=state.competition_id,
                        submission_path=submission_file,
                        message=f"iter-{state.iteration} oof={oof_score:.6f}",
                    )

                state.iteration += 1

                # Update planner memory with learnings from this iteration.
                try:
                    summary = await run_summarizer(
                        state,
                        project_dir,
                        latest_experiment=latest,
                        validation_notes=validation.get("notes", ""),
                    )
                    if summary:
                        logger.info(f"Summarizer: {summary}")
                except Exception as exc:
                    logger.warning(f"Summarizer failed (non-fatal): {exc}")

                state.phase = "planning"

        except Exception as exc:
            logger.error(
                f"Unhandled error in phase={state.phase}: {exc}", exc_info=True
            )
            state.error_log.append(
                {
                    "phase": state.phase,
                    "iteration": state.iteration,
                    "error": str(exc),
                }
            )
            state.consecutive_errors += 1
            if state.consecutive_errors >= 3:
                logger.critical("3 consecutive errors — halting")
                state.phase = "done"
            else:
                state.phase = "planning"

        finally:
            store.save(state)
            store.close()

    logger.info(
        f"Done. iterations={state.iteration}  "
        f"best_oof={state.best_oof_score:.6f}  "
        f"submissions={state.submission_count}"
    )
    return state


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gladius", description="Autonomous ML competition agent"
    )
    p.add_argument(
        "--competition-dir",
        required=True,
        help="Path to the competition directory (must contain README.md with frontmatter)",
    )
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--no-resume", action="store_true", help="Start fresh, ignore saved state"
    )
    p.add_argument(
        "--no-submit", action="store_true", help="Dry-run, skip platform submissions"
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run N implementers in parallel with different approaches (default: 1)",
    )
    return p


async def _amain() -> None:
    args = _build_parser().parse_args()
    await run_competition(
        competition_dir=args.competition_dir,
        max_iterations=args.iterations,
        resume_from_db=not args.no_resume,
        auto_submit=not args.no_submit,
        n_parallel=args.parallel,
    )


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
