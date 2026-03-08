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
import importlib.util
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

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


def _build_preflight_errors(
    *,
    competition_dir: str,
    platform: str,
    data_dir: str,
    target_metric: str | None,
    max_iterations: int,
    n_parallel: int,
) -> list[str]:
    errors: list[str] = []

    if max_iterations <= 0:
        errors.append("max_iterations must be > 0")
    if n_parallel <= 0:
        errors.append("--parallel must be >= 1")

    model_name = os.environ.get("GLADIUS_MODEL")
    if not model_name:
        errors.append("GLADIUS_MODEL is not set in environment/.env")

    cdir = Path(competition_dir)
    if not cdir.exists():
        errors.append(f"competition directory does not exist: {competition_dir}")

    if target_metric is not None:
        ddir = Path(data_dir)
        if not ddir.exists():
            errors.append(f"data_dir does not exist for metric task: {data_dir}")

    if platform == "kaggle" and shutil.which("kaggle") is None:
        errors.append(
            "kaggle CLI not found on PATH (required for kaggle platform checks/submission)"
        )
    if platform == "kaggle":
        has_env_creds = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        has_file_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
        if not (has_env_creds or has_file_creds):
            errors.append(
                "Kaggle credentials missing (set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)"
            )
    if platform == "zindi" and importlib.util.find_spec("zindi") is None:
        errors.append("zindi package is not installed (required for zindi platform)")
    if platform == "zindi":
        has_zindi_creds = bool(
            (os.getenv("ZINDI_USERNAME") or os.getenv("USER_NAME"))
            and (os.getenv("ZINDI_PASSWORD") or os.getenv("PASSWORD"))
        )
        if not has_zindi_creds:
            errors.append(
                "Zindi credentials missing (set ZINDI_USERNAME and ZINDI_PASSWORD)"
            )

    return errors


def _run_preflight_or_raise(
    *,
    competition_dir: str,
    platform: str,
    data_dir: str,
    target_metric: str | None,
    max_iterations: int,
    n_parallel: int,
) -> None:
    errors = _build_preflight_errors(
        competition_dir=competition_dir,
        platform=platform,
        data_dir=data_dir,
        target_metric=target_metric,
        max_iterations=max_iterations,
        n_parallel=n_parallel,
    )
    if errors:
        msg = "Preflight checks failed:\n- " + "\n- ".join(errors)
        raise ValueError(msg)


def _compute_hybrid_quality_score(
    *,
    implementer_quality_score: float | None,
    validator_quality_score: float | None,
    validation: dict,
) -> float:
    """Compute conservative hybrid quality score for open-ended tasks.

    Weight validator judgement more heavily, then apply deterministic penalties
    for known remaining gaps.
    """
    impl = float(implementer_quality_score or 0.0)
    val = validator_quality_score

    if val is None:
        score = impl
    else:
        score = 0.75 * float(val) + 0.25 * impl

    if validation.get("format_ok") is False:
        score -= 15.0

    next_dirs = validation.get("next_directions") or []
    if isinstance(next_dirs, list):
        score -= min(len(next_dirs), 3) * 2.0

    score = max(0.0, min(100.0, score))
    return round(score, 2)


# ── Platform submission helpers ───────────────────────────────────────────────
def _submit_to_kaggle(
    competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
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
        return False, "submission_failed"
    else:
        logger.info(f"Kaggle submission accepted: {r.stdout.strip()}")
        return True, None


def _submit_to_zindi(
    competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    try:
        from zindi.user import Zindian
    except ImportError:
        logger.error("zindi package not installed")
        return False, "dependency_missing"
    username = os.getenv("ZINDI_USERNAME") or os.getenv("USER_NAME")
    password = os.getenv("ZINDI_PASSWORD") or os.getenv("PASSWORD")
    if not username or not password:
        logger.error("Missing ZINDI_USERNAME / ZINDI_PASSWORD")
        return False, "auth_missing"
    try:
        user = Zindian(username=username, fixed_password=password)
        user.select_a_challenge(
            fixed_index=int(os.getenv("ZINDI_CHALLENGE_INDEX", "0"))
        )
        if user.remaining_subimissions <= 0:
            logger.warning("Zindi: no remaining submissions today")
            return False, "quota_exceeded"
        user.submit(filepaths=[submission_path], comments=[message])
        logger.info(
            f"Zindi submission accepted ({user.remaining_subimissions} remaining today)"
        )
        return True, None
    except Exception as exc:
        logger.error(f"Zindi submission error: {exc}")
        return False, "submission_failed"


def _submit_to_fake(
    competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    try:
        from gladius.tools.fake_platform_tools import _score_submission

        score = _score_submission(submission_path)
        logger.info(f"[FAKE PLATFORM] Scored: {score:.6f}")
        return True, None
    except Exception as exc:
        logger.error(f"[FAKE PLATFORM] Scoring failed: {exc}")
        return False, "scoring_failed"


def _submit(
    platform: str, competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    if platform == "none":
        # No external platform — artifact is recorded in state only.
        logger.info(f"[LOCAL] Submission artifact recorded: {submission_path}")
        return True, None
    if platform == "zindi":
        return _submit_to_zindi(competition_id, submission_path, message)
    elif platform == "fake":
        return _submit_to_fake(competition_id, submission_path, message)
    else:
        return _submit_to_kaggle(competition_id, submission_path, message)


def _score_submission_artifact(platform: str, submission_path: str) -> float | None:
    if platform != "fake":
        return None
    try:
        from gladius.tools.fake_platform_tools import _score_submission

        return float(_score_submission(submission_path))
    except Exception as exc:
        logger.warning(f"Could not compute submission score for {platform}: {exc}")
        return None


def _update_best_submission_score(
    *,
    state: CompetitionState,
    new_score: float,
) -> None:
    if state.best_submission_score is None:
        state.best_submission_score = new_score
        return
    if state.target_metric is None:
        if new_score > state.best_submission_score:
            state.best_submission_score = new_score
        return
    if state.metric_direction == "maximize":
        if new_score > state.best_submission_score:
            state.best_submission_score = new_score
    else:
        if new_score < state.best_submission_score:
            state.best_submission_score = new_score


def _halt_with_reason(state: CompetitionState, *, phase: str, reason: str) -> None:
    state.last_stop_reason = reason
    state.error_log.append(
        {
            "phase": phase,
            "iteration": state.iteration,
            "error": reason,
        }
    )
    state.phase = "done"
    logger.warning(f"Guardrail stop [{phase}]: {reason}")


# ── Improvement check ─────────────────────────────────────────────────────────
def _is_better(
    new_score: float,
    best_score: float | None,
    direction: str | None,
    threshold: float = 1e-4,
) -> bool:
    """Deterministic improvement check.  None best_score means no prior result.

    direction=None means open-ended task: quality score (0-100) where higher
    is always better, with a larger threshold of 2.0 points.
    """
    if best_score is None:  # no prior result — always an improvement
        return True
    if direction is None:  # open-ended: quality score, higher always better
        return new_score > best_score + 2.0
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
    mode: str = "experimental",
    max_iteration_seconds: int | None = None,
    max_agent_calls_per_iteration: int | None = None,
    max_failed_runs_total: int | None = None,
) -> CompetitionState:
    cfg = load_competition_config(competition_dir)
    competition_id = cfg["competition_id"]
    platform = cfg["platform"]
    data_dir = cfg["data_dir"]
    target_metric = cfg["metric"]
    metric_direction = cfg["direction"]

    if mode == "personal-production":
        if max_iteration_seconds is None:
            max_iteration_seconds = 1800
        if max_agent_calls_per_iteration is None:
            max_agent_calls_per_iteration = 5
        if max_failed_runs_total is None:
            max_failed_runs_total = 20
        n_parallel = 1

    _run_preflight_or_raise(
        competition_dir=competition_dir,
        platform=platform,
        data_dir=data_dir,
        target_metric=target_metric,
        max_iterations=max_iterations,
        n_parallel=n_parallel,
    )

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
        _best_str = (
            f"{state.best_oof_score:.6f}"
            if state.best_oof_score is not None
            else "none"
        )
        logger.info(
            f"Resuming from iteration {state.iteration}, phase={state.phase}, "
            f"best={_best_str}"
        )
        if state.max_iterations != max_iterations:
            logger.info(
                f"Updating max_iterations from CLI: "
                f"{state.max_iterations} → {max_iterations}"
            )
            state.max_iterations = max_iterations

    # Bootstrap project directory with Claude Code native config (.claude/,
    # skills, hooks, agent definitions, MEMORY.md). Idempotent — safe to call
    # on resume.
    logger.info("Setting up project directory")
    setup_project_dir(state, project_dir, platform=platform)

    while state.iteration < state.max_iterations and state.phase != "done":
        iteration_started = time.monotonic()
        agent_calls_this_iteration = 0

        def _consume_agent_call(agent_label: str) -> bool:
            return _consume_agent_calls(agent_label, 1)

        def _consume_agent_calls(agent_label: str, count: int) -> bool:
            nonlocal agent_calls_this_iteration
            agent_calls_this_iteration += count
            if (
                max_agent_calls_per_iteration is not None
                and agent_calls_this_iteration > max_agent_calls_per_iteration
            ):
                _halt_with_reason(
                    state,
                    phase="guardrail",
                    reason=(
                        "agent call budget exceeded "
                        f"({agent_calls_this_iteration}/{max_agent_calls_per_iteration}) "
                        f"before {agent_label}"
                    ),
                )
                return False
            return True

        def _check_iteration_runtime_budget() -> bool:
            if max_iteration_seconds is None:
                return True
            elapsed = time.monotonic() - iteration_started
            if elapsed > max_iteration_seconds:
                _halt_with_reason(
                    state,
                    phase="guardrail",
                    reason=(
                        "iteration runtime budget exceeded "
                        f"({elapsed:.1f}s > {max_iteration_seconds}s)"
                    ),
                )
                return False
            return True

        if (
            max_failed_runs_total is not None
            and len(state.failed_runs) >= max_failed_runs_total
        ):
            _halt_with_reason(
                state,
                phase="guardrail",
                reason=(
                    "failed run budget exceeded "
                    f"({len(state.failed_runs)}/{max_failed_runs_total})"
                ),
            )

        _score_str = (
            f"quality={f'{state.best_quality_score}/100' if state.best_quality_score is not None else 'none'}"
            if state.target_metric is None
            else f"best={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
        )
        logger.info(
            f"[iter {state.iteration:02d}/{state.max_iterations}] "
            f"phase={state.phase}  {_score_str}  "
            f"experiments={len(state.experiments)}"
        )

        # Refresh CLAUDE.md with current state so all agents see live context.
        write_claude_md(state, project_dir)

        try:
            # ── PLANNING ─────────────────────────────────────────────────────
            if state.phase == "planning":
                if not _consume_agent_call("planner"):
                    continue
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
                if not _check_iteration_runtime_budget():
                    continue

            # ── IMPLEMENTING ─────────────────────────────────────────────────
            elif state.phase == "implementing":
                if state.current_plan is None:
                    logger.warning(
                        "Resuming in 'implementing' phase with no current_plan "
                        "— falling back to planning"
                    )
                    state.phase = "planning"
                    store.save(state)
                    continue
                # Gather plans: use planner's multi-plan list when available
                # and n_parallel > 1, otherwise fall back to single plan.
                alt_plans: list[dict] = (
                    state.current_plan.get("plans", []) if state.current_plan else []
                )
                if n_parallel > 1 and len(alt_plans) > 1:
                    plans_to_run = alt_plans[:n_parallel]
                    if not _consume_agent_calls(
                        "parallel implementers", len(plans_to_run)
                    ):
                        continue
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
                            state.failed_runs.append(
                                {
                                    "iteration": state.iteration,
                                    "status": "error",
                                    "error": str(r),
                                    "approach": plans_to_run[i].get(
                                        "approach_summary", ""
                                    ),
                                }
                            )
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
                    # Pick the best result: by OOF for ML tasks, by quality_score for open tasks
                    if state.target_metric is None:
                        result = max(
                            successful, key=lambda r: r.get("quality_score", 0) or 0
                        )
                    else:
                        direction = state.metric_direction
                        result = max(
                            successful,
                            key=lambda r: (
                                r["oof_score"]
                                if direction == "maximize"
                                else -r["oof_score"]
                            ),
                        )
                    _best_score_str = (
                        f"quality {result.get('quality_score', 0)}/100"
                        if state.target_metric is None
                        else f"OOF {result['oof_score']:.6f}"
                    )
                    logger.info(
                        f"Best parallel result: {_best_score_str} "
                        f"(from {len(successful)}/{len(plans_to_run)} successful)"
                    )
                    # Record all successful runs as experiments — result last so
                    # experiments[-1] always points to the best.
                    for r in successful:
                        if r is not result:
                            state.experiments.append(
                                {
                                    "iteration": state.iteration,
                                    "oof_score": r.get("oof_score"),
                                    "quality_score": r.get("quality_score"),
                                    "solution_files": r.get("solution_files", []),
                                    "submission_file": r.get("submission_file", ""),
                                    "notes": r.get("notes", ""),
                                    "approach": "",
                                }
                            )
                    state.experiments.append(
                        {
                            "iteration": state.iteration,
                            "oof_score": result.get("oof_score"),
                            "quality_score": result.get("quality_score"),
                            "solution_files": result.get("solution_files", []),
                            "submission_file": result.get("submission_file", ""),
                            "notes": result.get("notes", ""),
                            "approach": "",
                        }
                    )
                else:
                    # ── Sequential single implementer ─────────────────────
                    if not _consume_agent_call("implementer"):
                        continue
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
                            "oof_score": result.get("oof_score"),
                            "quality_score": result.get("quality_score"),
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
                if state.target_metric:
                    _oof = result.get("oof_score")
                    logger.info(
                        f"Implementation done — OOF {state.target_metric}: "
                        f"{f'{_oof:.6f}' if _oof is not None else 'n/a'}"
                    )
                else:
                    _quality = result.get("quality_score", 0) or 0
                    logger.info(f"Implementation done — quality: {_quality}/100")

                # Do NOT update state.best_oof_score here — the validation agent
                # must compare against the *previous* best.  The update happens
                # in the validation phase after the agent confirms improvement.
                state.phase = "validation"
                if not _check_iteration_runtime_budget():
                    continue

            # ── VALIDATION ───────────────────────────────────────────────────
            elif state.phase == "validation":
                latest = state.experiments[-1]
                oof_score = latest.get("oof_score")  # None for open-ended tasks
                quality_score = (
                    latest.get("quality_score", 0) or 0
                )  # implementer self-score
                submission_file = latest["submission_file"]

                # Reset daily submission counter when the calendar date rolls over.
                from datetime import date as _date

                today = _date.today().isoformat()
                if state.last_submission_date != today:
                    if state.submission_count > 0:
                        logger.info(
                            f"New day ({today}) — resetting submission_count "
                            f"from {state.submission_count} to 0"
                        )
                    state.submission_count = 0
                    state.last_submission_date = today

                if not submission_file:
                    logger.warning(
                        "No submission file produced — skipping format check, "
                        "running deterministic improvement check only"
                    )
                    validation: dict = {
                        "is_improvement": None,
                        "submit": False,
                        "reasoning": "No submission file produced",
                    }
                else:
                    if not _consume_agent_call("validation"):
                        continue
                    validation = await run_validation_agent(
                        solution_path=", ".join(latest.get("solution_files", [])),
                        oof_score=oof_score,
                        quality_score=quality_score,
                        submission_path=submission_file,
                        state=state,
                        project_dir=project_dir,
                        platform=platform,
                    )

                validator_quality_score = validation.get("quality_score")
                if state.target_metric is None:
                    hybrid_quality_score = _compute_hybrid_quality_score(
                        implementer_quality_score=quality_score,
                        validator_quality_score=validator_quality_score,
                        validation=validation,
                    )
                    latest["quality_score"] = hybrid_quality_score
                    logger.info(
                        "Open-task quality scoring: "
                        f"implementer={quality_score}/100, "
                        f"validator={validator_quality_score if validator_quality_score is not None else 'n/a'}/100, "
                        f"hybrid={hybrid_quality_score}/100"
                    )

                # Deterministic improvement gate — LLM verdict is advisory only.
                # For ML tasks: compare oof_score against best_oof_score.
                # For open tasks: compare quality_score against best_quality_score.
                primary_score = (
                    hybrid_quality_score if state.target_metric is None else oof_score
                )
                best_primary = (
                    state.best_quality_score
                    if state.target_metric is None
                    else state.best_oof_score
                )
                deterministic_improvement = _is_better(
                    primary_score, best_primary, state.metric_direction
                )
                if validation.get("is_improvement") != deterministic_improvement:
                    _score_label = (
                        f"quality {primary_score}/100 vs best "
                        f"{f'{best_primary}/100' if best_primary is not None else 'none'}"
                        if state.target_metric is None
                        else f"OOF {oof_score:.6f} vs best "
                        f"{f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
                    )
                    logger.warning(
                        f"Validation agent is_improvement={validation.get('is_improvement')} "
                        f"overridden by deterministic check → {deterministic_improvement} "
                        f"({_score_label})"
                    )
                if deterministic_improvement:
                    if state.target_metric is None:
                        state.best_quality_score = primary_score
                        logger.info(f"New best quality score: {primary_score}/100")
                    else:
                        state.best_oof_score = oof_score
                        logger.info(f"New best OOF: {oof_score:.6f}")

                if (
                    deterministic_improvement
                    and submission_file
                    and (
                        platform == "none"
                        or state.submission_count < state.max_submissions_per_day
                    )
                    and auto_submit
                ):
                    logger.info(f"Submitting [{platform}]: {submission_file}")
                    _submit_msg = (
                        f"iter-{state.iteration} quality={primary_score}/100"
                        if state.target_metric is None
                        else f"iter-{state.iteration} oof={oof_score:.6f}"
                    )
                    submit_ok, submit_error_type = _submit(
                        platform=platform,
                        competition_id=state.competition_id,
                        submission_path=submission_file,
                        message=_submit_msg,
                    )
                    if submit_ok:
                        state.best_submission_path = submission_file
                        if platform != "none":
                            state.submission_count += 1
                        scored_lb = _score_submission_artifact(
                            platform=platform,
                            submission_path=submission_file,
                        )
                        if scored_lb is not None:
                            from datetime import datetime as _datetime

                            state.lb_scores.append(
                                {
                                    "score": scored_lb,
                                    "timestamp": _datetime.now().isoformat(),
                                    "public_lb": True,
                                }
                            )
                            _update_best_submission_score(
                                state=state,
                                new_score=scored_lb,
                            )
                            logger.info(
                                f"Recorded leaderboard score: {scored_lb:.6f} "
                                f"(best={state.best_submission_score:.6f})"
                            )
                    else:
                        state.error_log.append(
                            {
                                "phase": "submission",
                                "iteration": state.iteration,
                                "error": f"submit_failed:{submit_error_type or 'unknown'}",
                            }
                        )

                # Update planner memory with learnings from this iteration.
                # Increment iteration AFTER summarizer so MEMORY.md records the correct number.
                try:
                    if not _consume_agent_call("summarizer"):
                        continue
                    summary = await run_summarizer(
                        state,
                        project_dir,
                        latest_experiment=latest,
                        validation_notes=validation.get("reasoning", ""),
                    )
                    if summary:
                        logger.info(f"Summarizer: {summary}")
                except Exception as exc:
                    logger.warning(f"Summarizer failed (non-fatal): {exc}")

                if not _check_iteration_runtime_budget():
                    continue

                state.consecutive_errors = 0
                state.iteration += 1

                # ── Deterministic stop check ─────────────────────────────
                # Always computed in Python — never rely solely on the LLM.
                # Stopping requires ALL THREE conditions to be true simultaneously:
                #   1. Plateau detected (last 3 scores span < threshold)
                #   2. Validator set stop=True
                #   3. Validator listed no next_directions
                # Any subset is insufficient — e.g. a perfect score with
                # remaining suggested directions will keep iterating.
                deterministic_stop = False

                # Both task types: plateau detection — last 3 scored
                # experiments show no meaningful change.
                _plateau_key = (
                    "quality_score" if state.target_metric is None else "oof_score"
                )
                _scored = [
                    e[_plateau_key]
                    for e in state.experiments
                    if e.get(_plateau_key) is not None
                ]
                _plateau_threshold = 3.0 if state.target_metric is None else 0.001
                if len(_scored) >= 3:
                    _span = max(_scored[-3:]) - min(_scored[-3:])
                    if _span < _plateau_threshold:
                        deterministic_stop = True
                        logger.info(
                            f"Deterministic stop: last 3 scores span {_span:.4f} "
                            f"(< {_plateau_threshold}) — plateau detected."
                        )

                # Stop only when ALL THREE agree:
                #   1. deterministic_stop  — plateau detected (scores stagnant)
                #   2. agent_stop          — validator explicitly set stop=True
                #   3. no_next_directions  — validator listed no further improvements
                # Any single condition alone is insufficient.
                agent_stop = bool(validation.get("stop"))
                no_next_directions = not bool(validation.get("next_directions"))
                should_stop = deterministic_stop and agent_stop and no_next_directions
                if should_stop:
                    logger.info(
                        "Stopping: plateau detected + agent stop=True + no next directions."
                    )
                elif agent_stop and not should_stop:
                    _why = []
                    if not deterministic_stop:
                        _why.append("no plateau")
                    if not no_next_directions:
                        _why.append(
                            f"next_directions={validation.get('next_directions')}"
                        )
                    logger.info(
                        f"Agent requested stop=True but continuing — {', '.join(_why)}."
                    )
                state.phase = "done" if should_stop else "planning"

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
                state.last_stop_reason = "consecutive error budget exceeded (3/3)"
                state.phase = "done"
            else:
                state.phase = "planning"

        finally:
            store.save(state)

    store.close()
    if state.target_metric:
        _final_score = f"best_oof={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
    else:
        _final_score = f"best_quality={f'{state.best_quality_score}/100' if state.best_quality_score is not None else 'none'}"
    logger.info(
        f"Done. iterations={state.iteration}  "
        f"{_final_score}  "
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
    p.add_argument(
        "--mode",
        choices=["experimental", "personal-production"],
        default="experimental",
        help=(
            "Runtime profile: experimental (default) or personal-production "
            "with stricter guardrails"
        ),
    )
    p.add_argument(
        "--max-iteration-seconds",
        type=int,
        default=None,
        help="Optional hard runtime budget per iteration in seconds",
    )
    p.add_argument(
        "--max-agent-calls-per-iteration",
        type=int,
        default=None,
        help="Optional cap on total planner/implementer/validator/summarizer calls per iteration",
    )
    p.add_argument(
        "--max-failed-runs-total",
        type=int,
        default=None,
        help="Optional cap on cumulative failed implementer runs before forced stop",
    )
    return p


async def _amain() -> None:
    args = _build_parser().parse_args()
    # Load .env from the competition directory so Ollama / proxy env vars
    # (ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, etc.) are propagated to the
    # bundled claude subprocess which inherits os.environ.
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


if __name__ == "__main__":
    main()
