"""
Main competition loop — the only place routing and state mutation live.

Design rules:
  - Agents output structured JSON; the orchestrator acts on it.
  - Agents NEVER update best_oof_score or state directly.
  - All routing is explicit if/elif; no LangGraph edges.
  - State is saved to SQLite after every phase (crash-safe).
  - Session IDs are stored in state so agents resume their context.

Usage:
    python -m gladius.orchestrator \\
        --competition titanic \\
        --data-dir /data/titanic \\
        --project-dir . \\
        --metric auc_roc \\
        --direction maximize
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from gladius.agents.code import run_code_agent
from gladius.agents.ensemble import run_ensemble_agent
from gladius.agents.execution import run_execution_agent
from gladius.agents.strategy import run_strategy_agent
from gladius.agents.validation import run_validation_agent
from gladius.state import CompetitionState, StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gladius.orchestrator")

# ── Optional Kaggle submission helper ─────────────────────────────────────────
def _submit_to_kaggle(
    competition_id: str, submission_path: str, message: str
) -> None:
    """Fire-and-forget Kaggle CLI submission."""
    import subprocess

    result = subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", competition_id,
            "-f", submission_path,
            "-m", message,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Kaggle submit stderr: {result.stderr.strip()}")
    else:
        logger.info(f"Submission accepted: {result.stdout.strip()}")


# ── Main competition loop ─────────────────────────────────────────────────────
async def run_competition(
    competition_id: str,
    data_dir: str,
    project_dir: str,
    target_metric: str = "auc_roc",
    metric_direction: str = "maximize",
    max_iterations: int = 20,
    resume_from_db: bool = True,
    ensemble_every_n: int = 5,
    max_runtime_minutes: int = 90,
    auto_submit: bool = True,
) -> CompetitionState:
    """
    Run the autonomous competition loop.

    Returns the final CompetitionState.
    """
    gladius_dir = Path(project_dir) / ".gladius"
    gladius_dir.mkdir(parents=True, exist_ok=True)

    store = StateStore(str(gladius_dir / "state.db"))

    # ── Resume or initialise ──────────────────────────────────────────────────
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
            f"best_oof={state.best_oof_score:.6f}"
        )

    # ── Main loop ─────────────────────────────────────────────────────────────
    while state.iteration < state.max_iterations and state.phase != "done":
        logger.info(
            f"[iter {state.iteration:02d}/{state.max_iterations}] "
            f"phase={state.phase}  best_oof={state.best_oof_score:.6f}"
        )

        try:
            # ── STRATEGY ─────────────────────────────────────────────────────
            if state.phase == "strategy":
                hypothesis, session_id = await run_strategy_agent(state, data_dir)
                state.current_hypothesis = hypothesis
                state.strategy_session_id = session_id
                logger.info(f"Hypothesis: {hypothesis.get('hypothesis', '?')}")
                state.phase = "coding"

            # ── CODING ───────────────────────────────────────────────────────
            elif state.phase == "coding":
                code_result, session_id = await run_code_agent(
                    state.current_hypothesis, state, project_dir
                )
                state.code_session_id = session_id
                # Attach solution path to hypothesis dict for tracking
                state.current_hypothesis["solution_path"] = code_result["solution_path"]
                logger.info(f"Solution written: {code_result['solution_path']}")
                state.phase = "execution"

            # ── EXECUTION ────────────────────────────────────────────────────
            elif state.phase == "execution":
                solution_path = state.current_hypothesis["solution_path"]
                exec_result = await run_execution_agent(
                    solution_path=solution_path,
                    max_runtime_minutes=max_runtime_minutes,
                    state=state,
                    project_dir=project_dir,
                )

                if exec_result["status"] != "success":
                    logger.warning(
                        f"Execution {exec_result['status']}: {exec_result.get('error_message')}"
                    )
                    state.failed_hypotheses.append(
                        {
                            **state.current_hypothesis,
                            "reason": exec_result["status"],
                            "error": exec_result.get("error_message"),
                        }
                    )
                    state.consecutive_errors += 1
                    state.iteration += 1
                    state.phase = "strategy"
                else:
                    state.consecutive_errors = 0
                    state.current_hypothesis["oof_score"] = exec_result["oof_score"]
                    state.current_hypothesis["runtime_seconds"] = exec_result["runtime_seconds"]
                    logger.info(f"Execution OK, OOF={exec_result['oof_score']:.6f}")
                    state.phase = "validation"

            # ── VALIDATION ───────────────────────────────────────────────────
            elif state.phase == "validation":
                solution_path = state.current_hypothesis["solution_path"]
                oof_score: float = state.current_hypothesis["oof_score"]
                # Derive submission path from solution path convention
                submission_path = solution_path.replace(".py", "_sub.csv")
                # Fallback: look in .gladius/
                version_tag = Path(solution_path).stem  # e.g. "solution_v3"
                submission_path_alt = str(
                    gladius_dir / f"sub_{version_tag.split('_v')[-1]}.csv"
                )

                validation = await run_validation_agent(
                    solution_path=solution_path,
                    oof_score=oof_score,
                    submission_path=submission_path,
                    state=state,
                    project_dir=project_dir,
                )

                # ── ORCHESTRATOR owns state mutation ──────────────────────────
                if validation["is_improvement"]:
                    state.best_oof_score = oof_score
                    state.experiments.append(
                        {
                            "solution_path": solution_path,
                            "oof_score": oof_score,
                            "iteration": state.iteration,
                            "hypothesis": state.current_hypothesis.get("hypothesis"),
                            "runtime_seconds": state.current_hypothesis.get("runtime_seconds"),
                        }
                    )
                    logger.info(
                        f"New best OOF: {oof_score:.6f} "
                        f"(+{validation.get('improvement_delta', 0):.6f})"
                    )

                if (
                    validation["submit"]
                    and state.submission_count < state.max_submissions_per_day
                    and auto_submit
                ):
                    state.submission_count += 1
                    sub_path = validation.get("submission_path") or submission_path
                    state.best_submission_path = sub_path
                    logger.info(f"Submitting: {sub_path}")
                    _submit_to_kaggle(
                        competition_id=state.competition_id,
                        submission_path=sub_path,
                        message=f"iter-{state.iteration} oof={oof_score:.6f}",
                    )

                state.completed_hypotheses.append(state.current_hypothesis)
                state.iteration += 1

                # Trigger ensemble every N iterations if enough good experiments
                if (
                    state.iteration % ensemble_every_n == 0
                    and len(state.experiments) >= 3
                ):
                    state.phase = "ensemble"
                else:
                    state.phase = "strategy"

            # ── ENSEMBLE ─────────────────────────────────────────────────────
            elif state.phase == "ensemble":
                logger.info("Running ensemble agent")
                ensemble_result = await run_ensemble_agent(state, project_dir)

                improved = (
                    state.metric_direction == "maximize"
                    and ensemble_result["oof_score"] > state.best_oof_score
                ) or (
                    state.metric_direction == "minimize"
                    and ensemble_result["oof_score"] < state.best_oof_score
                )

                if improved:
                    state.best_oof_score = ensemble_result["oof_score"]
                    state.best_submission_path = ensemble_result["submission_path"]
                    logger.info(f"Ensemble improved OOF to {ensemble_result['oof_score']:.6f}")

                    if (
                        state.submission_count < state.max_submissions_per_day
                        and auto_submit
                    ):
                        state.submission_count += 1
                        _submit_to_kaggle(
                            competition_id=state.competition_id,
                            submission_path=ensemble_result["submission_path"],
                            message=f"ensemble iter-{state.iteration} oof={ensemble_result['oof_score']:.6f}",
                        )

                state.phase = "strategy"

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
                logger.critical("3 consecutive errors — halting competition loop")
                state.phase = "done"
            else:
                state.phase = "strategy"

        finally:
            store.save(state)

    logger.info(
        f"Competition loop finished. "
        f"iterations={state.iteration}  best_oof={state.best_oof_score:.6f}  "
        f"submissions={state.submission_count}"
    )
    store.close()
    return state


# ── Parallel experiments helper ───────────────────────────────────────────────
async def run_parallel_experiments(
    hypotheses: list[dict],
    state: CompetitionState,
    project_dir: str,
    max_parallel: int = 2,
) -> list[dict]:
    """
    Run up to max_parallel hypotheses concurrently.

    Each hypothesis goes through code → execution.
    Returns list of result dicts (may contain exceptions).
    """
    semaphore = asyncio.Semaphore(max_parallel)

    async def _run_one(hypothesis: dict) -> dict:
        async with semaphore:
            code_output, session_id = await run_code_agent(hypothesis, state, project_dir)
            exec_output = await run_execution_agent(
                solution_path=code_output["solution_path"],
                max_runtime_minutes=60,
                state=state,
                project_dir=project_dir,
            )
            return {
                "hypothesis": hypothesis,
                "code": code_output,
                "execution": exec_output,
                "session_id": session_id,
            }

    tasks = [_run_one(h) for h in hypotheses[:max_parallel]]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ── CLI entry point ───────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gladius",
        description="Autonomous Kaggle competition agent (Claude Agent SDK)",
    )
    p.add_argument("--competition", required=True, help="Kaggle competition slug")
    p.add_argument("--data-dir", required=True, help="Path to downloaded competition data")
    p.add_argument("--project-dir", default=".", help="Working directory (default: cwd)")
    p.add_argument("--metric", default="auc_roc", help="Target metric name")
    p.add_argument(
        "--direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimisation direction",
    )
    p.add_argument("--iterations", type=int, default=20, help="Max iterations")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (ignore existing state.db)",
    )
    p.add_argument(
        "--no-submit",
        action="store_true",
        help="Dry-run: do not actually submit to Kaggle",
    )
    p.add_argument(
        "--ensemble-every",
        type=int,
        default=5,
        help="Run ensemble every N iterations",
    )
    p.add_argument(
        "--max-runtime",
        type=int,
        default=90,
        help="Max minutes per training run",
    )
    return p


async def _amain() -> None:
    args = _build_parser().parse_args()
    await run_competition(
        competition_id=args.competition,
        data_dir=args.data_dir,
        project_dir=args.project_dir,
        target_metric=args.metric,
        metric_direction=args.direction,
        max_iterations=args.iterations,
        resume_from_db=not args.no_resume,
        ensemble_every_n=args.ensemble_every,
        max_runtime_minutes=args.max_runtime,
        auto_submit=not args.no_submit,
    )


def main() -> None:
    """Synchronous entry point (used by pyproject.toml scripts)."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
