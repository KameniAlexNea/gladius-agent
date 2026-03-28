"""
Agent launcher — iterative competition loop.

Each iteration:
  1. Renders CLAUDE.md from current CompetitionState.
  2. Runs the top-level coordinator agent (fresh session per iteration).
        3. Reads runtime EXPERIMENT_STATE.json to extract scores and update state.
  4. Stops when max_iterations reached, a stop sentinel is found, or
     3 consecutive agent errors occur.

CLAUDE.md is automatically injected into the agent context — no explicit read needed.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from loguru import logger

import gladius.claude_md as claude_md
from gladius.config import LAYOUT, SETTINGS, load_project_env
from gladius.db.store import StateStore
from gladius.project_setup import load_competition_config
from gladius.roles.agent_runner import run_agent
from gladius.state import CompetitionState
from gladius.utilities._orchestrator_helper import SYSTEM_PROMPT as _SYSTEM_PROMPT
from gladius.utilities._orchestrator_helper import TOP_LEVEL_TOOLS as _TOP_LEVEL_TOOLS
from gladius.utilities._orchestrator_helper import (
    archive_stale_outputs,
    build_redispatch_prompt,
    build_state,
    incomplete_agents,
)
from gladius.utilities._orchestrator_helper import (
    make_kickoff_prompt as _make_kickoff_prompt,
)
from gladius.utilities._orchestrator_helper import (
    missing_scout_artifact,
    resolve_start_iteration,
    update_state,
)
from gladius.utilities.langsmith_tracing import init_langsmith_tracing
from gladius.utilities.logging_setup import configure_logging
from gladius.utilities.process_cleanup import (
    cleanup_orphan_processes,
    should_cleanup_orphan_processes,
)
from gladius.utilities.session_replay import export_recent_session_diagnostics


async def run_competition(
    competition_dir: str,
    max_turns: int | None = None,
    max_iterations: int | None = None,
    config_path: str | None = None,
) -> None:
    cfg = load_competition_config(competition_dir, config_path=config_path)
    project_dir = Path(competition_dir)
    loaded_env = load_project_env(project_dir)
    configure_logging(project_dir)
    if loaded_env is not None:
        logger.info(f"Loaded project env: {loaded_env}")
    run_id = f"{cfg.get('competition_id', 'run')}-{uuid.uuid4().hex[:8]}"
    store = StateStore(str(LAYOUT.state_db_path(project_dir)))

    try:
        # Ensure GLADIUS_MODEL / GLADIUS_SMALL_MODEL are in the environment so
        # validate_runtime_invocation() and get_runtime_model() can find them.
        # project.yaml takes precedence; existing env vars are only used as fallback.
        if cfg.get("model"):
            os.environ.setdefault("GLADIUS_MODEL", cfg["model"])
        if cfg.get("small_model") and cfg["small_model"] != "inherit":
            os.environ.setdefault("GLADIUS_SMALL_MODEL", cfg["small_model"])

        state = build_state(project_dir, cfg)
        if max_iterations is not None:
            state.max_iterations = max_iterations

        loaded_state = store.load()
        if isinstance(loaded_state, CompetitionState) and (
            loaded_state.competition_id == state.competition_id
        ):
            state.team_session_ids.update(loaded_state.team_session_ids)
            if state.team_session_ids:
                logger.info(
                    f"Loaded {len(state.team_session_ids)} persisted session id(s) from state DB."
                )

        init_langsmith_tracing()

        start_iteration = resolve_start_iteration(state.max_iterations)
        if start_iteration > 1:
            # The loop increments state.iteration at the top of each cycle.
            state.iteration = start_iteration - 1
            logger.info(
                f"Resuming run from iteration {start_iteration} via ${SETTINGS.start_iteration_env_var}."
            )

        logger.info(
            f"Starting competition run: {cfg['competition_id']} "
            f"(max_iterations={state.max_iterations})"
        )
        store.record_event(
            iteration=state.iteration,
            topology=state.topology,
            event="run_start",
            detail=json.dumps(
                {
                    "run_id": run_id,
                    "competition_id": cfg.get("competition_id"),
                    "max_iterations": state.max_iterations,
                }
            ),
        )

        with logger.contextualize(run_id=run_id, agent="orchestrator", attempt="-"):
            while not state.done and state.iteration < state.max_iterations:
                iteration_started = time.monotonic()
                redispatches = 0
                attempts_used = 0
                state.iteration += 1

                with logger.contextualize(iteration=str(state.iteration), attempt="-"):
                    store.record_event(
                        iteration=state.iteration,
                        topology=state.topology,
                        event="iteration_start",
                        detail=json.dumps({"run_id": run_id}),
                    )

                    if should_cleanup_orphan_processes():
                        cleanup_orphan_processes(project_dir)

                    # Archive EXPERIMENT_STATE from the previous iteration so agents start fresh
                    exp_path = LAYOUT.runtime_experiment_state_path(project_dir)
                    if exp_path.exists():
                        archive = exp_path.with_name(
                            f"EXPERIMENT_STATE_iter{state.iteration - 1}.json"
                        )
                        exp_path.rename(archive)
                        logger.debug(
                            f"Archived previous EXPERIMENT_STATE → {archive.name}"
                        )

                    # Archive stale artifacts and logs so agents can't confuse old outputs
                    # with current iteration results.  best_params.json is preserved
                    # (intentionally reusable across iterations).
                    archive_stale_outputs(project_dir, state.iteration)

                    # Refresh CLAUDE.md with current state before each iteration
                    claude_md.write(state, str(project_dir))

                    logger.info(
                        f"Iteration {state.iteration}/{state.max_iterations} — "
                        f"best={state.best_oof_score or state.best_quality_score or 'none'}"
                    )

                    kickoff = _make_kickoff_prompt(state)

                    # Reset consecutive error counter at the start of each iteration — the
                    # threshold is meant to stop a genuinely stuck run, not to penalise the
                    # next iteration for a previous one's transient failure.
                    state.consecutive_errors = 0

                    # Agents that must reach status=success for the iteration to count.
                    # If the orchestrator returns early (text response without finishing the
                    # pipeline), re-dispatch it up to SETTINGS.max_redispatch times with current state.
                    prompt = kickoff
                    incomplete: list[str] = []
                    _iteration_error = False

                    _SKIP_TRACE_EVENTS = {"stream_event"}

                    def _trace_sink(payload: dict[str, object]) -> None:
                        if payload.get("event") in _SKIP_TRACE_EVENTS:
                            return
                        store.record_agent_run(
                            run_id=run_id,
                            iteration=state.iteration,
                            topology=state.topology,
                            **payload,
                        )

                    for _attempt in range(1 + SETTINGS.max_redispatch):
                        attempt_no = _attempt + 1
                        attempts_used = attempt_no
                        total_attempts = SETTINGS.max_redispatch + 1
                        logger.info(
                            f"Iteration {state.iteration}: orchestrator attempt {attempt_no}/{total_attempts}"
                        )
                        try:
                            resume_session = state.team_session_ids.get("gladius")
                            _, session_id = await run_agent(
                                agent_name="gladius",
                                prompt=prompt,
                                system_prompt=_SYSTEM_PROMPT,
                                allowed_tools=_TOP_LEVEL_TOOLS,
                                output_schema=None,
                                cwd=str(project_dir),
                                resume=resume_session,
                                max_turns=max_turns or SETTINGS.max_turns,
                                max_retries=SETTINGS.max_consecutive_errors,
                                trace_sink=_trace_sink,
                                trace_context={
                                    "run_id": run_id,
                                    "iteration": state.iteration,
                                    "attempt": attempt_no,
                                },
                                enable_trace_hooks=True,
                            )

                            if isinstance(session_id, str) and session_id:
                                state.team_session_ids["gladius"] = session_id
                                logger.info(
                                    f"Iteration {state.iteration}: orchestrator session={session_id[:12]}…"
                                )
                            state.consecutive_errors = 0
                        except Exception as exc:
                            logger.error(
                                f"Iteration {state.iteration} agent error: {exc}"
                            )
                            # Check whether the pipeline completed despite the exception
                            # (e.g. a forbidden-tool violation detected post-hoc).
                            exp_path = LAYOUT.runtime_experiment_state_path(project_dir)
                            incomplete = incomplete_agents(exp_path)
                            if missing_scout_artifact(state, project_dir):
                                incomplete = ["scout", *incomplete]
                            if not incomplete:
                                logger.warning(
                                    f"Iteration {state.iteration}: pipeline complete despite agent "
                                    f"error — not counting as failure."
                                )
                                state.error_log.append(
                                    {
                                        "iteration": state.iteration,
                                        "error": str(exc),
                                        "pipeline_complete": True,
                                    }
                                )
                                break
                            state.error_log.append(
                                {
                                    "iteration": state.iteration,
                                    "attempt": _attempt + 1,
                                    "error": str(exc),
                                }
                            )
                            if _attempt < SETTINGS.max_redispatch:
                                logger.warning(
                                    f"Iteration {state.iteration}: agent error on attempt {attempt_no}/{total_attempts} "
                                    f"— re-dispatching. agents still pending: {incomplete}"
                                )
                                redispatches += 1
                                prompt = build_redispatch_prompt(
                                    state,
                                    exp_path,
                                    incomplete,
                                    error_hint=str(exc) if exc else None,
                                )
                                continue
                            # All redispatch attempts exhausted — count as one iteration failure.
                            state.consecutive_errors += 1
                            if (
                                state.consecutive_errors
                                >= SETTINGS.max_consecutive_errors
                            ):
                                logger.error(
                                    f"{SETTINGS.max_consecutive_errors} consecutive errors — stopping run."
                                )
                                state.last_stop_reason = "consecutive_errors"
                                state.done = True
                            _iteration_error = True
                            break

                        # Check whether the pipeline actually completed.
                        exp_path = LAYOUT.runtime_experiment_state_path(project_dir)
                        incomplete = incomplete_agents(exp_path)
                        if missing_scout_artifact(state, project_dir):
                            incomplete = ["scout", *incomplete]
                        if not incomplete:
                            break
                        if _attempt < SETTINGS.max_redispatch:
                            logger.warning(
                                f"Iteration {state.iteration}: orchestrator returned early — "
                                f"agents still pending: {incomplete}. Re-dispatching (attempt {attempt_no + 1}/{total_attempts})."
                            )
                            redispatches += 1
                            prompt = build_redispatch_prompt(
                                state, exp_path, incomplete
                            )

                    if incomplete and not _iteration_error:
                        logger.error(
                            f"Iteration {state.iteration}: required agents still incomplete after "
                            f"{SETTINGS.max_redispatch + 1} orchestrator attempt(s): {incomplete}"
                        )
                        state.consecutive_errors += 1
                        state.error_log.append(
                            {
                                "iteration": state.iteration,
                                "error": f"incomplete_pipeline: {incomplete}",
                            }
                        )
                        if state.consecutive_errors >= SETTINGS.max_consecutive_errors:
                            logger.error(
                                f"{SETTINGS.max_consecutive_errors} consecutive errors — stopping run."
                            )
                            state.last_stop_reason = "consecutive_errors"
                            state.done = True

                    update_state(state, project_dir)
                    store.save(state)

                    elapsed_ms = int((time.monotonic() - iteration_started) * 1000)
                    logger.info(
                        f"Iteration {state.iteration} summary: attempts={attempts_used}, "
                        f"redispatches={redispatches}, pending={len(incomplete)}, "
                        f"errors={state.consecutive_errors}, elapsed_ms={elapsed_ms}"
                    )
                    store.record_event(
                        iteration=state.iteration,
                        topology=state.topology,
                        event="iteration_summary",
                        detail=json.dumps(
                            {
                                "attempts": attempts_used,
                                "redispatches": redispatches,
                                "pending": incomplete,
                                "elapsed_ms": elapsed_ms,
                            }
                        ),
                    )

                    if _iteration_error:
                        export_recent_session_diagnostics(
                            project_dir=project_dir,
                            output_file=project_dir
                            / ".gladius"
                            / "runtime"
                            / f"session_diagnostics_iter{state.iteration}.json",
                            limit_sessions=5,
                            limit_messages=50,
                        )

        reason = state.last_stop_reason or (
            "done_signal" if state.done else "max_iterations"
        )
        logger.info(
            f"Run complete — {state.iteration} iteration(s), "
            f"best={state.best_oof_score or state.best_quality_score or 'none'}, "
            f"stop_reason={reason}"
        )
        store.record_event(
            iteration=state.iteration,
            topology=state.topology,
            event="run_complete",
            detail=json.dumps(
                {
                    "run_id": run_id,
                    "reason": reason,
                    "best_oof_score": state.best_oof_score,
                    "best_quality_score": state.best_quality_score,
                }
            ),
        )
        store.save(state)
        export_recent_session_diagnostics(
            project_dir=project_dir,
            output_file=project_dir
            / ".gladius"
            / "runtime"
            / "session_diagnostics.json",
            limit_sessions=3,
            limit_messages=30,
        )
    finally:
        store.close()
