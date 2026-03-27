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
import shutil
import time
import uuid
from pathlib import Path

from loguru import logger

import gladius.claude_md as claude_md
from gladius import (
    RUNTIME_DATA_BRIEFING_RELATIVE_PATH,
    RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH,
    runtime_data_briefing_path,
    runtime_experiment_state_path,
    state_db_path,
)
from gladius._orchestrator_helper import SYSTEM_PROMPT as _SYSTEM_PROMPT
from gladius._orchestrator_helper import TOP_LEVEL_TOOLS as _TOP_LEVEL_TOOLS
from gladius._orchestrator_helper import make_kickoff_prompt as _make_kickoff_prompt
from gladius.config import MAX_CONSECUTIVE_ERRORS as _MAX_CONSECUTIVE_ERRORS
from gladius.config import MAX_REDISPATCH as _MAX_REDISPATCH
from gladius.config import MAX_STATE_SNIPPET_CHARS as _MAX_STATE_SNIPPET_CHARS
from gladius.config import MAX_TURNS as _DEFAULT_MAX_TURNS
from gladius.config import PERSISTENT_ARTIFACTS as _PERSISTENT_ARTIFACTS
from gladius.config import START_ITERATION_ENV_VAR as _START_ITERATION_ENV_VAR
from gladius.config import load_project_env
from gladius.db.store import StateStore
from gladius.langsmith_tracing import init_langsmith_tracing
from gladius.logging_setup import configure_logging
from gladius.process_cleanup import (
    cleanup_orphan_processes,
    should_cleanup_orphan_processes,
)
from gladius.project_setup import load_competition_config
from gladius.roles.agent_runner import run_agent
from gladius.session_replay import export_recent_session_diagnostics
from gladius.state import CompetitionState


def _build_state(project_dir: Path, cfg: dict) -> CompetitionState:
    """Build initial CompetitionState from config + README.md via claude_md."""
    state = claude_md.write_from_project(project_dir, cfg)
    state.max_iterations = int(cfg.get("max_iterations", state.max_iterations))
    return state


def _resolve_start_iteration(max_iterations: int) -> int:
    """Return the first iteration to execute, derived from env var if set."""
    raw = os.getenv(_START_ITERATION_ENV_VAR, "").strip()
    if not raw:
        return 1
    try:
        start_iteration = int(raw)
    except ValueError:
        logger.warning(
            f"Ignoring ${_START_ITERATION_ENV_VAR}={raw!r}: expected integer >= 1."
        )
        return 1

    if start_iteration < 1:
        logger.warning(
            f"Ignoring ${_START_ITERATION_ENV_VAR}={start_iteration}: value must be >= 1."
        )
        return 1
    if start_iteration > max_iterations:
        logger.warning(
            f"${_START_ITERATION_ENV_VAR}={start_iteration} exceeds max_iterations={max_iterations}; "
            f"clamping to {max_iterations}."
        )
        return max_iterations
    return start_iteration


def _archive_stale_outputs(project_dir: Path, iteration: int) -> None:
    """Move stale artifacts/ and agent logs to iteration-stamped archives.

    Prevents agents from mistaking previous-iteration outputs (oof.npy,
    model_f*.bin, train.log) for current-iteration results.
    ``best_params.json`` is copied forward since HPO results are reusable.
    ``logs/gladius.log`` is never touched — it's the orchestrator's own log.
    """
    prev = iteration - 1
    if prev < 1:
        return

    # --- artifacts/: move the whole directory ---
    art_dir = project_dir / "artifacts"
    if art_dir.is_dir() and any(art_dir.iterdir()):
        archive_dir = project_dir / f"artifacts_iter{prev}"
        try:
            shutil.move(str(art_dir), str(archive_dir))
            art_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Archived artifacts/ → {archive_dir.name}/")

            # Copy forward persistent artifacts
            for keep in _PERSISTENT_ARTIFACTS:
                archived = archive_dir / keep
                if archived.is_file():
                    shutil.copy2(str(archived), str(art_dir / keep))
                    logger.debug(f"Carried forward {keep} from iter {prev}")
        except Exception as exc:
            logger.warning(
                f"Could not archive artifacts for iter {prev}: {exc}. "
                "Continuing with existing directory state."
            )

    # --- logs/: move only agent-produced files, leave gladius.log in place ---
    logs_dir = project_dir / "logs"
    if logs_dir.is_dir():
        agent_logs = [
            f for f in logs_dir.iterdir() if f.is_file() and f.name != "gladius.log"
        ]
        if agent_logs:
            archive_dir = project_dir / f"logs_iter{prev}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            for f in agent_logs:
                try:
                    shutil.move(str(f), str(archive_dir / f.name))
                except Exception as exc:
                    logger.warning(f"Could not archive log {f.name!r}: {exc}")
            logger.debug(
                f"Archived {len(agent_logs)} log file(s) → {archive_dir.name}/"
            )


def _update_state(state: CompetitionState, project_dir: Path) -> None:
    """Read EXPERIMENT_STATE.json written by agents and update CompetitionState."""
    exp_path = runtime_experiment_state_path(project_dir)
    if not exp_path.exists():
        return

    try:
        data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not parse EXPERIMENT_STATE.json: {exc}")
        return

    # Extract OOF score — evaluator is authoritative, ml_engineer as fallback
    # Guard against agents writing a non-dict value (e.g. an error string) for these keys.
    _raw_evaluator = data.get("evaluator", {})
    _raw_ml_eng = data.get("ml_engineer", {})
    _raw_team_lead = data.get("team_lead", {})
    evaluator = _raw_evaluator if isinstance(_raw_evaluator, dict) else {}
    ml_eng = _raw_ml_eng if isinstance(_raw_ml_eng, dict) else {}
    team_lead = _raw_team_lead if isinstance(_raw_team_lead, dict) else {}
    if not isinstance(_raw_evaluator, dict) and _raw_evaluator:
        logger.warning(
            f"EXPERIMENT_STATE.json: 'evaluator' is not a dict ({type(_raw_evaluator).__name__!r}), ignoring."
        )
    if not isinstance(_raw_ml_eng, dict) and _raw_ml_eng:
        logger.warning(
            f"EXPERIMENT_STATE.json: 'ml_engineer' is not a dict ({type(_raw_ml_eng).__name__!r}), ignoring."
        )
    if not isinstance(_raw_team_lead, dict) and _raw_team_lead:
        logger.warning(
            f"EXPERIMENT_STATE.json: 'team_lead' is not a dict ({type(_raw_team_lead).__name__!r}), ignoring."
        )
    oof_score: float | None = evaluator.get("oof_score") or ml_eng.get("oof_score")
    quality_score: float | None = ml_eng.get("quality_score")
    metric = evaluator.get("metric") or state.target_metric
    notes = ml_eng.get("notes", "")
    solution_files = ml_eng.get("solution_files", [])
    submission_file = ml_eng.get("submission_file")

    entry = {
        "iteration": state.iteration,
        "oof_score": oof_score,
        "quality_score": quality_score,
        "metric": metric,
        "notes": notes,
        "approach_summary": team_lead.get("approach_summary", ""),
        "solution_files": solution_files,
        "approach": team_lead.get("approach_summary") or ml_eng.get("notes", ""),
    }
    state.experiments.append(entry)

    # Update best scores
    if oof_score is not None:
        if state.best_oof_score is None:
            state.best_oof_score = oof_score
            if submission_file:
                state.best_submission_path = submission_file
        elif state.metric_direction == "maximize" and oof_score >= state.best_oof_score:
            state.best_oof_score = oof_score
            if submission_file:
                state.best_submission_path = submission_file
        elif state.metric_direction == "minimize" and oof_score <= state.best_oof_score:
            state.best_oof_score = oof_score
            if submission_file:
                state.best_submission_path = submission_file

    if quality_score is not None and (
        state.best_quality_score is None or quality_score > state.best_quality_score
    ):
        state.best_quality_score = quality_score

    # Stop signal written by the coordinator after validator returns stop=True
    if data.get("done"):
        logger.info("Stop signal received from agent (done=true in EXPERIMENT_STATE).")
        state.done = True


# Agents required to have status=success for an iteration to be considered complete.
# team_lead is required so coordinators cannot skip planning and jump straight to execution.
# memory_keeper is excluded — it writes MEMORY.md, not EXPERIMENT_STATE.json, so including it
# here would cause _incomplete_agents() to always return ["memory_keeper"] and redispatch forever.
_REQUIRED_AGENTS = ("team_lead", "ml_engineer", "evaluator")


def _incomplete_agents(exp_path: Path) -> list[str]:
    """Return names of required agents that have not yet succeeded."""
    if not exp_path.exists():
        return list(_REQUIRED_AGENTS)
    try:
        data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception:
        return list(_REQUIRED_AGENTS)
    return [
        k
        for k in _REQUIRED_AGENTS
        if not (isinstance(data.get(k), dict) and data[k].get("status") == "success")
    ]


def _missing_scout_artifact(state: CompetitionState, project_dir: Path) -> bool:
    """Scout is mandatory in iteration 1 unless briefing already exists."""
    if state.iteration != 1:
        return False
    briefing_path = runtime_data_briefing_path(project_dir)
    return not briefing_path.exists()


def _read_experiment_state_snippet(exp_path: Path) -> str:
    """Return a bounded EXPERIMENT_STATE.json snippet for re-dispatch prompts."""
    if not exp_path.exists():
        return "{}"
    text = exp_path.read_text(encoding="utf-8")
    if len(text) <= _MAX_STATE_SNIPPET_CHARS:
        return text

    # Prefer a compact status-focused summary to limit prompt/token overhead.
    try:
        data = json.loads(text)
    except Exception:
        return text[:_MAX_STATE_SNIPPET_CHARS] + "\n...<truncated>..."

    if not isinstance(data, dict):
        return text[:_MAX_STATE_SNIPPET_CHARS] + "\n...<truncated>..."

    summary: dict[str, object] = {
        "done": bool(data.get("done", False)),
        "pending_agents": {
            k: (
                {
                    kk: vv
                    for kk, vv in data[k].items()
                    if kk
                    in {
                        "status",
                        "error",
                        "reason",
                        "oof_score",
                        "quality_score",
                        "metric",
                        "submission_file",
                    }
                }
                if isinstance(data.get(k), dict)
                else data.get(k)
            )
            for k in _REQUIRED_AGENTS
            if not (
                isinstance(data.get(k), dict) and data[k].get("status") == "success"
            )
        },
    }
    if "submission_error" in data:
        summary["submission_error"] = data.get("submission_error")

    compact = json.dumps(summary, ensure_ascii=True, indent=2)
    if len(compact) <= _MAX_STATE_SNIPPET_CHARS:
        return compact
    return compact[:_MAX_STATE_SNIPPET_CHARS] + "\n...<truncated>..."


def _build_redispatch_prompt(
    state: CompetitionState,
    exp_path: Path,
    incomplete: list[str],
    *,
    error_hint: str | None = None,
) -> str:
    """Build a strict continuation prompt when coordinator exits early."""
    state_text = _read_experiment_state_snippet(exp_path)
    pending = ", ".join(incomplete)
    error_section = (
        f"\n**Previous attempt failed with this error — fix it before retrying:**\n"
        f"```\n{error_hint}\n```\n"
        if error_hint
        else ""
    )
    return (
        "You returned before this iteration pipeline was complete. Continue immediately.\n\n"
        + error_section
        + f"Iteration context: {state.iteration}/{state.max_iterations}, topology={state.topology}.\n"
        f"Pending required agents (non-success): {pending}.\n\n"
        f"Current `{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}`:\n"
        f"```json\n{state_text}\n```\n\n"
        "Required actions:\n"
        "1. Start with a concise todo task list (3–7 bullets) and keep it updated while working.\n"
        f"2. Read `{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}` first.\n"
        "3. Dispatch only pending/failed specialists in correct topology order.\n"
        "4. Skip any specialist already marked `status: success` unless upstream changes require rerun.\n"
        f"5. If dispatching `team-lead`, require it to read `{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}`, "
        "latest EXPERIMENT_STATE_iter*.json, and current "
        f"`{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}` "
        "before suggesting the next iteration; team-lead is non-coding and must only return planning output.\n"
        "6. For each downstream specialist, include the exact relevant section under "
        "`## Your Instructions from the Team-Lead` verbatim.\n"
        "7. Do not return until all required agents are success and memory-keeper is success."
    )


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
    store = StateStore(str(state_db_path(project_dir)))

    try:
        # Ensure GLADIUS_MODEL / GLADIUS_SMALL_MODEL are in the environment so
        # validate_runtime_invocation() and get_runtime_model() can find them.
        # project.yaml takes precedence; existing env vars are only used as fallback.
        if cfg.get("model"):
            os.environ.setdefault("GLADIUS_MODEL", cfg["model"])
        if cfg.get("small_model") and cfg["small_model"] != "inherit":
            os.environ.setdefault("GLADIUS_SMALL_MODEL", cfg["small_model"])

        state = _build_state(project_dir, cfg)
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

        start_iteration = _resolve_start_iteration(state.max_iterations)
        if start_iteration > 1:
            # The loop increments state.iteration at the top of each cycle.
            state.iteration = start_iteration - 1
            logger.info(
                f"Resuming run from iteration {start_iteration} via ${_START_ITERATION_ENV_VAR}."
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
                    exp_path = runtime_experiment_state_path(project_dir)
                    if exp_path.exists():
                        archive = exp_path.with_name(
                            f"EXPERIMENT_STATE_iter{state.iteration - 1}.json"
                        )
                        exp_path.rename(archive)
                        logger.debug(f"Archived previous EXPERIMENT_STATE → {archive.name}")

                    # Archive stale artifacts and logs so agents can't confuse old outputs
                    # with current iteration results.  best_params.json is preserved
                    # (intentionally reusable across iterations).
                    _archive_stale_outputs(project_dir, state.iteration)

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
                    # pipeline), re-dispatch it up to _MAX_REDISPATCH times with current state.
                    prompt = kickoff
                    incomplete: list[str] = []
                    _iteration_error = False

                    def _trace_sink(payload: dict[str, object]) -> None:
                        store.record_agent_run(
                            run_id=run_id,
                            iteration=state.iteration,
                            topology=state.topology,
                            **payload,
                        )

                    for _attempt in range(1 + _MAX_REDISPATCH):
                        attempt_no = _attempt + 1
                        attempts_used = attempt_no
                        total_attempts = _MAX_REDISPATCH + 1
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
                                max_turns=max_turns or _DEFAULT_MAX_TURNS,
                                max_retries=_MAX_CONSECUTIVE_ERRORS,
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
                            logger.error(f"Iteration {state.iteration} agent error: {exc}")
                            # Check whether the pipeline completed despite the exception
                            # (e.g. a forbidden-tool violation detected post-hoc).
                            exp_path = runtime_experiment_state_path(project_dir)
                            incomplete = _incomplete_agents(exp_path)
                            if _missing_scout_artifact(state, project_dir):
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
                            if _attempt < _MAX_REDISPATCH:
                                logger.warning(
                                    f"Iteration {state.iteration}: agent error on attempt {attempt_no}/{total_attempts} "
                                    f"— re-dispatching. agents still pending: {incomplete}"
                                )
                                redispatches += 1
                                prompt = _build_redispatch_prompt(
                                    state, exp_path, incomplete,
                                    error_hint=str(exc) if exc else None,
                                )
                                continue
                            # All redispatch attempts exhausted — count as one iteration failure.
                            state.consecutive_errors += 1
                            if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                                logger.error(
                                    f"{_MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping run."
                                )
                                state.last_stop_reason = "consecutive_errors"
                                state.done = True
                            _iteration_error = True
                            break

                        # Check whether the pipeline actually completed.
                        exp_path = runtime_experiment_state_path(project_dir)
                        incomplete = _incomplete_agents(exp_path)
                        if _missing_scout_artifact(state, project_dir):
                            incomplete = ["scout", *incomplete]
                        if not incomplete:
                            break
                        if _attempt < _MAX_REDISPATCH:
                            logger.warning(
                                f"Iteration {state.iteration}: orchestrator returned early — "
                                f"agents still pending: {incomplete}. Re-dispatching (attempt {attempt_no + 1}/{total_attempts})."
                            )
                            redispatches += 1
                            prompt = _build_redispatch_prompt(state, exp_path, incomplete)

                    if incomplete and not _iteration_error:
                        logger.error(
                            f"Iteration {state.iteration}: required agents still incomplete after "
                            f"{_MAX_REDISPATCH + 1} orchestrator attempt(s): {incomplete}"
                        )
                        state.consecutive_errors += 1
                        state.error_log.append(
                            {
                                "iteration": state.iteration,
                                "error": f"incomplete_pipeline: {incomplete}",
                            }
                        )
                        if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                            logger.error(
                                f"{_MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping run."
                            )
                            state.last_stop_reason = "consecutive_errors"
                            state.done = True

                    _update_state(state, project_dir)
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
            output_file=project_dir / ".gladius" / "runtime" / "session_diagnostics.json",
            limit_sessions=3,
            limit_messages=30,
        )
    finally:
        store.close()
