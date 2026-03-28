import json
import os
import shutil
from pathlib import Path

from loguru import logger

import gladius.claude_md as claude_md
from gladius.config import LAYOUT, SETTINGS, SYSTEM_PROMPT_PATH as _SYSTEM_PROMPT_PATH
from gladius.state import CompetitionState


def _load_system_prompt() -> str:
    """Load and validate the coordinator system prompt markdown."""
    text = (
        _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        .replace(
            "{{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}}",
            LAYOUT.runtime_experiment_state_relative_path,
        )
        .replace(
            "{{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}}",
            LAYOUT.runtime_data_briefing_relative_path,
        )
        .replace("{{TEAM_LEAD_MEMORY_RELATIVE_PATH}}", LAYOUT.team_lead_memory_relative_path)
        .strip()
    )
    if len(text) < 500:
        raise RuntimeError(
            f"System prompt at {_SYSTEM_PROMPT_PATH} is unexpectedly short ({len(text)} chars)."
        )
    return text


SYSTEM_PROMPT = _load_system_prompt()


TOP_LEVEL_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "TodoWrite",
    "Agent",
    "TaskOutput",
]


_TOPOLOGY_FIRST_STEP: dict[str, str] = {
    "functional": "data-expert → feature-engineer → ml-engineer → evaluator → validator → memory-keeper",
    "two-pizza": "full-stack-coordinator (who decides which specialists to spawn) → validator → memory-keeper",
    "matrix": "ml-engineer (full pipeline) → dual review (team-lead + domain-expert) → evaluator → validator → memory-keeper",
    "autonomous": "N parallel mini-teams each running (data-expert → feature-engineer → ml-engineer → evaluator) → validator → memory-keeper",
    "platform": "platform-layer (data-expert) → product-layer (feature-engineer → ml-engineer) → evaluator → validator → memory-keeper",
}


def make_kickoff_prompt(state: CompetitionState) -> str:
    """Build an iteration-aware kickoff prompt."""
    flow = _TOPOLOGY_FIRST_STEP.get(
        state.topology, "follow the topology defined in CLAUDE.md"
    )

    if state.iteration == 1:
        return (
            "This is the FIRST iteration — no experiments have run yet.\n"
            f"1. Check if `{LAYOUT.runtime_data_briefing_relative_path}` exists. If NOT, delegate to `scout` — "
            "it will explore the data and write the briefing with shapes, distributions, "
            "risks, and strategic angles. If it already exists, skip scout entirely.\n"
            "2. Delegate to `team-lead` to plan a baseline experiment "
            "(team-lead must read DATA_BRIEFING.md, latest EXPERIMENT_STATE_iter*.json, "
            "current EXPERIMENT_STATE.json if present, and MEMORY.md before planning; "
            "team-lead is non-coding and returns planning output only).\n"
            f"3. Then follow the **{state.topology}** topology: {flow}.\n"
            "4. For every downstream specialist, forward the exact relevant team-lead section verbatim "
            "under `## Your Instructions from the Team-Lead` (no paraphrase).\n"
            "Focus on getting a clean, reproducible baseline score above all else."
        )

    best = (
        f"{state.best_oof_score:.6f}"
        if state.best_oof_score is not None
        else (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
    )
    metric_label = state.target_metric or "quality"
    return (
        f"This is iteration {state.iteration}/{state.max_iterations}. "
        f"Current best {metric_label}: **{best}**.\n\n"
        f"1. Skip `scout` — `{LAYOUT.runtime_data_briefing_relative_path}` already exists from iteration 1.\n"
        "2. Delegate to `team-lead` to plan the next experiment "
        "(team-lead must read DATA_BRIEFING.md, latest EXPERIMENT_STATE_iter*.json, "
        "current EXPERIMENT_STATE.json, and MEMORY.md before suggesting the next iteration; "
        "team-lead is non-coding and returns planning output only).\n"
        f"3. Then follow the **{state.topology}** topology: {flow}.\n"
        '   Skip any agent whose EXPERIMENT_STATE.json entry is already `"status": "success"`.'
        "\n4. Forward each downstream specialist's team-lead section verbatim under "
        "`## Your Instructions from the Team-Lead`; do not summarize.\n"
        "5. After validation, have `memory-keeper` update MEMORY.md.\n\n"
        "Avoid any approach listed under 'Failed Approaches' in your context."
    )


# ---------------------------------------------------------------------------
# Orchestrator helper functions (moved from orchestrator.py)
# ---------------------------------------------------------------------------


def build_state(project_dir: Path, cfg: dict) -> CompetitionState:
    """Build initial CompetitionState from config + README.md via claude_md."""
    state = claude_md.write_from_project(project_dir, cfg)
    state.max_iterations = int(cfg.get("max_iterations", state.max_iterations))
    return state


def resolve_start_iteration(max_iterations: int) -> int:
    """Return the first iteration to execute, derived from env var if set."""
    raw = os.getenv(SETTINGS.start_iteration_env_var, "").strip()
    if not raw:
        return 1
    try:
        start_iteration = int(raw)
    except ValueError:
        logger.warning(
            f"Ignoring ${SETTINGS.start_iteration_env_var}={raw!r}: expected integer >= 1."
        )
        return 1

    if start_iteration < 1:
        logger.warning(
            f"Ignoring ${SETTINGS.start_iteration_env_var}={start_iteration}: value must be >= 1."
        )
        return 1
    if start_iteration > max_iterations:
        logger.warning(
            f"${SETTINGS.start_iteration_env_var}={start_iteration} exceeds max_iterations={max_iterations}; "
            f"clamping to {max_iterations}."
        )
        return max_iterations
    return start_iteration


def archive_stale_outputs(project_dir: Path, iteration: int) -> None:
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
            for keep in SETTINGS.persistent_artifacts:
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


def update_state(state: CompetitionState, project_dir: Path) -> None:
    """Read EXPERIMENT_STATE.json written by agents and update CompetitionState."""
    exp_path = LAYOUT.runtime_experiment_state_path(project_dir)
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
REQUIRED_AGENTS = ("team_lead", "ml_engineer", "evaluator")


def incomplete_agents(exp_path: Path) -> list[str]:
    """Return names of required agents that have not yet succeeded."""
    if not exp_path.exists():
        return list(REQUIRED_AGENTS)
    try:
        data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception:
        return list(REQUIRED_AGENTS)
    return [
        k
        for k in REQUIRED_AGENTS
        if not (isinstance(data.get(k), dict) and data[k].get("status") == "success")
    ]


def missing_scout_artifact(state: CompetitionState, project_dir: Path) -> bool:
    """Scout is mandatory in iteration 1 unless briefing already exists."""
    if state.iteration != 1:
        return False
    briefing_path = LAYOUT.runtime_data_briefing_path(project_dir)
    return not briefing_path.exists()


def read_experiment_state_snippet(exp_path: Path) -> str:
    """Return a bounded EXPERIMENT_STATE.json snippet for re-dispatch prompts."""
    if not exp_path.exists():
        return "{}"
    text = exp_path.read_text(encoding="utf-8")
    if len(text) <= SETTINGS.max_state_snippet_chars:
        return text

    # Prefer a compact status-focused summary to limit prompt/token overhead.
    try:
        data = json.loads(text)
    except Exception:
        return text[:SETTINGS.max_state_snippet_chars] + "\n...<truncated>..."

    if not isinstance(data, dict):
        return text[:SETTINGS.max_state_snippet_chars] + "\n...<truncated>..."

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
            for k in REQUIRED_AGENTS
            if not (
                isinstance(data.get(k), dict) and data[k].get("status") == "success"
            )
        },
    }
    if "submission_error" in data:
        summary["submission_error"] = data.get("submission_error")

    compact = json.dumps(summary, ensure_ascii=True, indent=2)
    if len(compact) <= SETTINGS.max_state_snippet_chars:
        return compact
    return compact[:SETTINGS.max_state_snippet_chars] + "\n...<truncated>..."


def build_redispatch_prompt(
    state: CompetitionState,
    exp_path: Path,
    incomplete: list[str],
    *,
    error_hint: str | None = None,
) -> str:
    """Build a strict continuation prompt when coordinator exits early."""
    state_text = read_experiment_state_snippet(exp_path)
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
        f"Current `{LAYOUT.runtime_experiment_state_relative_path}`:\n"
        f"```json\n{state_text}\n```\n\n"
        "Required actions:\n"
        "1. Start with a concise todo task list (3–7 bullets) and keep it updated while working.\n"
        f"2. Read `{LAYOUT.runtime_experiment_state_relative_path}` first.\n"
        "3. Dispatch only pending/failed specialists in correct topology order.\n"
        "4. Skip any specialist already marked `status: success` unless upstream changes require rerun.\n"
        f"5. If dispatching `team-lead`, require it to read `{LAYOUT.runtime_data_briefing_relative_path}`, "
        "latest EXPERIMENT_STATE_iter*.json, and current "
        f"`{LAYOUT.runtime_experiment_state_relative_path}` "
        "before suggesting the next iteration; team-lead is non-coding and must only return planning output.\n"
        "6. For each downstream specialist, include the exact relevant section under "
        "`## Your Instructions from the Team-Lead` verbatim.\n"
        "7. Do not return until all required agents are success and memory-keeper is success."
    )
