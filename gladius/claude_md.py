"""
CLAUDE.md renderer.

Loads src/CLAUDE.md.template, fills every {{placeholder}} with live competition
state, and writes the result to <project_dir>/CLAUDE.md.

Called once per iteration by the orchestrator (replaces the inline f-string in
src/utils/project_setup.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from gladius import team_lead_memory_path
from gladius.topologies._catalog import TOPOLOGY_CATALOG

if TYPE_CHECKING:
    from gladius.state import CompetitionState

_TEMPLATE = Path(__file__).parent / "CLAUDE.md.template"

_STAGNATION_THRESHOLD_METRIC = 0.001
_STAGNATION_THRESHOLD_QUALITY = 3.0


def _phase_guidance(iteration: int, max_iterations: int) -> str:
    """Return a short phase-appropriate guidance block."""
    if max_iterations <= 2:
        # Too few iterations for phasing — skip
        return ""

    if iteration <= 2:
        phase, label, advice = "EARLY", "🟢 Early (baseline)", (
            "> **Priority: get a clean, reproducible baseline.**\n"
            "> - Don't over-engineer features or models.\n"
            "> - Validate the full pipeline end-to-end (data → features → train → OOF → submission).\n"
            "> - A simple model with correct CV is worth more than a complex model with broken validation."
        )
    elif iteration >= max_iterations - 1:
        phase, label, advice = "LATE", "🔴 Late (polish)", (
            "> **Priority: maximise the final score — no risky pivots.**\n"
            "> - Ensemble top-performing models from prior iterations.\n"
            "> - Run HPO on the best model if not done yet.\n"
            "> - Ensure the best submission is saved. Do NOT start a new approach from scratch."
        )
    else:
        phase, label, advice = "MID", "🟡 Mid (explore)", (
            "> **Priority: try diverse approaches to find signal.**\n"
            "> - Feature engineering and model variety are high-impact.\n"
            "> - HPO is worthwhile once you have a strong feature set.\n"
            "> - If stagnating, pivot to a fundamentally different strategy."
        )

    return f"## Iteration Phase: {label}\n\n{advice}"


def render(state: "CompetitionState", project_dir: str) -> str:
    """Return the rendered CLAUDE.md content for the current state."""
    template = _TEMPLATE.read_text(encoding="utf-8")

    topo = TOPOLOGY_CATALOG.get(state.topology)
    topology_section = (
        topo.claude_md_section
        if topo
        else f"**{state.topology}** — no description available."
    )

    # ── metric row ──────────────────────────────────────────────────────────
    if state.target_metric:
        direction_str = (
            "↑ higher is better"
            if state.metric_direction == "maximize"
            else "↓ lower is better"
        )
        metric_row = f"| Target metric | `{state.target_metric}` ({direction_str}) |"
    else:
        metric_row = "| Task type | open-ended (self-assessed quality 0-100) |"

    # ── performance section ──────────────────────────────────────────────────
    if state.target_metric:
        best_oof = (
            f"{state.best_oof_score:.6f}"
            if state.best_oof_score is not None
            else "none yet"
        )
        best_lb = (
            f"{state.best_submission_score:.6f}"
            if state.best_submission_score is not None
            else "none yet"
        )
        threshold_val = state.submission_threshold
        threshold_str = (
            f"{threshold_val:.6f}" if threshold_val is not None else "not set"
        )
        threshold_note = (
            f"> ⛔ **Do not build a submission unless your OOF score beats {threshold_str}.**"
            if threshold_val is not None
            else "> ⚠️ Threshold not set — `WebSearch` the leaderboard and use the current median score as your bar."
        )
        performance_section = (
            f"## Current Best\n\n"
            f"| Metric | Score |\n| --- | --- |\n"
            f"| Best OOF ({state.target_metric}) | **{best_oof}** |\n"
            f"| Best leaderboard | **{best_lb}** |\n"
            f"| Submissions today | {state.submission_count} / {state.max_submissions_per_day} |\n"
            f"| **Minimum submission threshold** | **{threshold_str}** |\n\n"
            f"{threshold_note}"
        )
    else:
        best_quality = (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
        performance_section = (
            f"## Current Progress\n\n"
            f"| Metric | Value |\n| --- | --- |\n"
            f"| Best quality score (0-100) | **{best_quality}** |\n"
            f"| Deliverables submitted | {state.submission_count} |"
        )

    # ── recent experiments ───────────────────────────────────────────────────
    exps = list(reversed(state.experiments[-5:]))
    if exps:
        score_header = "OOF Score" if state.target_metric else "Quality"
        rows = []
        for e in exps:
            score_col = (
                str(e.get("oof_score", "?"))
                if state.target_metric
                else (
                    f"{e.get('quality_score')}/100"
                    if e.get("quality_score") is not None
                    else "?"
                )
            )
            approach = e.get("approach_summary") or e.get("notes", "")[:100]
            rows.append(
                f"| iter {e.get('iteration', '?')} | {score_col} | {approach} |"
            )
        recent_experiments = (
            f"| Iteration | {score_header} | Approach |\n| --- | --- | --- |\n"
            + "\n".join(rows)
        )
    else:
        recent_experiments = "_(none yet)_"

    # ── failed approaches ────────────────────────────────────────────────────
    if state.failed_runs:
        failed_approaches = "\n".join(
            f"- iter {f.get('iteration', '?')}: {f.get('error', '?')[:80]}"
            for f in state.failed_runs[-5:]
        )
    else:
        failed_approaches = "_(none)_"

    # ── stagnation warning ───────────────────────────────────────────────────
    stagnation_block = ""
    if state.target_metric:
        scored = [e for e in state.experiments if e.get("oof_score") is not None]
        score_key, threshold_label, threshold = (
            "oof_score",
            state.target_metric,
            _STAGNATION_THRESHOLD_METRIC,
        )
    else:
        scored = [e for e in state.experiments if e.get("quality_score") is not None]
        score_key, threshold_label, threshold = (
            "quality_score",
            "quality",
            _STAGNATION_THRESHOLD_QUALITY,
        )

    if len(scored) >= 3:
        last3 = [e[score_key] for e in scored[-3:]]
        span = max(last3) - min(last3)
        if span < threshold:
            stagnation_block = (
                f"## ⚠️ STAGNATION WARNING\n\n"
                f"> The last **{len(last3)} experiments** moved the {threshold_label} score by only\n"
                f"> **{span:.4f}** (threshold: {threshold}). Incremental tweaks are not working.\n>\n"
                f"> **Team lead: stop tuning. Go back to first principles.**\n"
                f"> - Re-examine the task description and deliverables.\n"
                f"> - Try a completely different approach or architecture.\n"
                f"> - WebSearch for breakthrough techniques specific to this task type."
            )

    # ── data / submission sections ───────────────────────────────────────────
    if state.target_metric:
        data_section = (
            f"## Data Files\n\n"
            f"```bash\nls {state.data_dir}\n```\n\n"
            f"Standard files to expect:\n"
            f"- `train.csv` — training set (target column present)\n"
            f"- `test.csv` — test set (no target column)\n"
            f"- `sample_submission.csv` — submission format template"
        )
        submission_section = (
            "## Submission Rules\n\n"
            "1. **Gate:** Only build a submission once your OOF score beats the `Minimum submission threshold` shown above.\n"
            '   - If threshold is "not set", `WebSearch` the leaderboard first and use the current median score as your bar.\n'
            "2. Load `sample_submission.csv` to get the exact submission format.\n"
            "3. Your submission must match its columns and row count exactly.\n"
            "4. Save to `submissions/submission.csv`.\n"
            "5. Report the path in `submission_file` in your output."
        )
    else:
        data_section = ""
        submission_section = (
            "## Deliverable Rules\n\n"
            "1. Read README.md thoroughly to understand what deliverable is required.\n"
            "2. Build and verify the deliverable works end-to-end.\n"
            "3. Package it as described in README.md (zip, binary, URL file, etc.).\n"
            "4. Report the path in `submission_file` in your output.\n"
            "5. Self-assess quality 0-100: rate completeness and correctness against README requirements."
        )

    memory_path = str(team_lead_memory_path(project_dir).resolve())

    # ── data briefing (produced by scout in iteration 1) ─────────────────────
    briefing_path = Path(project_dir) / ".claude" / "DATA_BRIEFING.md"
    if briefing_path.is_file():
        data_briefing_section = (
            "## Data Briefing\n\n"
            "> Produced by the scout agent. All agents may read this for data context.\n\n"
            + briefing_path.read_text(encoding="utf-8").strip()
        )
    else:
        data_briefing_section = (
            "## Data Briefing\n\n"
            "_(not yet available — the scout agent will produce `.claude/DATA_BRIEFING.md` "
            "in iteration 1)_"
        )

    # ── phase guidance ───────────────────────────────────────────────────────
    phase_guidance = _phase_guidance(state.iteration, state.max_iterations)

    # ── substitute ───────────────────────────────────────────────────────────
    replacements = {
        "competition_id": state.competition_id,
        "metric_row": metric_row,
        "data_dir": str(state.data_dir),
        "output_dir": str(state.output_dir),
        "iteration": str(state.iteration),
        "max_iterations": str(state.max_iterations),
        "topology": state.topology,
        "topology_section": topology_section,
        "performance_section": performance_section,
        "recent_experiments": recent_experiments,
        "failed_approaches": failed_approaches,
        "stagnation_block": stagnation_block,
        "phase_guidance": phase_guidance,
        "data_section": data_section,
        "data_briefing_section": data_briefing_section,
        "submission_section": submission_section,
        "memory_path": memory_path,
    }

    content = template
    for key, value in replacements.items():
        content = content.replace("{{" + key + "}}", value)

    return content


def write(state: "CompetitionState", project_dir: str) -> None:
    """Render and write CLAUDE.md to project_dir."""
    path = Path(project_dir) / "CLAUDE.md"
    path.write_text(render(state, project_dir), encoding="utf-8")


def _parse_readme(root: Path) -> tuple[dict, str]:
    """Parse README.md frontmatter (YAML between ---) and body text."""
    readme = root / "README.md"
    if not readme.is_file():
        return {}, ""
    text = readme.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if m:
        front = yaml.safe_load(m.group(1)) or {}
        body = m.group(2).strip()
    else:
        front, body = {}, text.strip()
    return front, body


def write_from_project(root: Path, cfg: dict) -> "CompetitionState":
    """Build CompetitionState from config + README.md, write CLAUDE.md, and return the state."""
    from gladius.state import CompetitionState

    front, _body = _parse_readme(root)

    # README frontmatter can override/supplement cfg (e.g. submission_threshold)
    metric = cfg.get("metric") or front.get("metric") or None
    direction = cfg.get("direction") or front.get("direction") or None
    data_dir = cfg.get("data_dir") or str(root / front.get("data_dir", "data"))
    topology = cfg.get("topology", "functional")
    submission_threshold = front.get("submission_threshold") or cfg.get(
        "submission_threshold"
    )
    max_sub_day = int(
        front.get("max_submissions_per_day", cfg.get("max_submissions_per_day", 5))
    )

    state = CompetitionState(
        competition_id=cfg["competition_id"],
        data_dir=data_dir,
        output_dir=str(root / "artifacts"),
        target_metric=metric,
        metric_direction=direction,
        topology=topology,
        submission_threshold=submission_threshold,
        max_submissions_per_day=max_sub_day,
    )
    write(state, str(root))
    print("  claude → CLAUDE.md")
    return state
