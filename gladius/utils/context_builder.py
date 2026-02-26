import json
from datetime import datetime, timezone
from pathlib import Path

from gladius.state import GraphState

KNOWLEDGE_PATH = Path("state/knowledge.json")
EXPERIMENTS_DIR = Path("state/experiments")
SCRIPTS_DIR = Path("state/scripts")
VERSIONS_DIR = Path("state/versions")


class ContextBuilder:
    @staticmethod
    def build_strategy_context(state: GraphState) -> dict:
        knowledge = _load_knowledge()
        competition = state.get("competition", {})
        all_experiments = _load_all_experiments()
        total_days = competition.get("days_remaining", 0) + _days_elapsed(all_experiments)
        return {
            "competition": competition,
            "top10_by_oof": _top_n_by_oof(all_experiments, 10),
            "last5_chronological": all_experiments[-5:],
            "knowledge_summary": knowledge[-20:] if knowledge else [],
            "oof_score": state.get("oof_score"),
            "lb_score": state.get("lb_score"),
            "best_oof": state.get("best_oof"),
            "gap_history": state.get("gap_history", []),
            "submissions_today": state.get("submissions_today", 0),
            "session_summary": state.get("session_summary", ""),
            "days_remaining": competition.get("days_remaining", 0),
            "exploration_flag": state.get("exploration_flag", True),
            "consecutive_same_directive": state.get("consecutive_same_directive", 0),
            "experiment_type_distribution": _type_distribution(all_experiments),
            "exploration_budget": _exploration_budget(competition, total_days),
        }

    @staticmethod
    def build_hypothesis_context(state: GraphState) -> dict:
        knowledge = _load_knowledge()
        directive = state.get("directive", {})
        target_model = directive.get("target_model", "")
        relevant_knowledge = [
            k for k in knowledge if k.get("scope", {}).get("model_type") == target_model
        ]
        parent_script_source = _load_parent_script_source(directive)
        return {
            "directive": directive,
            "relevant_knowledge": relevant_knowledge[-10:],
            "competition": state.get("competition", {}),
            "parent_script_source": parent_script_source,
        }


def _load_knowledge() -> list:
    if KNOWLEDGE_PATH.exists():
        try:
            return json.loads(KNOWLEDGE_PATH.read_text())
        except Exception:
            pass
    return []


def _load_all_experiments() -> list:
    if not EXPERIMENTS_DIR.exists():
        return []
    experiments = []
    for f in sorted(EXPERIMENTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            experiments.append(json.loads(f.read_text()))
        except Exception:
            pass
    return experiments


def _top_n_by_oof(experiments: list, n: int) -> list:
    scored = [e for e in experiments if e.get("oof_score") is not None]
    return sorted(scored, key=lambda e: e["oof_score"], reverse=True)[:n]


def _type_distribution(experiments: list) -> dict:
    dist: dict = {}
    for e in experiments:
        model = e.get("directive", {}).get("target_model", "unknown")
        dist[model] = dist.get(model, 0) + 1
    return dist


def _days_elapsed(experiments: list) -> float:
    """Compute actual elapsed days from earliest to latest experiment timestamp."""
    if not experiments:
        return 0.0
    timestamps = []
    for e in experiments:
        # Try finding a timestamp in the experiment record or its finding
        ts_str = e.get("timestamp") or (e.get("finding", {}) or {}).get("timestamp")
        if ts_str:
            try:
                timestamps.append(datetime.fromisoformat(ts_str))
            except (ValueError, TypeError):
                pass
    if len(timestamps) < 2:
        return 0.0
    earliest = min(timestamps)
    latest = max(timestamps)
    return max(0.0, (latest - earliest).total_seconds() / 86400.0)


def _exploration_budget(competition: dict, total_days: float) -> float:
    remaining = competition.get("days_remaining", 0)
    if total_days <= 0:
        return 1.0
    return min(1.0, remaining / total_days)


def _load_parent_script_source(directive: dict) -> str:
    """Load parent script source using version metadata for accurate lookup."""
    parent_version = directive.get("parent_version", "")

    # If directive specifies a parent_version, load that specific versioned script
    if parent_version:
        versioned_script = SCRIPTS_DIR / f"{parent_version}.py"
        if versioned_script.exists():
            try:
                return versioned_script.read_text()
            except Exception:
                pass

    # Fall back: find best experiment for the target model via version metadata
    target_model = directive.get("target_model", "")
    if target_model and VERSIONS_DIR.exists():
        best_version = _find_best_version_for_model(target_model)
        if best_version:
            versioned_script = SCRIPTS_DIR / f"{best_version}.py"
            if versioned_script.exists():
                try:
                    return versioned_script.read_text()
                except Exception:
                    pass

    # Last resort: most recent versioned script
    scripts = sorted(SCRIPTS_DIR.glob("v*.py"), reverse=True)
    if scripts:
        try:
            return scripts[0].read_text()
        except Exception:
            pass
    return ""


def _find_best_version_for_model(target_model: str) -> "str | None":
    """Find the version tag of the best OOF experiment for a given model type."""
    if not EXPERIMENTS_DIR.exists():
        return None
    best_score = None
    best_version = None
    for f in EXPERIMENTS_DIR.glob("*.json"):
        try:
            record = json.loads(f.read_text())
            model = (record.get("directive") or {}).get("target_model", "")
            oof = record.get("oof_score")
            if model == target_model and oof is not None:
                if best_score is None or oof > best_score:
                    best_score = oof
                    best_version = record.get("run_id")
        except Exception:
            pass
    return best_version
