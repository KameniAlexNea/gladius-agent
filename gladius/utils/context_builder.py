import json
from pathlib import Path

from gladius.state import GraphState

KNOWLEDGE_PATH = Path("state/knowledge.json")
EXPERIMENTS_DIR = Path("state/experiments")


class ContextBuilder:
    @staticmethod
    def build_strategy_context(state: GraphState) -> dict:
        knowledge = _load_knowledge()
        competition = state.get("competition", {})
        return {
            "competition": competition,
            "knowledge_summary": knowledge[-20:] if knowledge else [],
            "oof_score": state.get("oof_score"),
            "lb_score": state.get("lb_score"),
            "gap_history": state.get("gap_history", []),
            "submissions_today": state.get("submissions_today", 0),
            "session_summary": state.get("session_summary", ""),
            "days_remaining": competition.get("days_remaining", 0),
            "exploration_flag": state.get("exploration_flag", True),
            "consecutive_same_directive": state.get("consecutive_same_directive", 0),
        }

    @staticmethod
    def build_hypothesis_context(state: GraphState) -> dict:
        knowledge = _load_knowledge()
        directive = state.get("directive", {})
        target_model = directive.get("target_model", "")
        # Filter knowledge for this model
        relevant_knowledge = [
            k for k in knowledge if k.get("scope", {}).get("model_type") == target_model
        ]
        return {
            "directive": directive,
            "relevant_knowledge": relevant_knowledge[-10:],
            "competition": state.get("competition", {}),
        }


def _load_knowledge() -> list:
    if KNOWLEDGE_PATH.exists():
        try:
            return json.loads(KNOWLEDGE_PATH.read_text())
        except Exception:
            pass
    return []
