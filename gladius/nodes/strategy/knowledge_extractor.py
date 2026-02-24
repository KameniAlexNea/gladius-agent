import json
from pathlib import Path
from datetime import datetime
from gladius.state import GraphState

KNOWLEDGE_PATH = Path("state/knowledge.json")


def knowledge_extractor_node(state: GraphState) -> GraphState:
    run_id = state.get("run_id", "unknown")
    status = state.get("experiment_status", "unknown")
    oof_score = state.get("oof_score")
    lb_score = state.get("lb_score")
    error_message = state.get("error_message", "")
    spec = state.get("current_experiment", {})
    directive = state.get("directive", {})

    finding_type = _classify_finding(status, error_message, oof_score, lb_score, state)

    finding = {
        "experiment_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "finding_type": finding_type,
        "scope": {
            "model_type": directive.get("target_model", "unknown"),
            "directive_type": directive.get("directive_type", "unknown"),
            "changes": spec.get("changes", []) if spec else [],
        },
        "evidence": {
            "oof_score": oof_score,
            "lb_score": lb_score,
            "gap_vs_best": _compute_gap(state),
            "failure_reason": error_message or status,
        },
        "conclusion": _build_conclusion(finding_type, spec or {}, error_message, directive),
    }

    _append_knowledge(finding)

    return {
        "next_node": "router",
        "error_message": None,
        "experiment_status": "pending",
        "current_experiment": None,
        "run_id": None,
        "running_pid": None,
        "oof_score": None,
        "lb_score": None,
    }


def _classify_finding(status, error, oof, lb, state):
    if status in ("killed", "failed"):
        if "OOM" in (error or "") or "memory" in (error or "").lower():
            return "param_failure"
        return "model_failure"
    if oof is not None and lb is not None and (oof - lb) > 0.02:
        return "overfitting_signal"
    if status == "failed":
        return "feature_failure"
    return "model_failure"


def _compute_gap(state):
    oof = state.get("oof_score")
    lb = state.get("lb_score")
    if oof is not None and lb is not None:
        return oof - lb
    return None


def _build_conclusion(finding_type, spec, error, directive):
    model = directive.get("target_model", "model")
    if finding_type == "param_failure":
        changes = spec.get("changes", [])
        param_changes = [c for c in changes if c.get("type") == "param_change"]
        if param_changes:
            c = param_changes[0]
            return f"{c['param']}={c['new']} caused failure for {model}. Revert to {c.get('old', 'previous')}."
    if finding_type == "overfitting_signal":
        return f"OOF-LB gap too large for {model}. Reduce model complexity or add regularization."
    return f"{model} experiment ended with status {error or 'unknown'}. Review logs."


def _append_knowledge(finding: dict):
    KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    findings = []
    if KNOWLEDGE_PATH.exists():
        try:
            findings = json.loads(KNOWLEDGE_PATH.read_text())
        except Exception:
            pass
    findings.append(finding)
    KNOWLEDGE_PATH.write_text(json.dumps(findings, indent=2))
