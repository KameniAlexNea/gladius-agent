"""
Metric tools exposed as an MCP server for Claude agents.

NOTE: Metric computation (auc_roc, rmse, logloss, …) is intentionally NOT
hard-coded here — the CodeAgent writes the evaluation logic as part of the
solution script so it can be adapted to any competition metric on the fly.

This server only exposes *ensemble-support* utilities that need to be
available at orchestration time, independent of any competition-specific
metric.

Usage:
    from gladius.tools.metric_tools import metric_server
    options = ClaudeAgentOptions(mcp_servers={"metrics": metric_server}, ...)
"""

from typing import Any

import numpy as np
from claude_agent_sdk import create_sdk_mcp_server, tool


@tool(
    "compute_oof_correlation",
    (
        "Compute pairwise Pearson correlations between OOF arrays for ensemble diversity analysis. "
        "Pass a JSON list of .npy file paths. Returns a correlation matrix as text."
    ),
    {"oof_paths": list},
)
async def compute_oof_correlation(args: dict[str, Any]) -> dict[str, Any]:
    try:
        paths = args["oof_paths"]
        arrays = []
        for p in paths:
            arr = np.load(p)
            if arr.ndim == 2:
                arr = arr[:, 1] if arr.shape[1] == 2 else arr.mean(axis=1)
            arrays.append(arr)

        n = len(arrays)
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                corr[i, j] = float(np.corrcoef(arrays[i], arrays[j])[0, 1])

        lines = ["Correlation matrix:"]
        names = [p.split("/")[-1] for p in paths]
        lines.append("  " + "  ".join(f"{n:>12}" for n in names))
        for i, row_name in enumerate(names):
            lines.append(
                f"{row_name:>12}  "
                + "  ".join(f"{corr[i, j]:>12.4f}" for j in range(n))
            )

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


@tool(
    "check_oof_improvement",
    (
        "Check whether the most recent OOF score in a history list is an improvement "
        "over all previous entries. Pass the full ordered list of OOF scores "
        "(oldest first, newest last), whether higher is better (maximize=True for AUC, "
        "maximize=False for RMSE/log-loss), and an optional threshold (default 1e-4). "
        "Returns is_improvement, previous_best, delta, and a human-readable verdict."
    ),
    {"oof_scores": list, "maximize": bool, "threshold": float},
)
async def check_oof_improvement(args: dict[str, Any]) -> dict[str, Any]:
    try:
        scores = args["oof_scores"]
        if not scores:
            return {
                "content": [
                    {"type": "text", "text": "Error: oof_scores list is empty"}
                ],
                "is_error": True,
            }
        maximize = bool(args.get("maximize", True))
        threshold = float(args.get("threshold") or 1e-4)

        latest = float(scores[-1])
        history = [float(s) for s in scores[:-1]]

        if not history:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"First experiment: {latest:.6f} — no previous best to compare."
                        ),
                    }
                ]
            }

        prev_best = max(history) if maximize else min(history)
        delta = latest - prev_best  # positive = better for maximize
        is_improvement = (
            latest > prev_best + threshold
            if maximize
            else latest < prev_best - threshold
        )

        verdict = (
            f"{'IMPROVEMENT ✅' if is_improvement else 'NO IMPROVEMENT ❌'}\n"
            f"Latest OOF   : {latest:.6f}\n"
            f"Previous best: {prev_best:.6f}\n"
            f"Delta        : {delta:+.6f} ({'higher is better' if maximize else 'lower is better'})\n"
            f"Threshold    : {threshold}\n"
            f"Improved     : {is_improvement}"
        )
        return {"content": [{"type": "text", "text": verdict}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


# ── MCP server instance ────────────────────────────────────────────────
metric_server = create_sdk_mcp_server(
    name="metrics",
    version="1.0.0",
    tools=[compute_oof_correlation, check_oof_improvement],
)
