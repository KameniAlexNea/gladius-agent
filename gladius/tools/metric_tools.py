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
            lines.append(f"{row_name:>12}  " + "  ".join(f"{corr[i,j]:>12.4f}" for j in range(n)))

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


# ── MCP server instance ───────────────────────────────────────────────────────
metric_server = create_sdk_mcp_server(
    name="metrics",
    version="1.0.0",
    tools=[compute_oof_correlation],
)
