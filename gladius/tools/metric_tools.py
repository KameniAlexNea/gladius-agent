"""
Metric computation tools exposed as an MCP server for Claude agents.

All functions are memory-safe (no O(n²) outer products).

Usage:
    from gladius.tools.metric_tools import metric_server
    options = ClaudeAgentOptions(mcp_servers={"metrics": metric_server}, ...)
"""
from typing import Any

import numpy as np
from claude_agent_sdk import create_sdk_mcp_server, tool


@tool(
    "compute_oof_metric",
    (
        "Compute an OOF metric score from numpy arrays saved on disk. "
        "Supported metrics: auc_roc, rmse, logloss, accuracy. "
        "For multiclass AUC-ROC uses macro OVR (memory-safe). "
        "Returns METRIC_SCORE: X.XXXXXX on success."
    ),
    {"metric": str, "oof_path": str, "labels_path": str},
)
async def compute_oof_metric(args: dict[str, Any]) -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            log_loss,
            mean_squared_error,
            roc_auc_score,
        )

        oof = np.load(args["oof_path"])
        y = np.load(args["labels_path"])
        metric = args["metric"].lower()

        if metric == "auc_roc":
            if oof.ndim == 2 and oof.shape[1] > 2:
                # Multiclass: macro OVR — sklearn handles it without outer products
                score = roc_auc_score(y, oof, multi_class="ovr", average="macro")
            else:
                oof_1d = oof[:, 1] if oof.ndim == 2 else oof
                score = float(roc_auc_score(y, oof_1d))
        elif metric == "rmse":
            score = float(np.sqrt(mean_squared_error(y, oof)))
        elif metric == "logloss":
            score = float(log_loss(y, oof))
        elif metric == "accuracy":
            preds = np.argmax(oof, axis=1) if oof.ndim == 2 else (oof > 0.5).astype(int)
            score = float(accuracy_score(y, preds))
        else:
            return {
                "content": [{"type": "text", "text": f"Unknown metric: {metric}"}],
                "is_error": True,
            }

        return {"content": [{"type": "text", "text": f"METRIC_SCORE: {score:.6f}"}]}

    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"File not found: {e}"}], "is_error": True}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


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
    tools=[compute_oof_metric, compute_oof_correlation],
)
