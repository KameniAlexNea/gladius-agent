"""
Fake competition platform — scores submissions locally for offline testing.

No internet, no account needed.

Environment variables:
    FAKE_ANSWERS_PATH    — CSV with columns [id, target], default: data/.answers.csv
    FAKE_PLATFORM_DIR    — directory to store submission history, default: .fake_platform/

The submission CSV must have columns [id, target] where target is a probability
(or a label; AUC-ROC is used for scoring).

Usage:
    from gladius.tools.fake_platform_tools import fake_server
    options = ClaudeAgentOptions(mcp_servers={"fake": fake_server}, ...)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from gladius.config import SETTINGS as _SETTINGS
from gladius.tools._response import err as _err
from gladius.tools._response import ok as _ok

# ── Seeded fake competitors on the leaderboard ────────────────────────────────
_FAKE_LEADERBOARD = [
    {"username": "ml_wizard", "score": 0.9512, "submissions": 12},
    {"username": "DataNinja42", "score": 0.9388, "submissions": 7},
    {"username": "gradient_god", "score": 0.9201, "submissions": 21},
    {"username": "overfit_king", "score": 0.9047, "submissions": 44},
    {"username": "baseline_bob", "score": 0.8733, "submissions": 3},
]


def _answers_path() -> Path:
    return Path(_SETTINGS.fake_answers_path)


def _platform_dir() -> Path:
    d = Path(_SETTINGS.fake_platform_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_history() -> list[dict]:
    p = _platform_dir() / "history.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


def _save_history(history: list[dict]) -> None:
    p = _platform_dir() / "history.json"
    p.write_text(json.dumps(history, indent=2))


def _score_submission(sub_path: str) -> float:
    """Compute AUC-ROC of submission against the hidden answer key."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    answers = pd.read_csv(_answers_path()).set_index("id")["target"]
    submission = pd.read_csv(sub_path).set_index("id")["target"]

    # Align on shared IDs
    common = answers.index.intersection(submission.index)
    if len(common) == 0:
        raise ValueError("No matching IDs between submission and answer key")

    y_true = answers.loc[common].values.astype(int)
    y_pred = submission.loc[common].values.astype(float)

    # Clamp probabilities
    y_pred = np.clip(y_pred, 0.0, 1.0)

    return float(roc_auc_score(y_true, y_pred))


# ── MCP tools ─────────────────────────────────────────────────────────────────


@tool(
    "fake_submit",
    (
        "Submit a prediction CSV to the fake local platform. "
        "Scores the file against the hidden answer key using AUC-ROC. "
        "Records the result and returns the public score."
    ),
    {"file_path": str, "comment": str},
)
async def fake_submit(args: dict[str, Any]) -> dict[str, Any]:
    try:
        file_path = args["file_path"]
        comment = args.get("comment", "")

        if not Path(file_path).exists():
            return _err("file_not_found", f"File not found: {file_path}")

        score = _score_submission(file_path)
        history = _load_history()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "file": str(file_path),
            "score": round(score, 6),
            "comment": comment,
            "rank": None,  # filled below
        }

        history.append(entry)
        _save_history(history)

        # Compute rank among fake leaderboard + own submissions
        all_scores = sorted(
            [e["score"] for e in history] + [c["score"] for c in _FAKE_LEADERBOARD],
            reverse=True,
        )
        rank = all_scores.index(score) + 1

        return _ok(
            (
                f"Submission accepted.\n"
                f"Public AUC-ROC: {score:.6f}\n"
                f"Leaderboard rank: {rank}/{len(all_scores)}\n"
                f"File: {file_path}"
            ),
            data={"score": round(score, 6), "rank": rank, "total": len(all_scores)},
        )
    except Exception as e:
        return _err("scoring_failed", f"Scoring error: {e}")


@tool(
    "fake_leaderboard",
    (
        "Show the fake platform leaderboard including your own submissions. "
        "Returns a ranked table sorted by AUC-ROC descending."
    ),
    {"top_n": int},
)
async def fake_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    try:
        top_n = int(args.get("top_n", 20))
        history = _load_history()

        # Best score per user (own submissions)
        own_best = max((e["score"] for e in history), default=None)

        rows = list(_FAKE_LEADERBOARD)  # copy
        if own_best is not None:
            rows.append(
                {"username": "YOU", "score": own_best, "submissions": len(history)}
            )

        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:top_n]

        header = f"{'Rank':>4}  {'Username':<20}  {'Score':>10}  {'Submissions':>11}"
        lines = [header, "-" * len(header)]
        for i, r in enumerate(rows, 1):
            marker = " ←" if r["username"] == "YOU" else ""
            lines.append(
                f"{i:>4}  {r['username']:<20}  {r['score']:>10.6f}  {r['submissions']:>11}{marker}"
            )

        return _ok("\n".join(lines), data={"top_n": top_n})
    except Exception as e:
        return _err("leaderboard_failed", f"Error: {e}")


@tool(
    "fake_submission_history",
    (
        "List all your previous submissions on the fake platform with scores and timestamps."
    ),
    {},
)
async def fake_submission_history(args: dict[str, Any]) -> dict[str, Any]:
    try:
        history = _load_history()
        if not history:
            return _ok("No submissions yet.", data={"count": 0})

        lines = [f"{'#':>3}  {'Timestamp':<24}  {'Score':>10}  Comment"]
        for i, e in enumerate(history, 1):
            lines.append(
                f"{i:>3}  {e['timestamp']:<24}  {e['score']:>10.6f}  {e.get('comment', '')}"
            )
        return _ok("\n".join(lines), data={"count": len(history)})
    except Exception as e:
        return _err("history_failed", f"Error: {e}")


@tool(
    "fake_status",
    ("Check your current rank and submission count on the fake platform."),
    {},
)
async def fake_status(args: dict[str, Any]) -> dict[str, Any]:
    try:
        history = _load_history()
        own_best = max((e["score"] for e in history), default=None)

        if own_best is None:
            rank_str = "unranked (no submissions yet)"
        else:
            all_scores = sorted(
                [e["score"] for e in history] + [c["score"] for c in _FAKE_LEADERBOARD],
                reverse=True,
            )
            rank = all_scores.index(own_best) + 1
            rank_str = f"{rank}/{len(all_scores)}"

        lines = [
            "Platform: FAKE (local, no internet)",
            f"Total submissions: {len(history)}",
            (
                f"Best public score: {own_best:.6f}"
                if own_best
                else "Best public score: N/A"
            ),
            f"Current rank: {rank_str}",
            "Remaining submissions: unlimited",
        ]
        return _ok(
            "\n".join(lines),
            data={
                "total_submissions": len(history),
                "best_score": own_best,
                "rank": rank_str,
            },
        )
    except Exception as e:
        return _err("status_failed", f"Error: {e}")


# ── MCP server instance ───────────────────────────────────────────────────────
fake_server = create_sdk_mcp_server(
    name="fake",
    version="1.0.0",
    tools=[fake_submit, fake_leaderboard, fake_submission_history, fake_status],
)
