"""Platform submission helpers and score tracking."""

from __future__ import annotations

import os
import subprocess

from loguru import logger

from src.state import CompetitionState
from src.tools.zindi_common import (
    create_zindi_user_from_env,
    select_zindi_challenge,
)

# ── Per-platform submit functions ─────────────────────────────────────────────


def _submit_to_kaggle(
    competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    r = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition_id,
            "-f",
            submission_path,
            "-m",
            message,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.warning(f"Kaggle submit stderr: {r.stderr.strip()}")
        return False, "submission_failed"
    logger.info(f"Kaggle submission accepted: {r.stdout.strip()}")
    return True, None


def _submit_to_zindi(
    competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    try:
        user = create_zindi_user_from_env()
        _select_zindi_challenge(user=user, competition_id=competition_id)
        if user.remaining_subimissions <= 0:
            logger.warning("Zindi: no remaining submissions today")
            return False, "quota_exceeded"
        user.submit(filepaths=[submission_path], comments=[message])
        logger.info(
            f"Zindi submission accepted ({user.remaining_subimissions} remaining today)"
        )
        return True, None
    except Exception as exc:
        logger.error(f"Zindi submission error: {exc}")
        return False, "submission_failed"


def _select_zindi_challenge(*, user, competition_id: str) -> None:
    """Select challenge by competition_id, then env challenge id, then env index."""
    select_zindi_challenge(
        user=user,
        competition_id=competition_id,
        env_challenge_id=os.getenv("ZINDI_CHALLENGE_ID"),
        env_challenge_index=os.getenv("ZINDI_CHALLENGE_INDEX"),
    )


def _submit_to_fake(
    competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    try:
        from src.tools.fake_platform_tools import _score_submission

        score = _score_submission(submission_path)
        logger.info(f"[FAKE PLATFORM] Scored: {score:.6f}")
        return True, None
    except Exception as exc:
        logger.error(f"[FAKE PLATFORM] Scoring failed: {exc}")
        return False, "scoring_failed"


# ── Public dispatch ───────────────────────────────────────────────────────────


def submit(
    platform: str, competition_id: str, submission_path: str, message: str
) -> tuple[bool, str | None]:
    """Route a submission to the appropriate platform handler."""
    if platform == "none":
        logger.info(f"[LOCAL] Submission artifact recorded: {submission_path}")
        return True, None
    if platform == "zindi":
        return _submit_to_zindi(competition_id, submission_path, message)
    if platform == "fake":
        return _submit_to_fake(competition_id, submission_path, message)
    return _submit_to_kaggle(competition_id, submission_path, message)


def score_submission_artifact(platform: str, submission_path: str) -> float | None:
    """Return the scored value for platforms that support local scoring (fake only)."""
    if platform != "fake":
        return None
    try:
        from src.tools.fake_platform_tools import _score_submission

        return float(_score_submission(submission_path))
    except Exception as exc:
        logger.warning(f"Could not compute submission score for {platform}: {exc}")
        return None


def update_best_submission_score(
    *,
    state: CompetitionState,
    new_score: float,
) -> None:
    """Update state.best_submission_score if new_score is better."""
    if state.best_submission_score is None:
        state.best_submission_score = new_score
        return
    if state.target_metric is None:
        if new_score > state.best_submission_score:
            state.best_submission_score = new_score
        return
    if state.metric_direction == "maximize":
        if new_score > state.best_submission_score:
            state.best_submission_score = new_score
    else:
        if new_score < state.best_submission_score:
            state.best_submission_score = new_score
