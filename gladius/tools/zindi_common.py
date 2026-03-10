"""Shared Zindi auth/challenge helpers used by submission + MCP tools."""

from __future__ import annotations

import logging
import os

from zindi.user import Zindian

logger = logging.getLogger(__name__)


def _get_selected_challenge_id(user: Zindian) -> str | None:
    """Best-effort extraction of currently selected challenge id."""
    try:
        current = user.which_challenge
        if isinstance(current, str) and current.strip():
            return current.strip()
    except Exception:
        pass

    # Fallback for zindi package internals.
    data = getattr(user, "_Zindian__challenge_data", None)
    if isinstance(data, dict):
        candidate = data.get("id") or data.get("slug")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _normalize(text: str) -> str:
    return text.strip().lower().replace("_", "-")


def _matches_selected(user: Zindian, challenge_id: str) -> bool:
    selected = _get_selected_challenge_id(user)
    if not selected:
        return False
    return _normalize(selected) == _normalize(challenge_id)


def select_zindi_challenge(
    *,
    user: Zindian,
    competition_id: str | None,
    env_challenge_id: str | None = None,
    env_challenge_index: str | None = None,
) -> str:
    """Select challenge deterministically and return selected challenge id.

    Resolution order:
    1) competition_id (from README/frontmatter config)
    2) env_challenge_id (ZINDI_CHALLENGE_ID)
    3) env_challenge_index (ZINDI_CHALLENGE_INDEX)
    """
    candidates = [competition_id, env_challenge_id]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            user.select_a_challenge(challenge_id=candidate)
        except Exception as exc:
            logger.warning(
                "Could not select Zindi challenge by challenge_id='%s': %s",
                candidate,
                exc,
            )
            continue

        if _matches_selected(user, candidate):
            selected = _get_selected_challenge_id(user)
            return selected or candidate

        logger.warning(
            "Zindi challenge selection by challenge_id='%s' did not bind expected challenge",
            candidate,
        )

    if env_challenge_index is None:
        raise RuntimeError(
            "Could not resolve Zindi challenge. Ensure competition_id in config "
            "matches your Zindi challenge_id or set ZINDI_CHALLENGE_INDEX."
        )

    try:
        idx = int(env_challenge_index)
    except ValueError as exc:
        raise RuntimeError("ZINDI_CHALLENGE_INDEX must be an integer") from exc

    user.select_a_challenge(fixed_index=idx)
    selected = _get_selected_challenge_id(user)
    if not selected:
        raise RuntimeError(
            "Zindi challenge selection by index did not set active challenge"
        )
    return selected


def create_zindi_user_from_env() -> Zindian:
    """Authenticate and return a connected Zindian instance."""
    try:
        from zindi.user import Zindian
    except ImportError as exc:
        raise RuntimeError("zindi package not installed") from exc

    username = os.getenv("ZINDI_USERNAME") or os.getenv("USER_NAME")
    password = os.getenv("ZINDI_PASSWORD") or os.getenv("PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Zindi credentials not found. "
            "Set ZINDI_USERNAME and ZINDI_PASSWORD environment variables."
        )

    return Zindian(username=username, fixed_password=password)
