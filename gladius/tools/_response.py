"""Shared MCP tool response helpers used by all platform tool modules."""

from __future__ import annotations

from typing import Any


def ok(text: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "status": "ok",
    }
    if data is not None:
        payload["data"] = data
    return payload


def err(error_type: str, text: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "status": "error",
        "error_type": error_type,
        "is_error": True,
    }
