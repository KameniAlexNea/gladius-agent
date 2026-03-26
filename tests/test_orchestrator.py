"""Tests for orchestrator iteration start/resume helpers."""

from __future__ import annotations

from gladius.orchestrator import _resolve_start_iteration


def test_resolve_start_iteration_default_when_env_missing(monkeypatch):
    monkeypatch.delenv("GLADIUS_START_ITERATION", raising=False)
    assert _resolve_start_iteration(max_iterations=20) == 1


def test_resolve_start_iteration_uses_env_value(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "4")
    assert _resolve_start_iteration(max_iterations=20) == 4


def test_resolve_start_iteration_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "abc")
    assert _resolve_start_iteration(max_iterations=20) == 1


def test_resolve_start_iteration_rejects_values_below_one(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "0")
    assert _resolve_start_iteration(max_iterations=20) == 1


def test_resolve_start_iteration_clamps_to_max_iterations(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "99")
    assert _resolve_start_iteration(max_iterations=7) == 7
