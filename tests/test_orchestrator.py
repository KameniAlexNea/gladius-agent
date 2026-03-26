"""Tests for orchestrator iteration start/resume helpers."""

from __future__ import annotations

import os

from gladius.langsmith_tracing import configure_langsmith_env
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


def test_langsmith_env_not_enabled_without_ollama_bridge(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert configure_langsmith_env() is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_langsmith_env_defaults_are_set_for_ollama_bridge(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    assert configure_langsmith_env() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_PROJECT"] == "gladius-agent"


def test_langsmith_env_requires_langsmith_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert configure_langsmith_env() is False
