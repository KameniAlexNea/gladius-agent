from __future__ import annotations

import types

from gladius.utilities import langsmith_tracing as lst


def test_should_enable_langsmith_for_ollama_bridge_true(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    assert lst.should_enable_langsmith_for_ollama_bridge() is True


def test_init_langsmith_tracing_returns_false_when_not_configured(monkeypatch):
    monkeypatch.delenv("GLADIUS_ENABLE_LANGSMITH", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert lst.init_langsmith_tracing() is False


def test_init_langsmith_tracing_returns_false_without_api_key(monkeypatch):
    monkeypatch.setenv("GLADIUS_ENABLE_LANGSMITH", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert lst.init_langsmith_tracing() is False


def test_init_langsmith_tracing_calls_configure(monkeypatch):
    monkeypatch.setenv("GLADIUS_ENABLE_LANGSMITH", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    called = []

    def _configure():
        called.append(True)

    fake_integration = types.SimpleNamespace(configure_claude_agent_sdk=_configure)
    monkeypatch.setitem(
        __import__("sys").modules,
        "langsmith.integrations.claude_agent_sdk",
        fake_integration,
    )

    result = lst.init_langsmith_tracing()
    assert result is True
    assert called == [True]


def test_init_langsmith_tracing_returns_false_on_import_error(monkeypatch):
    monkeypatch.setenv("GLADIUS_ENABLE_LANGSMITH", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langsmith.integrations.claude_agent_sdk",
        None,
    )
    assert lst.init_langsmith_tracing() is False
