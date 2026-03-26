from __future__ import annotations

import contextlib
import types

from gladius import langsmith_tracing as lst


class _DummyCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_should_enable_langsmith_for_ollama_bridge_true(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    assert lst.should_enable_langsmith_for_ollama_bridge() is True


def test_init_langsmith_client_returns_none_when_not_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert lst.init_langsmith_client() == (None, None)


def test_init_langsmith_client_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    class _FakeClient:
        def __init__(self, api_key: str, api_url: str):
            self.api_key = api_key
            self.api_url = api_url

    fake_mod = types.SimpleNamespace(Client=_FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "langsmith", fake_mod)

    client, project = lst.init_langsmith_client()
    assert isinstance(client, _FakeClient)
    assert project == "gladius-agent"


def test_langsmith_tracing_context_falls_back_to_nullcontext(monkeypatch):
    # Simulate missing run_helpers import
    monkeypatch.setitem(__import__("sys").modules, "langsmith.run_helpers", None)
    ctx = lst.langsmith_tracing_context(object(), "p")
    assert isinstance(ctx, contextlib.AbstractContextManager)


def test_langsmith_tracing_context_uses_tracing_context(monkeypatch):
    def _tracing_context(**kwargs):
        assert kwargs["project_name"] == "p"
        return _DummyCtx()

    fake_helpers = types.SimpleNamespace(tracing_context=_tracing_context)
    monkeypatch.setitem(__import__("sys").modules, "langsmith.run_helpers", fake_helpers)

    ctx = lst.langsmith_tracing_context(object(), "p")
    with ctx:
        pass
