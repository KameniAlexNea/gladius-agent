from __future__ import annotations

from types import SimpleNamespace

import gladius.roles._console as c


class _Text:
    def __init__(self, text):
        self.text = text


class _Thinking:
    def __init__(self, thinking):
        self.thinking = thinking


class _ToolUse:
    def __init__(self, name, input, block_id="b"):
        self.name = name
        self.input = input
        self.id = block_id


class _ToolResult:
    def __init__(self, content, is_error=False):
        self.content = content
        self.is_error = is_error


class _Assistant:
    def __init__(self, content, parent_tool_use_id=None, error=None):
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id
        self.error = error


class _System:
    def __init__(self, subtype, data):
        self.subtype = subtype
        self.data = data


class _User:
    def __init__(self, content):
        self.content = content


class _Result:
    def __init__(self, is_error=False):
        self.is_error = is_error
        self.total_cost_usd = 0.0
        self.num_turns = 1
        self.duration_ms = 100


def test_fmt_helpers():
    assert "a" in c._fmt_input({"a": 1})
    assert c._fmt_result(None) == "(empty)"
    assert "x" in c._fmt_result([{"text": "x"}])


def test_log_message_branches(monkeypatch):
    logs = []
    monkeypatch.setattr(c.logger, "debug", lambda msg: logs.append(str(msg)))
    monkeypatch.setattr(c, "SystemMessage", _System)
    monkeypatch.setattr(c, "AssistantMessage", _Assistant)
    monkeypatch.setattr(c, "UserMessage", _User)
    monkeypatch.setattr(c, "ResultMessage", _Result)
    monkeypatch.setattr(c, "TextBlock", _Text)
    monkeypatch.setattr(c, "ThinkingBlock", _Thinking)
    monkeypatch.setattr(c, "ToolUseBlock", _ToolUse)
    monkeypatch.setattr(c, "ToolResultBlock", _ToolResult)

    c._log_message("a", _System("init", {"session_id": "abc123"}))
    c._log_message(
        "a",
        _Assistant(
            [
                _Text("hello world"),
                _Thinking("pondering"),
                _ToolUse("TodoWrite", {"todos": [{"status": "completed", "content": "x"}]}),
                _ToolUse("ExitPlanMode", {"plan": "line1"}),
                _ToolUse("Agent", {"subagent_type": "feature-engineer", "description": "d", "prompt": "p"}),
                _ToolUse("Read", {"path": "x"}),
            ]
        ),
    )
    c._log_message("a", _User([_ToolResult("ok")]))
    c._log_message("a", _Result(False))

    assert logs
