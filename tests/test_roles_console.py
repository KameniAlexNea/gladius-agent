from __future__ import annotations


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
        self.duration_api_ms = 90
        self.subtype = "success"
        self.stop_reason = "end_turn"
        self.result = "ok"
        self.usage = {
            "input_tokens": 12,
            "output_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }


class _Stream:
    def __init__(self):
        self.uuid = "u1"
        self.session_id = "s1234567890"
        self.event = {"type": "content_block_delta"}
        self.parent_tool_use_id = None


class _TaskStarted:
    def __init__(self, task_id="t1", description="desc", task_type="local_agent"):
        self.task_id = task_id
        self.description = description
        self.task_type = task_type
        self.session_id = "sess1"
        self.tool_use_id = "tu1"


class _TaskProgress:
    def __init__(self, task_id="t1", description="working", last_tool_name="Read"):
        self.task_id = task_id
        self.description = description
        self.session_id = "sess1"
        self.usage = {"total_tokens": 12, "tool_uses": 1, "duration_ms": 55}
        self.last_tool_name = last_tool_name


class _TaskNotification:
    def __init__(self, task_id="t1", status="completed", summary="done"):
        self.task_id = task_id
        self.status = status
        self.summary = summary
        self.session_id = "sess1"
        self.tool_use_id = "tu1"
        self.output_file = "out.txt"
        self.usage = {"total_tokens": 20}


def test_fmt_helpers():
    assert "a" in c._fmt_input({"a": 1})
    assert c._fmt_result(None) == "(empty)"
    assert "x" in c._fmt_result([{"text": "x"}])
    redacted = c._fmt_input({"api_key": "sk-abc123456789", "safe": "ok"})
    assert "REDACTED" in redacted


def test_log_message_branches(monkeypatch):
    logs = []
    monkeypatch.delenv("GLADIUS_LOG_STREAM_EVENTS", raising=False)
    monkeypatch.setattr(c.logger, "debug", lambda msg: logs.append(str(msg)))
    monkeypatch.setattr(c, "SystemMessage", _System)
    monkeypatch.setattr(c, "AssistantMessage", _Assistant)
    monkeypatch.setattr(c, "UserMessage", _User)
    monkeypatch.setattr(c, "ResultMessage", _Result)
    monkeypatch.setattr(c, "TextBlock", _Text)
    monkeypatch.setattr(c, "ThinkingBlock", _Thinking)
    monkeypatch.setattr(c, "ToolUseBlock", _ToolUse)
    monkeypatch.setattr(c, "ToolResultBlock", _ToolResult)
    monkeypatch.setattr(c, "StreamEvent", _Stream)
    monkeypatch.setattr(c, "TaskStartedMessage", _TaskStarted)
    monkeypatch.setattr(c, "TaskProgressMessage", _TaskProgress)
    monkeypatch.setattr(c, "TaskNotificationMessage", _TaskNotification)

    c._log_message("a", _Stream())
    c._log_message("a", _TaskStarted(task_type="local_agent", description="subtask"))
    c._log_message("a", _TaskProgress(last_tool_name="Bash"))
    c._log_message("a", _TaskNotification(status="failed", summary="x"))

    c._log_message("a", _System("init", {"session_id": "abc123"}))
    c._log_message(
        "a",
        _Assistant(
            [
                _Text("hello world"),
                _Thinking("pondering"),
                _ToolUse(
                    "TodoWrite", {"todos": [{"status": "completed", "content": "x"}]}
                ),
                _ToolUse("ExitPlanMode", {"plan": "line1"}),
                _ToolUse(
                    "Agent",
                    {
                        "subagent_type": "feature-engineer",
                        "description": "d",
                        "prompt": "p",
                    },
                ),
                _ToolUse("Read", {"path": "x", "token": "sk-secret-token"}),
            ]
        ),
    )
    c._log_message("a", _User([_ToolResult("ok")]))
    c._log_message("a", _Result(False))

    assert logs
    joined = "\n".join(logs)
    assert "task:subagent" in joined
    assert "task failed" in joined
    assert "subtype=success" in joined
    assert "stream=content_block_delta" not in joined
    assert "REDACTED" in joined


def test_stream_event_logging_opt_in(monkeypatch):
    logs = []
    monkeypatch.setenv("GLADIUS_LOG_STREAM_EVENTS", "1")
    monkeypatch.setattr(c.logger, "debug", lambda msg: logs.append(str(msg)))
    monkeypatch.setattr(c, "StreamEvent", _Stream)

    c._log_message("a", _Stream())

    assert logs
    assert any("stream=content_block_delta" in line for line in logs)
