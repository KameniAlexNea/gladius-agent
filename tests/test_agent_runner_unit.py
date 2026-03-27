from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import gladius.roles.agent_runner as ar


class _FakeToolUseBlock:
    def __init__(self, name: str, input: dict, block_id: str = "b1"):
        self.name = name
        self.input = input
        self.id = block_id


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeAssistantMessage:
    def __init__(self, content, parent_tool_use_id=None):
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id


class _FakeSystemMessage:
    def __init__(self, subtype: str, data: dict):
        self.subtype = subtype
        self.data = data
        self.parent_tool_use_id = None


class _FakeResultMessage:
    def __init__(
        self,
        structured_output=None,
        is_error=False,
        result="",
        session_id="s",
        subtype="success",
        stop_reason="end_turn",
    ):
        self.structured_output = structured_output
        self.is_error = is_error
        self.result = result
        self.session_id = session_id
        self.subtype = subtype
        self.stop_reason = stop_reason
        self.duration_ms = 100
        self.duration_api_ms = 80
        self.num_turns = 1
        self.total_cost_usd = 0.0
        self.usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }


class _FakeTaskStartedMessage:
    def __init__(self, task_type: str, tool_use_id: str | None, session_id: str = ""):
        self.task_type = task_type
        self.tool_use_id = tool_use_id
        self.session_id = session_id


class _FakeProcessError(Exception):
    def __init__(self, stderr: str = ""):
        super().__init__(stderr)
        self.stderr = stderr


def _patch_runtime(monkeypatch):
    monkeypatch.setattr(ar, "validate_runtime_invocation", lambda **kwargs: None)
    monkeypatch.setattr(ar, "get_runtime_model", lambda: "m")
    monkeypatch.setattr(ar, "ClaudeAgentOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(ar, "SystemMessage", _FakeSystemMessage)
    monkeypatch.setattr(ar, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(ar, "TaskStartedMessage", _FakeTaskStartedMessage)
    monkeypatch.setattr(ar, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(ar, "ToolUseBlock", _FakeToolUseBlock)
    monkeypatch.setattr(ar, "TextBlock", _FakeTextBlock)


def test_extract_subagent_type_aliases():
    assert ar._extract_subagent_type({"subagent_type": "x"}) == "x"
    assert ar._extract_subagent_type({}) == ""


def test_extract_session_id_from_system_message_guarded_type():
    assert ar._extract_session_id_from_system_message(_FakeSystemMessage("init", {})) is None
    assert (
        ar._extract_session_id_from_system_message(
            _FakeSystemMessage("init", {"session_id": "abc"})
        )
        == "abc"
    )


def test_register_subagent_policy_from_task_start_local_agent():
    delegated: dict[str, list[str]] = {}
    pending = ar.collections.deque([["Read", "Write"]])
    ar._register_subagent_policy_from_task_start(
        agent_name="a",
        message=_FakeTaskStartedMessage(
            task_type="local_agent", tool_use_id="tu_123", session_id="sid"
        ),
        delegated_tool_policies=delegated,
        pending_subagent_tools=pending,
    )
    assert delegated["tu_123"] == ["Read", "Write"]
    assert not pending


def test_register_subagent_policy_from_task_start_ignores_non_agent():
    delegated: dict[str, list[str]] = {}
    pending = ar.collections.deque([["Read", "Write"]])
    ar._register_subagent_policy_from_task_start(
        agent_name="a",
        message=_FakeTaskStartedMessage(
            task_type="local_bash", tool_use_id="tu_123", session_id="sid"
        ),
        delegated_tool_policies=delegated,
        pending_subagent_tools=pending,
    )
    assert delegated == {}
    assert pending


def test_parse_structured_from_assistant_text_returns_dict(monkeypatch):
    monkeypatch.setattr(ar, "_parse_json", lambda text: {"ok": 1})
    monkeypatch.setattr(ar, "TextBlock", _FakeTextBlock)
    msg = _FakeAssistantMessage([_FakeTextBlock('{"ok": 1}')])
    assert ar._parse_structured_from_assistant_text(agent_name="a", last_assistant_msg=msg) == {
        "ok": 1
    }


def test_resolve_effective_allowed_tools_fifo_fallback():
    delegated = {}
    pending = ar.collections.deque([["Read", "Write"]])
    tools, label = ar._resolve_effective_allowed_tools(
        agent_name="a",
        message_parent_tool_use_id="p1",
        default_allowed_tools=["Read"],
        delegated_tool_policies=delegated,
        pending_subagent_tools=pending,
    )
    assert tools == ["Read", "Write"]
    assert "subagent_allowed_tools" in label
    assert delegated["p1"] == ["Read", "Write"]


def test_handle_tool_use_block_invalid_subagent_returns_error():
    err = ar._handle_tool_use_block(
        agent_name="a",
        block=_FakeToolUseBlock("Agent", {"subagent_type": "nope"}),
        message_parent_tool_use_id=None,
        allowed_tools=["Read"],
        delegated_tool_policies={},
        pending_subagent_tools=ar.collections.deque(),
    )
    assert "without a valid subagent_type" in (err or "")


def test_run_agent_raises_tool_permission_error(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(ar, "is_tool_allowed", lambda name, allowed: False)

    async def _fake_query(prompt, options):
        yield _FakeAssistantMessage([_FakeToolUseBlock("Read", {})])
        yield _FakeResultMessage(structured_output={})

    monkeypatch.setattr(ar, "query", _fake_query)

    with pytest.raises(ar.ToolPermissionError):
        asyncio.run(
            ar.run_agent(
                prompt="p",
                system_prompt="s",
                allowed_tools=["Read"],
                cwd=str(tmp_path),
                max_retries=1,
                verbose=False,
            )
        )


def test_run_agent_structured_text_fallback(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(ar, "is_tool_allowed", lambda name, allowed: True)
    monkeypatch.setattr(ar, "_parse_json", lambda text: {"rescued": True})

    async def _fake_query(prompt, options):
        yield _FakeAssistantMessage([_FakeTextBlock('{"rescued": true}')])
        yield _FakeResultMessage(structured_output=None, is_error=False, session_id="s1")

    monkeypatch.setattr(ar, "query", _fake_query)

    out, sid = asyncio.run(
        ar.run_agent(
            prompt="p",
            system_prompt="s",
            allowed_tools=["Read"],
            output_schema={"type": "object"},
            cwd=str(tmp_path),
            max_retries=1,
            verbose=False,
        )
    )
    assert out == {"rescued": True}
    assert sid == "s1"


def test_run_agent_process_error_recovers_from_assistant_text(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(ar, "ProcessError", _FakeProcessError)
    monkeypatch.setattr(ar, "is_tool_allowed", lambda name, allowed: True)
    monkeypatch.setattr(ar, "_parse_json", lambda text: {"rescued": 1})

    async def _fake_query(prompt, options):
        yield _FakeSystemMessage("init", {"session_id": "early"})
        yield _FakeAssistantMessage([_FakeTextBlock('{"rescued": 1}')])
        raise _FakeProcessError("boom")

    monkeypatch.setattr(ar, "query", _fake_query)

    out, sid = asyncio.run(
        ar.run_agent(
            prompt="p",
            system_prompt="s",
            allowed_tools=["Read"],
            cwd=str(tmp_path),
            max_retries=1,
            verbose=False,
        )
    )
    assert out == {"rescued": 1}
    assert sid == "early"
