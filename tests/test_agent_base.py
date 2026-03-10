from gladius.agents._base import _is_tool_allowed


def test_tool_allowed_direct_match():
    assert _is_tool_allowed("Read", ["Read", "Grep"]) is True


def test_tool_allowed_agent_task_alias():
    assert _is_tool_allowed("Task", ["Agent(ml-scaffolder,ml-developer)"]) is True


def test_tool_forbidden_when_not_in_list():
    assert _is_tool_allowed("Bash", ["Read", "Grep"]) is False
