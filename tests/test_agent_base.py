from gladius.agents._base import _is_bash_command_scoped_to_cwd, _is_tool_allowed


def test_tool_allowed_direct_match():
    assert _is_tool_allowed("Read", ["Read", "Grep"]) is True


def test_tool_allowed_agent_task_alias():
    assert _is_tool_allowed("Task", ["Agent(ml-scaffolder,ml-developer)"]) is True


def test_tool_forbidden_when_not_in_list():
    assert _is_tool_allowed("Bash", ["Read", "Grep"]) is False


def test_structured_output_is_runtime_allowed():
    assert _is_tool_allowed("StructuredOutput", ["Read", "Write"]) is True


def test_bash_scope_allows_in_project_paths():
    cwd = "/tmp/project"
    assert _is_bash_command_scoped_to_cwd("ls src && head -n 5 data/train.csv", cwd)


def test_bash_scope_blocks_absolute_outside_project():
    cwd = "/tmp/project"
    assert not _is_bash_command_scoped_to_cwd("ls /etc", cwd)


def test_bash_scope_blocks_cd_escape():
    cwd = "/tmp/project"
    assert not _is_bash_command_scoped_to_cwd("cd .. && ls", cwd)

