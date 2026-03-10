from gladius.agents._base import _is_bash_command_scoped_to_cwd, _is_tool_allowed
from gladius.agents.runtime.planning_runner import (
    _extract_plan_from_write_block,
    _extract_plan_text_from_mapping,
)


def test_tool_allowed_direct_match():
    assert _is_tool_allowed("Read", ["Read", "Grep"]) is True


def test_tool_allowed_agent_task_alias():
    assert _is_tool_allowed("Task", ["Agent(ml-scaffolder,ml-developer)"]) is True


def test_tool_forbidden_when_not_in_list():
    assert _is_tool_allowed("Bash", ["Read", "Grep"]) is False


def test_bash_scope_allows_in_project_paths():
    cwd = "/tmp/project"
    assert _is_bash_command_scoped_to_cwd("ls src && head -n 5 data/train.csv", cwd)


def test_bash_scope_blocks_absolute_outside_project():
    cwd = "/tmp/project"
    assert not _is_bash_command_scoped_to_cwd("ls /etc", cwd)


def test_bash_scope_blocks_cd_escape():
    cwd = "/tmp/project"
    assert not _is_bash_command_scoped_to_cwd("cd .. && ls", cwd)


def test_extract_plan_from_write_block_only_for_plan_path():
    class _Block:
        name = "Write"
        input = {
            "file_path": "/tmp/.claude/plans/iter-00.md",
            "content": "# Plan\n\n1. Step one\n2. Step two",
        }

    assert (
        _extract_plan_from_write_block(_Block()) == "# Plan\n\n1. Step one\n2. Step two"
    )


def test_extract_plan_from_write_block_ignores_other_paths():
    class _Block:
        name = "Write"
        input = {
            "file_path": "/tmp/notes.md",
            "content": "# Plan\n\n1. Step one",
        }

    assert _extract_plan_from_write_block(_Block()) is None


def test_extract_plan_text_from_mapping_prefers_plan_then_content_then_result():
    assert (
        _extract_plan_text_from_mapping({"plan": "P", "content": "C", "result": "R"})
        == "P"
    )
    assert _extract_plan_text_from_mapping({"content": "C", "result": "R"}) == "C"
    assert _extract_plan_text_from_mapping({"result": "R", "text": "T"}) == "R"


def test_extract_plan_from_write_block_supports_result_key():
    class _Block:
        name = "Write"
        input = {
            "file_path": "/tmp/.claude/plans/iter-00.md",
            "result": "# Plan from result",
        }

    assert _extract_plan_from_write_block(_Block()) == "# Plan from result"
