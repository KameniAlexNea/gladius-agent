"""Runner for planner calls in permission_mode='plan'."""

from __future__ import annotations

import asyncio
import logging

from claude_agent_sdk import (
    ClaudeAgentOptions,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    query,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk.types import (
    AssistantMessage,
    PermissionResultAllow,
    SystemMessage,
    ToolUseBlock,
)

from gladius.agents._console import _BLUE, _BOLD, _c, _log_message
from gladius.agents.runtime.helpers import (
    PLAN_MODE_ALLOWED_TOOLS,
    PLAN_MODE_DENIED_TOOLS,
    build_runtime_agents,
    get_runtime_model,
    is_bash_command_scoped_to_cwd,
    is_tool_allowed,
    stderr_cb,
    validate_runtime_invocation,
)

logger = logging.getLogger(__name__)


def _extract_plan_text_from_mapping(data: dict) -> str | None:
    """Return first non-empty plan-like payload from common keys."""
    for key in ("plan", "content", "result", "text"):
        value = data.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _extract_plan_from_write_block(block: ToolUseBlock) -> str | None:
    if block.name != "Write":
        return None
    file_path = str(block.input.get("file_path", ""))
    if "/.claude/plans/" not in file_path:
        return None
    return _extract_plan_text_from_mapping(block.input)


async def approve_exit_plan_mode(tool_name: str, input_data: dict, context: object):
    """Block write tools in planning mode; auto-approve everything else."""
    from claude_agent_sdk.types import PermissionResultDeny

    if tool_name in PLAN_MODE_DENIED_TOOLS:
        return PermissionResultDeny(
            message=(
                f"Tool '{tool_name}' is not permitted in planning mode. "
                "Use only Read, Glob, Grep, WebSearch, Skill, TodoWrite, or ExitPlanMode."
            )
        )
    if tool_name not in PLAN_MODE_ALLOWED_TOOLS:
        return PermissionResultDeny(
            message=(
                f"Tool '{tool_name}' is not allowed in planning mode. "
                "Use only Read, Glob, Grep, WebSearch, Skill, TodoWrite, or ExitPlanMode."
            )
        )
    return PermissionResultAllow(updated_input=input_data)


async def run_planning_agent(
    *,
    agent_name: str = "planner",
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    cwd: str,
    resume: str | None = None,
    mcp_servers: dict | None = None,
    max_turns: int | None = None,
    max_retries: int = 3,
    verbose: bool = True,
) -> tuple[str, str]:
    """Run a Claude agent in planning mode and return (plan_text, session_id)."""
    validate_runtime_invocation(
        agent_name=agent_name,
        cwd=cwd,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )

    planning_allowed_tools = list(dict.fromkeys([*allowed_tools, "ExitPlanMode"]))

    runtime_model = get_runtime_model()
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt,
        },
        allowed_tools=planning_allowed_tools,
        permission_mode="plan",
        can_use_tool=approve_exit_plan_mode,
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=stderr_cb,
        setting_sources=["project"],
        agents=build_runtime_agents(runtime_model),
        model=runtime_model,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            early_session_id: str | None = None
            captured_plan: str | None = None
            last_written_plan: str | None = None
            forbidden_tool_error: str | None = None

            if verbose:
                resume_str = f"  resume={resume[:8]}…" if resume else ""
                print(
                    _c(_BOLD + _BLUE, f"\n▶ [{agent_name}] (plan mode)")
                    + f"  tools={planning_allowed_tools}"
                    + resume_str
                )

            async def _prompt_stream():
                yield {"type": "user", "message": {"role": "user", "content": prompt}}

            try:
                async for message in query(prompt=_prompt_stream(), options=options):
                    if verbose:
                        _log_message(agent_name, message)
                    if isinstance(message, SystemMessage) and message.subtype == "init":
                        early_session_id = message.data.get("session_id")
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                write_plan = _extract_plan_from_write_block(block)
                                if write_plan:
                                    last_written_plan = write_plan
                                if not is_tool_allowed(
                                    block.name, planning_allowed_tools
                                ):
                                    forbidden_tool_error = (
                                        f"[{agent_name}] attempted forbidden tool '{block.name}' in plan mode. "
                                        f"allowed_tools={planning_allowed_tools}"
                                    )
                                elif block.name == "Bash":
                                    cmd = str(block.input.get("command", ""))
                                    if not is_bash_command_scoped_to_cwd(cmd, cwd):
                                        forbidden_tool_error = (
                                            f"[{agent_name}] attempted out-of-project Bash command in plan mode. "
                                            f"cwd={cwd} command={cmd!r}"
                                        )

                            if (
                                isinstance(block, ToolUseBlock)
                                and block.name == "ExitPlanMode"
                            ):
                                captured_plan = _extract_plan_text_from_mapping(
                                    block.input
                                )
                    if isinstance(message, ResultMessage):
                        result_msg = message
            except Exception:
                raise

            if forbidden_tool_error and result_msg is None:
                raise RuntimeError(forbidden_tool_error)
            if forbidden_tool_error and result_msg is not None:
                logger.warning(
                    f"[{agent_name}] forbidden tool attempt was blocked by policy; continuing with available output"
                )
            if result_msg is None:
                raise RuntimeError("No ResultMessage received from planning agent")
            if result_msg.is_error:
                raise RuntimeError(
                    f"Planning agent returned error result: {result_msg.result}"
                )

            if not captured_plan:
                if last_written_plan:
                    logger.warning(
                        f"[{agent_name}] ExitPlanMode missing — using plan content from Write tool call."
                    )
                    captured_plan = last_written_plan

            if not captured_plan:
                candidate: str | None = None
                if isinstance(result_msg.result, str):
                    candidate = result_msg.result.strip() or None
                elif isinstance(result_msg.result, dict):
                    candidate = _extract_plan_text_from_mapping(result_msg.result)

                if candidate and "<system-reminder>" not in candidate:
                    logger.warning(
                        f"[{agent_name}] ExitPlanMode payload missing — using ResultMessage.result as plan fallback."
                    )
                    captured_plan = candidate

            if not captured_plan:
                raise RuntimeError(
                    "Planning agent did not emit ExitPlanMode and produced no "
                    "usable plan output."
                )

            return captured_plan, result_msg.session_id or early_session_id or ""

        except CLINotFoundError:
            raise

        except ProcessError as e:
            stderr = e.stderr or ""
            if "rate limit" in stderr.lower() and attempt < max_retries - 1:
                wait = 60 * (2**attempt)
                logger.warning(f"Rate-limited, waiting {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                logger.warning(f"ProcessError on attempt {attempt + 1}: {e}")

        except CLIJSONDecodeError:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"JSON decode error on attempt {attempt + 1}, retrying")

        except MessageParseError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"MessageParseError on attempt {attempt + 1}: {e}, retrying")

        except RuntimeError as e:
            if "system-reminder" in str(e) or attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt + 1}: {e}, retrying")

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                f"Unexpected error in planning stream on attempt {attempt + 1}: {e}, retrying"
            )

    raise RuntimeError("run_planning_agent: max retries exceeded")
