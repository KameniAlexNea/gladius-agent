"""Runner for schema-driven agent calls."""

from __future__ import annotations

import asyncio
import collections
from typing import Any

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
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from llm_output_parser import parse_json as _parse_json
from loguru import logger

from gladius.roles import ROLE_CATALOG, ROLES
from gladius.roles._console import _BLUE, _BOLD, _c, _log_message
from gladius.roles.helpers import (
    get_runtime_model,
    is_bash_command_scoped_to_cwd,
    is_tool_allowed,
    stderr_cb,
    validate_runtime_invocation,
)


async def run_agent(
    *,
    agent_name: str = "agent",
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    output_schema: dict[str, Any] | None = None,
    cwd: str = "",
    resume: str | None = None,
    mcp_servers: dict | None = None,
    max_turns: int | None = None,
    max_retries: int = 3,
    verbose: bool = True,
    permission_mode: str = "bypassPermissions",
    **option_kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """Run a single Claude agent call, streaming all messages to stdout."""
    validate_runtime_invocation(
        agent_name=agent_name,
        cwd=cwd,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )

    runtime_model = get_runtime_model()
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt,
        },
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
        output_format=(
            {"type": "json_schema", "schema": output_schema}
            if output_schema is not None
            else None
        ),
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=stderr_cb,
        setting_sources=["project"],
        agents=ROLE_CATALOG,
        model=runtime_model,
        **option_kwargs,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            last_assistant_msg: AssistantMessage | None = None
            forbidden_tool_error: str | None = None
            early_session_id: str | None = None
            delegated_tool_policies: dict[str, list[str]] = {}
            # FIFO queue of tool-lists for dispatched subagents.
            # Used as a fallback when the CLI assigns a different parent_tool_use_id
            # to subagent messages than the block.id stored in delegated_tool_policies.
            _pending_subagent_tools: collections.deque[list[str]] = collections.deque()

            if verbose:
                resume_str = f"  resume={resume[:8]}…" if resume else ""
                logger.debug(
                    _c(_BOLD + _BLUE, f"\n▶ [{agent_name}]")
                    + f"  tools={allowed_tools}"
                    + resume_str
                )

            async for message in query(prompt=prompt, options=options):
                if verbose:
                    _log_message(agent_name, message)
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    early_session_id = message.data.get("session_id")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            if block.name in {"Task", "Agent"}:
                                subagent_type = str(
                                    block.input.get("subagent_type")
                                    or block.input.get("agent_name")
                                    or block.input.get("agent")
                                    or block.input.get("name")
                                    or ""
                                )
                                if subagent_type in ROLE_CATALOG:
                                    tools = list(
                                        ROLE_CATALOG[subagent_type].tools
                                    )
                                    delegated_tool_policies[block.id] = tools
                                    _pending_subagent_tools.append(tools)
                                else:
                                    msg = (
                                        f"[{agent_name}] Agent called without a valid "
                                        f"agent_name (got {subagent_type!r}). "
                                        f"You MUST pass agent_name as one of: {list(ROLES)}. "
                                        f"Example: Agent({{\"agent_name\": \"feature-engineer\", \"prompt\": \"...\"}})"
                                    )
                                    logger.error(msg)
                                    raise RuntimeError(msg)

                            effective_allowed_tools = allowed_tools
                            policy_label = f"allowed_tools={allowed_tools}"
                            if message.parent_tool_use_id:
                                delegated = delegated_tool_policies.get(
                                    message.parent_tool_use_id
                                )
                                if delegated is None and _pending_subagent_tools:
                                    # The CLI assigned a different parent_tool_use_id
                                    # than the block.id we stored — resolve via FIFO.
                                    logger.debug(
                                        f"[{agent_name}] parent_tool_use_id "
                                        f"{message.parent_tool_use_id!r} not in "
                                        f"delegated_tool_policies (keys="
                                        f"{list(delegated_tool_policies)!r}); "
                                        "applying FIFO subagent policy fallback."
                                    )
                                    delegated = _pending_subagent_tools.popleft()
                                    delegated_tool_policies[
                                        message.parent_tool_use_id
                                    ] = delegated
                                if delegated is not None:
                                    effective_allowed_tools = delegated
                                    policy_label = f"subagent_allowed_tools={effective_allowed_tools}"

                            is_subagent = bool(message.parent_tool_use_id)
                            if not is_tool_allowed(block.name, effective_allowed_tools):
                                msg = (
                                    f"[{agent_name}] attempted forbidden tool '{block.name}'. "
                                    f"{policy_label}"
                                )
                                forbidden_tool_error = msg
                                if is_subagent:
                                    logger.error(f"sub-agent policy violation: {msg}")
                            elif block.name == "Bash":
                                cmd = str(block.input.get("command", ""))
                                if not is_bash_command_scoped_to_cwd(cmd, cwd):
                                    msg = (
                                        f"[{agent_name}] attempted out-of-project Bash command. "
                                        f"cwd={cwd} command={cmd!r}"
                                    )
                                    forbidden_tool_error = msg
                                    if is_subagent:
                                        logger.error(
                                            f"sub-agent scope violation: {msg}"
                                        )
                    last_assistant_msg = message
                if isinstance(message, ResultMessage):
                    result_msg = message

            if forbidden_tool_error:
                raise RuntimeError(forbidden_tool_error)
            if result_msg is None:
                raise RuntimeError("No ResultMessage received from agent")

            structured = result_msg.structured_output

            if output_schema is not None:
                if structured is None and last_assistant_msg is not None:
                    full_text = "\n".join(
                        block.text.strip()
                        for block in last_assistant_msg.content
                        if isinstance(block, TextBlock) and block.text.strip()
                    )
                    if full_text:
                        try:
                            structured = _parse_json(full_text)
                            logger.warning(
                                f"[{agent_name}] structured_output was None — "
                                "extracted JSON from assistant text (llm_output_parser fallback)"
                            )
                        except Exception as exc:
                            logger.warning(
                                f"[{agent_name}] structured_output fallback parse failed: {exc}"
                            )

                if result_msg.is_error and structured is None:
                    raise RuntimeError(
                        f"Agent returned error result: {result_msg.result}"
                    )
                if result_msg.is_error and structured is not None:
                    logger.warning(
                        f"[{agent_name}] ResultMessage is_error=True but structured_output "
                        "was captured — using it (model ran extra turns after StructuredOutput)."
                    )

                if structured is None:
                    raise RuntimeError(
                        "Agent returned no structured_output (schema not satisfied?)"
                    )
            else:
                if result_msg.is_error:
                    raise RuntimeError(
                        f"Agent returned error result: {result_msg.result}"
                    )

            return structured or {}, result_msg.session_id or early_session_id or ""

        except CLINotFoundError:
            raise

        except ProcessError as e:
            if result_msg is not None and result_msg.structured_output is not None:
                logger.warning(
                    f"[{agent_name}] ProcessError after StructuredOutput captured"
                    " — returning result (CLI exited abnormally during cleanup)."
                )
                return (
                    result_msg.structured_output,
                    result_msg.session_id or early_session_id or "",
                )
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
            if "attempted forbidden tool" in str(e):
                raise
            if attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt + 1}: {e}, retrying")

    raise RuntimeError("run_agent: max retries exceeded")
