"""Runner for schema-driven agent calls."""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Any, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLIJSONDecodeError,
    CLINotFoundError,
    HookMatcher,
    ProcessError,
    ResultMessage,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk.types import (
    AssistantMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ToolUseBlock,
)
from llm_output_parser import parse_json as _parse_json
from loguru import logger

from gladius.roles import ROLE_CATALOG, ROLES
from gladius.roles._console import _BLUE, _BOLD, _c, _log_message
from gladius.roles.helpers import (
    get_runtime_model,
    is_tool_allowed,
    stderr_cb,
    validate_runtime_invocation,
)


class ToolPermissionError(RuntimeError):
    """Raised when an agent or subagent uses a forbidden tool."""


class AgentDispatchError(RuntimeError):
    """Raised when delegation metadata is invalid or missing."""


async def _stream_via_client(prompt: str, options: "ClaudeAgentOptions"):  # type: ignore[return]
    """Yield SDK messages using ClaudeSDKClient instead of the standalone query()."""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            yield message


TraceSink = Callable[[dict[str, Any]], None]


def _emit_trace(trace_sink: TraceSink | None, *, event: str, **payload: Any) -> None:
    """Emit best-effort trace payload to optional sink."""
    if trace_sink is None:
        return
    try:
        trace_sink({"event": event, **payload})
    except Exception as exc:
        logger.debug(f"trace sink dropped event={event!r}: {exc}")


def _merge_hooks(
    base_hooks: dict[str, list[HookMatcher]],
    user_hooks: dict[str, list[HookMatcher]] | None,
) -> dict[str, list[HookMatcher]]:
    """Merge internally generated and caller-provided hook matchers."""
    if not user_hooks:
        return base_hooks
    merged = {k: list(v) for k, v in base_hooks.items()}
    for event, matchers in user_hooks.items():
        merged.setdefault(event, []).extend(matchers)
    return merged


def _build_trace_hooks(
    *,
    agent_name: str,
    trace_sink: TraceSink | None,
    enabled: bool,
) -> dict[str, list[HookMatcher]] | None:
    """Build non-blocking hook callbacks for runtime trace telemetry."""
    if not enabled:
        return None

    async def _trace_hook(
        input_data: dict[str, Any], tool_use_id: str | None, _context: Any
    ) -> dict[str, Any]:
        hook_event = str(input_data.get("hook_event_name", "unknown"))
        tool_name = str(input_data.get("tool_name", ""))
        logger.debug(
            f"[{agent_name}] hook={hook_event}"
            + (f" tool={tool_name}" if tool_name else "")
            + (f" tool_use_id={tool_use_id}" if tool_use_id else "")
        )
        _emit_trace(
            trace_sink,
            event="hook",
            agent_name=agent_name,
            hook_event=hook_event,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=input_data.get("session_id"),
        )
        return {}

    events = [
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "SubagentStart",
        "SubagentStop",
        "Notification",
        "Stop",
    ]
    return {event: [HookMatcher(hooks=[_trace_hook])] for event in events}


def _extract_session_id_from_system_message(message: SystemMessage) -> str | None:
    """Return session id from an init system message when present and valid."""
    if message.subtype != "init" or not isinstance(message.data, dict):
        return None
    session_id = message.data.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def _register_subagent_policy_from_task_start(
    *,
    agent_name: str,
    message: TaskStartedMessage,
    delegated_tool_policies: dict[str, list[str]],
    pending_subagent_tools: collections.deque[list[str]],
) -> None:
    """Bind subagent tool policy using TaskStartedMessage emitter metadata."""
    if message.task_type != "local_agent":
        return

    tool_use_id = message.tool_use_id
    if not isinstance(tool_use_id, str) or not tool_use_id:
        return
    if tool_use_id in delegated_tool_policies:
        return
    if not pending_subagent_tools:
        return

    delegated = pending_subagent_tools.popleft()
    delegated_tool_policies[tool_use_id] = delegated
    logger.debug(
        f"[{agent_name}] mapped local_agent task tool_use_id "
        f"{tool_use_id!r} to delegated policy {delegated!r} via TaskStartedMessage."
    )


def _extract_subagent_type(block_input: dict[str, Any]) -> str:
    """Extract subagent selector from any supported key alias."""
    for key in ("subagent_type",):
        value = block_input.get(key)
        if value:
            return str(value).strip()
    return ""


def _parse_structured_from_assistant_text(
    *, agent_name: str, last_assistant_msg: AssistantMessage | None
) -> dict[str, Any] | None:
    """Attempt to recover structured JSON from assistant text blocks."""
    if last_assistant_msg is None:
        return None

    full_text = "\n".join(
        block.text.strip()
        for block in last_assistant_msg.content
        if isinstance(block, TextBlock) and block.text.strip()
    )
    if not full_text:
        return None

    try:
        parsed = _parse_json(full_text)
    except Exception as exc:
        logger.warning(f"[{agent_name}] structured_output fallback parse failed: {exc}")
        return None

    if isinstance(parsed, dict):
        return parsed

    logger.warning(
        f"[{agent_name}] structured_output fallback produced {type(parsed).__name__}, "
        "ignoring non-dict parsed output."
    )
    return None


def _register_delegated_subagent_tools(
    *,
    agent_name: str,
    block: ToolUseBlock,
    delegated_tool_policies: dict[str, list[str]],
    pending_subagent_tools: collections.deque[list[str]],
) -> str | None:
    """Validate delegation call and register subagent tool policy."""
    subagent_type = _extract_subagent_type(block.input)

    if subagent_type in ROLE_CATALOG:
        tools = list(ROLE_CATALOG[subagent_type].tools)
        delegated_tool_policies[block.id] = tools
        pending_subagent_tools.append(tools)
        return None

    msg = (
        f"[{agent_name}] Agent called without a valid "
        f"subagent_type (got {subagent_type!r}). "
        f"You MUST pass subagent_type as one of: {list(ROLES)}. "
        f'Example: Agent({{"subagent_type": "feature-engineer", "prompt": "..."}})'
    )
    logger.error(msg)
    return msg


def _resolve_effective_allowed_tools(
    *,
    agent_name: str,
    message_parent_tool_use_id: str | None,
    default_allowed_tools: list[str],
    delegated_tool_policies: dict[str, list[str]],
    pending_subagent_tools: collections.deque[list[str]],
) -> tuple[list[str], str]:
    """Resolve tool policy for a block, including subagent FIFO fallback."""
    effective_allowed_tools = default_allowed_tools
    policy_label = f"allowed_tools={default_allowed_tools}"

    if message_parent_tool_use_id:
        delegated = delegated_tool_policies.get(message_parent_tool_use_id)
        if delegated is None and pending_subagent_tools:
            # The CLI assigned a different parent_tool_use_id than the block.id
            # stored at delegation time; resolve policy via FIFO fallback.
            logger.debug(
                f"[{agent_name}] parent_tool_use_id "
                f"{message_parent_tool_use_id!r} not in delegated_tool_policies "
                f"(keys={list(delegated_tool_policies)!r}); "
                "applying FIFO subagent policy fallback."
            )
            delegated = pending_subagent_tools.popleft()
            delegated_tool_policies[message_parent_tool_use_id] = delegated
        if delegated is not None:
            effective_allowed_tools = delegated
            policy_label = f"subagent_allowed_tools={effective_allowed_tools}"

    return effective_allowed_tools, policy_label


def _handle_tool_use_block(
    *,
    agent_name: str,
    block: ToolUseBlock,
    message_parent_tool_use_id: str | None,
    allowed_tools: list[str],
    delegated_tool_policies: dict[str, list[str]],
    pending_subagent_tools: collections.deque[list[str]],
) -> str | None:
    """Validate delegation metadata and tool permissions for one tool-use block."""
    if block.name in {"Task", "Agent"}:
        err = _register_delegated_subagent_tools(
            agent_name=agent_name,
            block=block,
            delegated_tool_policies=delegated_tool_policies,
            pending_subagent_tools=pending_subagent_tools,
        )
        if err is not None:
            return err
        return None

    effective_allowed_tools, policy_label = _resolve_effective_allowed_tools(
        agent_name=agent_name,
        message_parent_tool_use_id=message_parent_tool_use_id,
        default_allowed_tools=allowed_tools,
        delegated_tool_policies=delegated_tool_policies,
        pending_subagent_tools=pending_subagent_tools,
    )
    is_subagent = bool(message_parent_tool_use_id)
    if not is_tool_allowed(block.name, effective_allowed_tools):
        msg = f"[{agent_name}] attempted forbidden tool '{block.name}'. {policy_label}"
        if is_subagent:
            logger.error(f"sub-agent policy violation: {msg}")
        return msg
    return None


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
    trace_sink: TraceSink | None = None,
    trace_context: dict[str, Any] | None = None,
    enable_trace_hooks: bool = True,
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
    trace_context = trace_context or {}

    # Enable StructuredOutput session-wide when any registered subagent declares it,
    # so subagents can emit structured results even if the outer agent does not.
    if output_schema is not None:
        _output_format: dict[str, Any] | None = {
            "type": "json_schema",
            "schema": output_schema,
        }
    elif any("StructuredOutput" in role.tools for role in ROLE_CATALOG.values()):
        _output_format = {
            "type": "json_schema",
            "schema": {"type": "object", "additionalProperties": True},
        }
    else:
        _output_format = None

    include_partial_messages = bool(
        option_kwargs.pop("include_partial_messages", verbose)
    )
    user_hooks = option_kwargs.pop("hooks", None)
    trace_hooks = _build_trace_hooks(
        agent_name=agent_name,
        trace_sink=trace_sink,
        enabled=enable_trace_hooks,
    )
    merged_hooks = _merge_hooks(trace_hooks or {}, user_hooks)

    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt,
        },
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
        output_format=_output_format,
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=stderr_cb,
        setting_sources=["project"],
        include_partial_messages=include_partial_messages,
        hooks=merged_hooks or None,
        agents=ROLE_CATALOG,
        model=runtime_model,
        **option_kwargs,
    )

    with logger.contextualize(
        run_id=str(trace_context.get("run_id", "-")),
        iteration=str(trace_context.get("iteration", "-")),
        attempt=str(trace_context.get("attempt", "-")),
        agent=agent_name,
    ):
        for attempt in range(max_retries):
            attempt_no = attempt + 1
            attempt_started = time.monotonic()
            try:
                result_msg: ResultMessage | None = None
                last_assistant_msg: AssistantMessage | None = None
                forbidden_tool_error: str | None = None
                early_session_id: str | None = None
                delegated_tool_policies: dict[str, list[str]] = {}
                # FIFO queue of tool-lists for dispatched subagents.
                # Used as a fallback when the CLI assigns a different parent_tool_use_id
                # to subagent messages than the block.id stored in delegated_tool_policies.
                _pending_subagent_tools: collections.deque[list[str]] = (
                    collections.deque()
                )
                # tool_use_id → subagent name, task_id → subagent name
                _trace_subagent_by_tool_id: dict[str, str] = {}
                _trace_subagent_by_task_id: dict[str, str] = {}

                with logger.contextualize(attempt=str(attempt_no)):
                    if verbose:
                        resume_str = f"  resume={resume[:8]}…" if resume else ""
                        logger.debug(
                            _c(_BOLD + _BLUE, f"\n▶ [{agent_name}]")
                            + f"  tools={allowed_tools}"
                            + resume_str
                        )
                    logger.info(
                        f"[{agent_name}] attempt {attempt_no}/{max_retries} started"
                    )

                    _emit_trace(
                        trace_sink,
                        event="attempt_start",
                        agent_name=agent_name,
                        attempt=attempt_no,
                        max_retries=max_retries,
                        resume=resume,
                    )

                    async for message in _stream_via_client(
                        prompt=prompt, options=options
                    ):
                        if verbose:
                            _log_message(agent_name, message)
                        if isinstance(message, StreamEvent):
                            if (
                                isinstance(message.session_id, str)
                                and message.session_id
                            ):
                                early_session_id = message.session_id
                            _sub = _trace_subagent_by_tool_id.get(
                                message.parent_tool_use_id or "", ""
                            )
                            _emit_trace(
                                trace_sink,
                                event="stream_event",
                                agent_name=agent_name,
                                subagent_name=_sub or None,
                                session_id=message.session_id,
                                parent_tool_use_id=message.parent_tool_use_id,
                                stream_type=message.event.get("type"),
                            )
                        if isinstance(message, SystemMessage):
                            session_id = _extract_session_id_from_system_message(
                                message
                            )
                            if session_id:
                                early_session_id = session_id
                        if isinstance(message, TaskStartedMessage):
                            if (
                                isinstance(message.session_id, str)
                                and message.session_id
                            ):
                                early_session_id = message.session_id
                            _register_subagent_policy_from_task_start(
                                agent_name=agent_name,
                                message=message,
                                delegated_tool_policies=delegated_tool_policies,
                                pending_subagent_tools=_pending_subagent_tools,
                            )
                            _sub = _trace_subagent_by_tool_id.get(
                                message.tool_use_id or "", ""
                            )
                            if _sub and message.task_id:
                                _trace_subagent_by_task_id[message.task_id] = _sub
                            _emit_trace(
                                trace_sink,
                                event="task_started",
                                agent_name=agent_name,
                                subagent_name=_sub or None,
                                task_id=message.task_id,
                                task_type=message.task_type,
                                session_id=message.session_id,
                                tool_use_id=message.tool_use_id,
                            )
                        if isinstance(message, TaskProgressMessage):
                            _sub = _trace_subagent_by_task_id.get(
                                message.task_id or "", ""
                            )
                            _emit_trace(
                                trace_sink,
                                event="task_progress",
                                agent_name=agent_name,
                                subagent_name=_sub or None,
                                task_id=message.task_id,
                                description=message.description,
                                last_tool_name=message.last_tool_name,
                                usage=message.usage,
                            )
                        if isinstance(message, TaskNotificationMessage):
                            _sub = _trace_subagent_by_task_id.get(
                                message.task_id or "", ""
                            )
                            _emit_trace(
                                trace_sink,
                                event="task_notification",
                                agent_name=agent_name,
                                subagent_name=_sub or None,
                                task_id=message.task_id,
                                status=message.status,
                                summary=message.summary,
                                usage=message.usage,
                            )
                        if isinstance(message, AssistantMessage):
                            if forbidden_tool_error is None:
                                for block in message.content:
                                    if isinstance(block, ToolUseBlock):
                                        # Record subagent name before policy check
                                        if block.name in {"Task", "Agent"}:
                                            _stype = _extract_subagent_type(block.input)
                                            if _stype:
                                                _trace_subagent_by_tool_id[block.id] = (
                                                    _stype
                                                )
                                        forbidden_tool_error = _handle_tool_use_block(
                                            agent_name=agent_name,
                                            block=block,
                                            message_parent_tool_use_id=message.parent_tool_use_id,
                                            allowed_tools=allowed_tools,
                                            delegated_tool_policies=delegated_tool_policies,
                                            pending_subagent_tools=_pending_subagent_tools,
                                        )
                                        if forbidden_tool_error is not None:
                                            _emit_trace(
                                                trace_sink,
                                                event="policy_violation",
                                                agent_name=agent_name,
                                                message=forbidden_tool_error,
                                            )
                                            logger.warning(
                                                f"[{agent_name}] policy violation — aborting stream."
                                            )
                                            break
                            last_assistant_msg = message
                            if forbidden_tool_error is not None:
                                break
                        if isinstance(message, ResultMessage):
                            result_msg = message
                            _emit_trace(
                                trace_sink,
                                event="result_message",
                                agent_name=agent_name,
                                session_id=message.session_id,
                                subtype=message.subtype,
                                stop_reason=message.stop_reason,
                                is_error=message.is_error,
                                duration_ms=message.duration_ms,
                                duration_api_ms=message.duration_api_ms,
                                num_turns=message.num_turns,
                                usage=message.usage,
                                total_cost_usd=message.total_cost_usd,
                            )

                if forbidden_tool_error:
                    if "without a valid subagent_type" in forbidden_tool_error:
                        raise AgentDispatchError(forbidden_tool_error)
                    raise ToolPermissionError(forbidden_tool_error)
                if result_msg is None:
                    raise RuntimeError("No ResultMessage received from agent")

                structured = result_msg.structured_output

                if output_schema is not None:
                    if structured is None and last_assistant_msg is not None:
                        structured = _parse_structured_from_assistant_text(
                            agent_name=agent_name, last_assistant_msg=last_assistant_msg
                        )
                        if structured is not None:
                            logger.warning(
                                f"[{agent_name}] structured_output was None — "
                                "extracted JSON from assistant text (llm_output_parser fallback)"
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
                    # output_schema is None: outer agent (e.g. orchestrator) doesn't need
                    # structured output itself.  If output_format was set for session-wide
                    # StructuredOutput support, the outer agent never calls StructuredOutput,
                    # so is_error may be True only because "no structured output produced".
                    # Only raise for genuine failures (not schema-enforcement errors).
                    if result_msg.is_error and _output_format is None:
                        raise RuntimeError(
                            f"Agent returned error result: {result_msg.result}"
                        )
                    if result_msg.is_error and _output_format is not None:
                        logger.debug(
                            f"[{agent_name}] session is_error=True with permissive output_format "
                            f"(subagents consumed StructuredOutput): {result_msg.result}"
                        )

                elapsed_ms = int((time.monotonic() - attempt_started) * 1000)
                logger.info(
                    f"[{agent_name}] attempt {attempt_no}/{max_retries} completed "
                    f"in {elapsed_ms}ms subtype={result_msg.subtype!r} "
                    f"stop_reason={result_msg.stop_reason!r}"
                )
                _emit_trace(
                    trace_sink,
                    event="attempt_success",
                    agent_name=agent_name,
                    attempt=attempt_no,
                    elapsed_ms=elapsed_ms,
                    session_id=result_msg.session_id or early_session_id,
                    subtype=result_msg.subtype,
                    stop_reason=result_msg.stop_reason,
                    is_error=result_msg.is_error,
                )

                return structured or {}, result_msg.session_id or early_session_id or ""

            except CLINotFoundError:
                raise

            except ProcessError as e:
                elapsed_ms = int((time.monotonic() - attempt_started) * 1000)
                structured_fallback = (
                    result_msg.structured_output if result_msg else None
                )
                source = "structured_output"
                if structured_fallback is None:
                    recovered = _parse_structured_from_assistant_text(
                        agent_name=agent_name, last_assistant_msg=last_assistant_msg
                    )
                    if recovered is not None:
                        structured_fallback = recovered
                        source = "assistant_text_fallback"

                if structured_fallback is not None:
                    logger.warning(
                        f"[{agent_name}] ProcessError with recoverable output from {source} "
                        "— returning partial result (CLI exited abnormally during cleanup)."
                    )
                    if not isinstance(structured_fallback, dict):
                        logger.warning(
                            f"[{agent_name}] recovered output is {type(structured_fallback).__name__}, "
                            "coercing to empty dict for stable return type."
                        )
                        structured_fallback = {}
                    _emit_trace(
                        trace_sink,
                        event="attempt_partial",
                        agent_name=agent_name,
                        attempt=attempt_no,
                        elapsed_ms=elapsed_ms,
                        source=source,
                    )
                    return (
                        structured_fallback,
                        (result_msg.session_id if result_msg else None)
                        or early_session_id
                        or "",
                    )
                stderr = e.stderr or ""
                if "rate limit" in stderr.lower() and attempt < max_retries - 1:
                    wait = 60 * (2**attempt)
                    logger.warning(
                        f"Rate-limited, waiting {wait}s (attempt {attempt_no}/{max_retries})"
                    )
                    _emit_trace(
                        trace_sink,
                        event="attempt_retry_wait",
                        agent_name=agent_name,
                        attempt=attempt_no,
                        elapsed_ms=elapsed_ms,
                        wait_seconds=wait,
                        reason="rate_limit",
                    )
                    await asyncio.sleep(wait)
                elif attempt == max_retries - 1:
                    raise
                else:
                    logger.warning(
                        f"ProcessError on attempt {attempt_no}/{max_retries} after {elapsed_ms}ms: {e}"
                    )
                    _emit_trace(
                        trace_sink,
                        event="attempt_retry",
                        agent_name=agent_name,
                        attempt=attempt_no,
                        elapsed_ms=elapsed_ms,
                        reason="process_error",
                        error=str(e),
                    )

            except CLIJSONDecodeError:
                elapsed_ms = int((time.monotonic() - attempt_started) * 1000)
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"JSON decode error on attempt {attempt_no}/{max_retries} after {elapsed_ms}ms, retrying"
                )
                _emit_trace(
                    trace_sink,
                    event="attempt_retry",
                    agent_name=agent_name,
                    attempt=attempt_no,
                    elapsed_ms=elapsed_ms,
                    reason="json_decode",
                )

            except MessageParseError as e:
                elapsed_ms = int((time.monotonic() - attempt_started) * 1000)
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"MessageParseError on attempt {attempt_no}/{max_retries} "
                    f"after {elapsed_ms}ms: {e}, retrying"
                )
                _emit_trace(
                    trace_sink,
                    event="attempt_retry",
                    agent_name=agent_name,
                    attempt=attempt_no,
                    elapsed_ms=elapsed_ms,
                    reason="message_parse",
                    error=str(e),
                )

            except (ToolPermissionError, AgentDispatchError):
                raise

            except RuntimeError as e:
                elapsed_ms = int((time.monotonic() - attempt_started) * 1000)
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"RuntimeError on attempt {attempt_no}/{max_retries} "
                    f"after {elapsed_ms}ms: {e}, retrying"
                )
                _emit_trace(
                    trace_sink,
                    event="attempt_retry",
                    agent_name=agent_name,
                    attempt=attempt_no,
                    elapsed_ms=elapsed_ms,
                    reason="runtime_error",
                    error=str(e),
                )

    raise RuntimeError("run_agent: max retries exceeded")
