"""
Shared agent runner helper.

Every agent calls run_agent() which wraps claude_agent_sdk.query() and
returns (structured_output, session_id).
"""
import asyncio
import logging
from typing import Any

from claude_agent_sdk import (
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ProcessError,
    ResultMessage,
    query,
)

logger = logging.getLogger(__name__)


async def run_agent(
    *,
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    output_schema: dict[str, Any],
    cwd: str,
    resume: str | None = None,
    mcp_servers: dict | None = None,
    max_turns: int | None = None,
    max_retries: int = 3,
) -> tuple[dict[str, Any], str]:
    """
    Run a single Claude agent call.

    Returns
    -------
    (structured_output, session_id)
    """
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        output_format={"type": "json_schema", "schema": output_schema},
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_msg = message

            if result_msg is None:
                raise RuntimeError("No ResultMessage received from agent")
            if result_msg.is_error:
                raise RuntimeError(f"Agent returned error result: {result_msg.result}")
            if result_msg.structured_output is None:
                raise RuntimeError("Agent returned no structured_output (schema not satisfied?)")

            return result_msg.structured_output, result_msg.session_id

        except CLINotFoundError:
            raise  # Fatal — Claude Code CLI not installed

        except ProcessError as e:
            stderr = e.stderr or ""
            if "rate limit" in stderr.lower() and attempt < max_retries - 1:
                wait = 60 * (2**attempt)
                logger.warning(f"Rate-limited, waiting {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                logger.warning(f"ProcessError on attempt {attempt+1}: {e}")

        except CLIJSONDecodeError:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"JSON decode error on attempt {attempt+1}, retrying")

        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt+1}: {e}, retrying")

    raise RuntimeError("run_agent: max retries exceeded")
