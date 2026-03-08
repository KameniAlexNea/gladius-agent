"""
Jupyter MCP server config — fixed defaults, no user configuration required.

The implementer agent always receives these MCP server definitions so that
``mcp__jupyter__*`` tools are available in every session.  The agent only
actually uses them when it decides to work in a Jupyter notebook; it learns
how to start the required Jupyter and MCP server processes by invoking the
``jupyter-mcp`` skill (see .claude/skills/jupyter-mcp/SKILL.md).

Fixed defaults:
    Jupyter URL   http://localhost:8888
    Token         gladius-mcp
    Notebook      solution.ipynb   (agent can create a different one)
"""

from __future__ import annotations

_JUPYTER_URL = "http://localhost:8888"
_JUPYTER_TOKEN = "gladius-mcp"
_DEFAULT_NOTEBOOK = "solution.ipynb"


def build_jupyter_mcp_config(notebook_id: str = _DEFAULT_NOTEBOOK) -> dict:
    """
    Return an ``mcp_servers`` dict wiring ``jupyter-mcp-server``.

    Passed directly to ``ClaudeAgentOptions(mcp_servers=...)``.
    The server process is started by the agent itself via Bash after reading
    the ``jupyter-mcp`` skill — the orchestrator does not start it.
    """
    return {
        "jupyter": {
            "command": "uv",
            "args": ["run", "-m", "jupyter_mcp_server"],
            "env": {
                "DOCUMENT_URL": _JUPYTER_URL,
                "DOCUMENT_TOKEN": _JUPYTER_TOKEN,
                "DOCUMENT_ID": notebook_id,
                "RUNTIME_URL": _JUPYTER_URL,
                "RUNTIME_TOKEN": _JUPYTER_TOKEN,
            },
        }
    }
