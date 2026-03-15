"""Custom MCP tool servers for Gladius agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.tools.fake_platform_tools import fake_server
from src.tools.kaggle_tools import kaggle_server
from src.tools.zindi_tools import zindi_server

__all__ = ["fake_server", "kaggle_server", "zindi_server", "write_mcp_json"]

_PLATFORM_MODULE = {
    "kaggle": "src.tools.kaggle_tools",
    "zindi":  "src.tools.zindi_tools",
    "fake":   "src.tools.fake_platform_tools",
}
_PLATFORM_SERVER = {
    "kaggle": "kaggle_server",
    "zindi":  "zindi_server",
    "fake":   "fake_server",
}


def write_mcp_json(root: Path, cfg: dict) -> None:
    """Write .mcp.json based on config (always overwritten)."""
    skills_dir = str(root / ".claude" / "skills")
    mcp: dict = {
        "mcpServers": {
            "skills-on-demand": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "skills_on_demand.server"],
                "env": {"SKILLS_DIR": skills_dir},
            }
        }
    }

    platform = cfg.get("platform", "none")
    if cfg.get("mcp", {}).get("platform_server") and platform not in ("none", ""):
        mod = _PLATFORM_MODULE.get(platform)
        srv = _PLATFORM_SERVER.get(platform)
        if mod and srv:
            mcp["mcpServers"][f"{platform}-tools"] = {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-c", f"from {mod} import {srv}; import asyncio; asyncio.run({srv}.run())"],
                "env": {},
            }

    for name, server_cfg in (cfg.get("mcp", {}).get("extra") or {}).items():
        mcp["mcpServers"][name] = server_cfg

    path = root / ".mcp.json"
    path.write_text(json.dumps(mcp, indent=2) + "\n", encoding="utf-8")
    print("  mcp    → .mcp.json")

