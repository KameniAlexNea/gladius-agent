"""Tests for project setup helpers."""

import json
from pathlib import Path

from gladius.state import CompetitionState
from gladius.utils.project_setup import setup_project_dir


def _state(tmp_path: Path) -> CompetitionState:
    return CompetitionState(
        competition_id="comp-1",
        data_dir=str((tmp_path / "data").resolve()),
        output_dir=str((tmp_path / ".gladius").resolve()),
        target_metric="auc_roc",
        metric_direction="maximize",
    )


def test_setup_writes_fake_mcp_server(tmp_path):
    state = _state(tmp_path)
    setup_project_dir(state, str(tmp_path), platform="fake")

    mcp_path = tmp_path / ".mcp.json"
    cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "fake-tools" in cfg["mcpServers"]
    cmd = cfg["mcpServers"]["fake-tools"]["args"][1]
    assert "fake_server" in cmd
    assert "fake_server.run()" in cmd
