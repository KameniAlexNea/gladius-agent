from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from gladius.tools import write_mcp_json
from gladius.tools._response import err, ok
from gladius.tools.fake_platform_tools import (
    _load_history,
    _save_history,
    fake_leaderboard,
    fake_status,
    fake_submission_history,
    fake_submit,
)
from gladius.tools.kaggle_tools import (
    kaggle_leaderboard,
    kaggle_submission_history,
    kaggle_submit,
)
from gladius.tools.zindi_tools import (
    zindi_leaderboard,
    zindi_status,
    zindi_submission_history,
    zindi_submit,
)


def test_response_ok_and_err_shapes():
    assert ok("x")["status"] == "ok"
    e = err("t", "bad")
    assert e["status"] == "error"
    assert e["is_error"] is True


def test_write_mcp_json_with_platform_and_extra(tmp_path: Path):
    cfg = {
        "platform": "fake",
        "mcp": {"platform_server": True, "extra": {"x": {"command": "echo"}}},
    }
    write_mcp_json(tmp_path, cfg)
    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "skills-on-demand" in data["mcpServers"]
    assert "fake-tools" in data["mcpServers"]
    assert "arxiv-mcp-server" in data["mcpServers"]
    assert "x" in data["mcpServers"]


def test_kaggle_tools_success_and_failure(monkeypatch):
    def _run_ok(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def _run_fail(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad")

    monkeypatch.setattr("gladius.tools.kaggle_tools.subprocess.run", _run_ok)
    res = asyncio.run(
        kaggle_submit.handler(
            {"competition": "c", "file_path": "f.csv", "message": "m"}
        )
    )
    assert res["status"] == "ok"

    res = asyncio.run(kaggle_leaderboard.handler({"competition": "c", "top_n": 2}))
    assert res["status"] == "ok"

    monkeypatch.setattr("gladius.tools.kaggle_tools.subprocess.run", _run_fail)
    res = asyncio.run(kaggle_submission_history.handler({"competition": "c"}))
    assert res["status"] == "error"


def test_fake_platform_history_save_and_load(monkeypatch, tmp_path: Path):
    import dataclasses
    import gladius.tools.fake_platform_tools as _mod

    monkeypatch.setattr(_mod, "_SETTINGS", dataclasses.replace(_mod._SETTINGS, fake_platform_dir=str(tmp_path / "p")))
    _save_history([{"score": 1.0}])
    got = _load_history()
    assert got[0]["score"] == 1.0


def test_fake_score_submission_and_tools(monkeypatch, tmp_path: Path):
    sub = tmp_path / "sub.csv"
    sub.write_text("id,target\n1,0.5\n", encoding="utf-8")
    import dataclasses
    import gladius.tools.fake_platform_tools as _mod

    monkeypatch.setattr(_mod, "_SETTINGS", dataclasses.replace(_mod._SETTINGS, fake_platform_dir=str(tmp_path / "fp")))
    monkeypatch.setattr(
        "gladius.tools.fake_platform_tools._score_submission", lambda p: 0.75
    )

    res = asyncio.run(fake_submit.handler({"file_path": str(sub), "comment": "c"}))
    assert res["status"] == "ok"

    lb = asyncio.run(fake_leaderboard.handler({"top_n": 5}))
    assert lb["status"] == "ok"

    hist = asyncio.run(fake_submission_history.handler({}))
    assert hist["status"] == "ok"

    st = asyncio.run(fake_status.handler({}))
    assert st["status"] == "ok"


def test_zindi_tools_with_fake_user(monkeypatch):
    class _Df:
        def head(self, n):
            return self

        def to_string(self):
            return "table"

    class _User:
        remaining_subimissions = 2
        which_challenge = "challenge"
        my_rank = 3

        def submit(self, filepaths, comments):
            self.remaining_subimissions = 1

        def leaderboard(self):
            return _Df()

        def submission_board(self):
            return _Df()

    monkeypatch.setattr("gladius.tools.zindi_tools._get_user", lambda: _User())

    assert (
        asyncio.run(zindi_submit.handler({"file_path": "f.csv", "comment": "c"}))[
            "status"
        ]
        == "ok"
    )
    assert asyncio.run(zindi_leaderboard.handler({"top_n": 3}))["status"] == "ok"
    assert asyncio.run(zindi_submission_history.handler({}))["status"] == "ok"
    assert asyncio.run(zindi_status.handler({}))["status"] == "ok"
