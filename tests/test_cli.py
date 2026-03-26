from __future__ import annotations

import types

import pytest

from gladius.cli import main


def test_cli_exits_when_config_missing(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(SystemExit) as exc:
        main([str(missing)])
    assert exc.value.code == 1


def test_cli_happy_path(monkeypatch, tmp_path):
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text("x: 1\n", encoding="utf-8")

    fake_cfg = {"project_dir": str(tmp_path), "max_iterations": 5}

    fake_project_setup = types.SimpleNamespace(
        load_config=lambda path: fake_cfg,
        setup=lambda path: None,
    )

    called = {}

    async def _fake_run_competition(**kwargs):
        called.update(kwargs)

    fake_orchestrator = types.SimpleNamespace(run_competition=_fake_run_competition)

    monkeypatch.setitem(__import__("sys").modules, "gladius.project_setup", fake_project_setup)
    monkeypatch.setitem(__import__("sys").modules, "gladius.orchestrator", fake_orchestrator)

    main([str(cfg_path), "--max-turns", "11"])

    assert called["competition_dir"] == str(tmp_path)
    assert called["max_turns"] == 11
    assert called["max_iterations"] == 5


def test_cli_exits_on_setup_error(monkeypatch, tmp_path):
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text("x: 1\n", encoding="utf-8")

    fake_project_setup = types.SimpleNamespace(
        load_config=lambda path: {"project_dir": str(tmp_path)},
        setup=lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setitem(__import__("sys").modules, "gladius.project_setup", fake_project_setup)

    with pytest.raises(SystemExit) as exc:
        main([str(cfg_path)])
    assert exc.value.code == 1
