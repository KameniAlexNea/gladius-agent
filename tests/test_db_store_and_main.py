from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from gladius.db.store import StateStore


def test_state_store_roundtrip_and_noops():
    s = StateStore()
    obj = {"a": 1}
    s.save(obj)
    assert s.load() == obj
    s.close()
    s.record_event(iteration=1, topology="t", event="e")
    s.record_plan(iteration=1, approach_summary="a", plan_text="p", session_id=None)
    s.record_code_snapshots(iteration=1, solution_files=[], project_dir=".")
    s.record_agent_run(x=1)


def test_main_invokes_cli_main(monkeypatch):
    called = {"ok": False}

    def _main():
        called["ok"] = True

    monkeypatch.setitem(sys.modules, "gladius.cli", SimpleNamespace(main=_main))
    if "gladius.__main__" in sys.modules:
        del sys.modules["gladius.__main__"]
    importlib.import_module("gladius.__main__")
    assert called["ok"] is True
