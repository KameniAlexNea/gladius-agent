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


def test_init_main_guard_invokes_cli_main(monkeypatch):
    """gladius/__init__.py has an `if __name__ == '__main__'` guard that calls cli.main."""
    called = {"ok": False}

    def _main():
        called["ok"] = True

    monkeypatch.setitem(sys.modules, "gladius.cli", SimpleNamespace(main=_main))
    import gladius as _pkg
    import importlib.util, types

    src = importlib.util.find_spec("gladius").origin
    code = compile(open(src, encoding="utf-8").read(), src, "exec")
    ns: dict = {"__name__": "__main__", "__file__": src, "__spec__": None}
    exec(code, ns)  # noqa: S102
    assert called["ok"] is True
