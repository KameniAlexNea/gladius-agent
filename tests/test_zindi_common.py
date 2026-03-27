from __future__ import annotations

import sys
import types

import pytest

import gladius.tools.zindi_common as zc


class _User:
    def __init__(self):
        self.which_challenge = ""
        self._Zindian__challenge_data = {}

    def select_a_challenge(self, challenge_id=None, fixed_index=None):
        if challenge_id:
            if challenge_id == "bad":
                raise RuntimeError("bad")
            self.which_challenge = challenge_id
            self._Zindian__challenge_data = {"id": challenge_id}
        else:
            self.which_challenge = f"idx-{fixed_index}"
            self._Zindian__challenge_data = {"id": self.which_challenge}


def test_get_selected_challenge_id_fallback_data():
    u = _User()
    u._Zindian__challenge_data = {"slug": "s"}
    assert zc._get_selected_challenge_id(u) == "s"


def test_select_zindi_challenge_prefers_competition_id():
    u = _User()
    got = zc.select_zindi_challenge(
        user=u,
        competition_id="comp-a",
        env_challenge_id="env-b",
        env_challenge_index="1",
    )
    assert got == "comp-a"


def test_select_zindi_challenge_uses_index_when_ids_fail():
    u = _User()
    got = zc.select_zindi_challenge(
        user=u,
        competition_id="bad",
        env_challenge_id="bad",
        env_challenge_index="2",
    )
    assert got == "idx-2"


def test_select_zindi_challenge_invalid_index_raises():
    with pytest.raises(RuntimeError):
        zc.select_zindi_challenge(
            user=_User(),
            competition_id=None,
            env_challenge_id=None,
            env_challenge_index="nope",
        )


def test_create_zindi_user_from_env(monkeypatch):
    monkeypatch.setenv("ZINDI_USERNAME", "u")
    monkeypatch.setenv("ZINDI_PASSWORD", "p")

    class _DummyZindian:
        def __init__(self, username, fixed_password):
            self.username = username
            self.fixed_password = fixed_password

    fake_user_module = types.ModuleType("zindi.user")
    fake_user_module.Zindian = _DummyZindian
    monkeypatch.setitem(sys.modules, "zindi", types.ModuleType("zindi"))
    monkeypatch.setitem(sys.modules, "zindi.user", fake_user_module)

    out = zc.create_zindi_user_from_env()
    assert out.username == "u"
    assert out.fixed_password == "p"


def test_create_zindi_user_from_env_missing_creds(monkeypatch):
    monkeypatch.delenv("ZINDI_USERNAME", raising=False)
    monkeypatch.delenv("ZINDI_PASSWORD", raising=False)
    monkeypatch.delenv("USER_NAME", raising=False)
    monkeypatch.delenv("PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        zc.create_zindi_user_from_env()
