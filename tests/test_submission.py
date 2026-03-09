import pytest

from gladius.submission import _select_zindi_challenge


class _DummyUser:
    def __init__(self, fail_challenge: bool = False):
        self.fail_challenge = fail_challenge
        self.calls = []

    def select_a_challenge(self, **kwargs):
        self.calls.append(kwargs)
        if "challenge_id" in kwargs and self.fail_challenge:
            raise RuntimeError("challenge not found")


def test_select_zindi_challenge_prefers_competition_id(monkeypatch):
    monkeypatch.delenv("ZINDI_CHALLENGE_INDEX", raising=False)
    user = _DummyUser()

    _select_zindi_challenge(user=user, competition_id="financial-well-being-sme")

    assert user.calls == [{"challenge_id": "financial-well-being-sme"}]


def test_select_zindi_challenge_falls_back_to_index(monkeypatch):
    monkeypatch.setenv("ZINDI_CHALLENGE_INDEX", "2")
    user = _DummyUser(fail_challenge=True)

    _select_zindi_challenge(user=user, competition_id="financial-well-being-sme")

    assert user.calls == [
        {"challenge_id": "financial-well-being-sme"},
        {"fixed_index": 2},
    ]


def test_select_zindi_challenge_raises_when_no_fallback(monkeypatch):
    monkeypatch.delenv("ZINDI_CHALLENGE_INDEX", raising=False)
    user = _DummyUser(fail_challenge=True)

    with pytest.raises(RuntimeError, match="Could not resolve Zindi challenge"):
        _select_zindi_challenge(user=user, competition_id="financial-well-being-sme")
