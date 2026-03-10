import pytest

from gladius.submission import _select_zindi_challenge


class _DummyUser:
    def __init__(
        self, fail_challenge: bool = False, silent_challenge_miss: bool = False
    ):
        self.fail_challenge = fail_challenge
        self.silent_challenge_miss = silent_challenge_miss
        self.calls = []
        self._selected = None

    @property
    def which_challenge(self):
        return self._selected

    def select_a_challenge(self, **kwargs):
        self.calls.append(kwargs)
        if "challenge_id" in kwargs and self.fail_challenge:
            raise RuntimeError("challenge not found")
        if "challenge_id" in kwargs and not self.silent_challenge_miss:
            self._selected = kwargs["challenge_id"]
        if "fixed_index" in kwargs:
            self._selected = f"idx-{kwargs['fixed_index']}"


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


def test_select_zindi_challenge_falls_back_when_challenge_id_silent_miss(monkeypatch):
    monkeypatch.setenv("ZINDI_CHALLENGE_INDEX", "3")
    user = _DummyUser(silent_challenge_miss=True)

    _select_zindi_challenge(user=user, competition_id="financial-well-being-sme")

    assert user.calls == [
        {"challenge_id": "financial-well-being-sme"},
        {"fixed_index": 3},
    ]


def test_select_zindi_challenge_raises_when_no_fallback(monkeypatch):
    monkeypatch.delenv("ZINDI_CHALLENGE_INDEX", raising=False)
    user = _DummyUser(fail_challenge=True)

    with pytest.raises(RuntimeError, match="Could not resolve Zindi challenge"):
        _select_zindi_challenge(user=user, competition_id="financial-well-being-sme")


def test_select_zindi_challenge_raises_on_bad_index(monkeypatch):
    monkeypatch.setenv("ZINDI_CHALLENGE_INDEX", "bad-index")
    user = _DummyUser(fail_challenge=True)

    with pytest.raises(RuntimeError, match="ZINDI_CHALLENGE_INDEX must be an integer"):
        _select_zindi_challenge(user=user, competition_id="financial-well-being-sme")
