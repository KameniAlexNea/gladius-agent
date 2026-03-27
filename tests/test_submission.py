"""Tests for gladius.submission.

Covers:
- submit() routing: "none" always succeeds, "fake" calls _score_submission,
  "kaggle"/"zindi" delegate to platform CLIs (mocked)
- score_submission_artifact(): only "fake" returns a float; every other
  platform returns None
- update_best_submission_score(): maximize / minimize direction logic,
  first score always set, no regression to worse score
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gladius.state import CompetitionState
from gladius.utilities.submission import (
    score_submission_artifact,
    submit,
    update_best_submission_score,
)

# ── submit() ─────────────────────────────────────────────────────────────────


class TestSubmitNone:
    def test_none_platform_always_succeeds(self, tmp_path):
        ok, err = submit("none", "comp-x", str(tmp_path / "sub.csv"), "msg")
        assert ok is True
        assert err is None


class TestSubmitFake:
    def test_fake_platform_calls_scorer(self, tmp_path):
        sub = tmp_path / "sub.csv"
        sub.write_text("id,target\n1,0\n")
        with patch(
            "gladius.tools.fake_platform_tools._score_submission", return_value=0.75
        ) as mock_score:
            ok, err = submit("fake", "comp-x", str(sub), "msg")
        mock_score.assert_called_once_with(str(sub))
        assert ok is True
        assert err is None

    def test_fake_platform_scoring_failure_returns_error(self, tmp_path):
        sub = tmp_path / "sub.csv"
        sub.write_text("id,target\n1,0\n")
        with patch(
            "gladius.tools.fake_platform_tools._score_submission",
            side_effect=RuntimeError("bad"),
        ):
            ok, err = submit("fake", "comp-x", str(sub), "msg")
        assert ok is False
        assert err == "scoring_failed"


class TestSubmitKaggle:
    def test_kaggle_success(self, tmp_path):
        sub = tmp_path / "sub.csv"
        sub.write_text("id,target\n1,0\n")
        mock_result = MagicMock(returncode=0, stdout="submitted", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            ok, err = submit("kaggle", "comp-x", str(sub), "msg")
        assert ok is True
        assert err is None

    def test_kaggle_cli_failure_returns_error(self, tmp_path):
        sub = tmp_path / "sub.csv"
        sub.write_text("id,target\n1,0\n")
        mock_result = MagicMock(returncode=1, stdout="", stderr="auth error")
        with patch("subprocess.run", return_value=mock_result):
            ok, err = submit("kaggle", "comp-x", str(sub), "msg")
        assert ok is False


# ── score_submission_artifact() ──────────────────────────────────────────────


class TestScoreSubmissionArtifact:
    def test_fake_platform_returns_float(self, tmp_path):
        sub = tmp_path / "sub.csv"
        sub.write_text("id,target\n1,0\n")
        with patch(
            "gladius.tools.fake_platform_tools._score_submission", return_value=0.88
        ):
            score = score_submission_artifact("fake", str(sub))
        assert score == pytest.approx(0.88)

    def test_non_fake_platform_returns_none(self, tmp_path):
        sub = tmp_path / "sub.csv"
        for platform in ("kaggle", "zindi", "none"):
            assert score_submission_artifact(platform, str(sub)) is None

    def test_fake_scorer_exception_returns_none(self, tmp_path):
        sub = tmp_path / "sub.csv"
        with patch(
            "gladius.tools.fake_platform_tools._score_submission",
            side_effect=Exception("broken"),
        ):
            assert score_submission_artifact("fake", str(sub)) is None


# ── update_best_submission_score() ───────────────────────────────────────────


def _maximize_state() -> CompetitionState:
    return CompetitionState(
        competition_id="x",
        data_dir="/d",
        output_dir="/o",
        target_metric="f1",
        metric_direction="maximize",
    )


def _minimize_state() -> CompetitionState:
    return CompetitionState(
        competition_id="x",
        data_dir="/d",
        output_dir="/o",
        target_metric="rmse",
        metric_direction="minimize",
    )


class TestUpdateBestSubmissionScore:
    def test_first_score_always_set(self):
        s = _maximize_state()
        update_best_submission_score(state=s, new_score=0.80)
        assert s.best_submission_score == pytest.approx(0.80)

    def test_first_score_set_for_minimize(self):
        s = _minimize_state()
        update_best_submission_score(state=s, new_score=0.50)
        assert s.best_submission_score == pytest.approx(0.50)

    def test_maximize_improves_on_higher(self):
        s = _maximize_state()
        s.best_submission_score = 0.75
        update_best_submission_score(state=s, new_score=0.90)
        assert s.best_submission_score == pytest.approx(0.90)

    def test_maximize_does_not_update_on_lower(self):
        s = _maximize_state()
        s.best_submission_score = 0.90
        update_best_submission_score(state=s, new_score=0.75)
        assert s.best_submission_score == pytest.approx(0.90)

    def test_minimize_improves_on_lower(self):
        s = _minimize_state()
        s.best_submission_score = 0.50
        update_best_submission_score(state=s, new_score=0.30)
        assert s.best_submission_score == pytest.approx(0.30)

    def test_minimize_does_not_update_on_higher(self):
        s = _minimize_state()
        s.best_submission_score = 0.30
        update_best_submission_score(state=s, new_score=0.50)
        assert s.best_submission_score == pytest.approx(0.30)

    def test_no_metric_improves_on_higher(self):
        """Open-ended tasks: higher quality score always wins."""
        s = CompetitionState(competition_id="x", data_dir="/d", output_dir="/o")
        s.best_submission_score = 70.0
        update_best_submission_score(state=s, new_score=80.0)
        assert s.best_submission_score == pytest.approx(80.0)

    def test_no_metric_does_not_update_on_lower(self):
        s = CompetitionState(competition_id="x", data_dir="/d", output_dir="/o")
        s.best_submission_score = 80.0
        update_best_submission_score(state=s, new_score=70.0)
        assert s.best_submission_score == pytest.approx(80.0)
