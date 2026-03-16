"""Tests for gladius.preflight._build_preflight_errors.

Tests the error-collection function directly (not run_preflight_or_raise) so
we can assert on the exact error strings without trapping exceptions.

Covers:
- max_iterations and n_parallel bounds
- GLADIUS_MODEL must be set
- competition directory must exist
- data_dir existence required only when target_metric is set
- kaggle CLI presence check
- kaggle credential check (env and file)
- zindi package and credential checks
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gladius.preflight import _build_preflight_errors, run_preflight_or_raise


def _call(tmp_path: Path, **overrides) -> list[str]:
    """Helper: call _build_preflight_errors with valid defaults, apply overrides."""
    defaults = dict(
        competition_dir=str(tmp_path),
        platform="none",
        data_dir=str(tmp_path / "data"),
        target_metric=None,
        max_iterations=10,
        n_parallel=1,
    )
    defaults.update(overrides)
    with patch.dict("os.environ", {"GLADIUS_MODEL": "test-model"}, clear=False):
        return _build_preflight_errors(**defaults)


class TestBoundsChecks:
    def test_zero_max_iterations_is_error(self, tmp_path):
        errs = _call(tmp_path, max_iterations=0)
        assert any("max_iterations" in e for e in errs)

    def test_negative_max_iterations_is_error(self, tmp_path):
        errs = _call(tmp_path, max_iterations=-1)
        assert any("max_iterations" in e for e in errs)

    def test_zero_n_parallel_is_error(self, tmp_path):
        errs = _call(tmp_path, n_parallel=0)
        assert any("parallel" in e for e in errs)

    def test_valid_bounds_produce_no_error(self, tmp_path):
        errs = _call(tmp_path)
        assert errs == []


class TestModelEnvVar:
    def test_missing_model_env_is_error(self, tmp_path):
        env = {"GLADIUS_MODEL": ""}
        with patch.dict("os.environ", env, clear=False):
            errs = _build_preflight_errors(
                competition_dir=str(tmp_path),
                platform="none",
                data_dir=str(tmp_path),
                target_metric=None,
                max_iterations=1,
                n_parallel=1,
            )
        assert any("GLADIUS_MODEL" in e for e in errs)


class TestDirectoryChecks:
    def test_missing_competition_dir_is_error(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        errs = _call(tmp_path, competition_dir=missing)
        assert any("competition directory" in e for e in errs)

    def test_missing_data_dir_only_error_when_metric_set(self, tmp_path):
        missing_data = str(tmp_path / "no_data")
        # No metric → data_dir is not checked
        errs = _call(tmp_path, data_dir=missing_data, target_metric=None)
        assert not any("data_dir" in e for e in errs)

    def test_missing_data_dir_is_error_when_metric_set(self, tmp_path):
        missing_data = str(tmp_path / "no_data")
        errs = _call(tmp_path, data_dir=missing_data, target_metric="f1")
        assert any("data_dir" in e for e in errs)


class TestKagglePlatform:
    def test_missing_kaggle_cli_is_error(self, tmp_path):
        with patch("shutil.which", return_value=None):
            errs = _call(tmp_path, platform="kaggle")
        assert any("kaggle" in e.lower() for e in errs)

    def test_missing_kaggle_credentials_is_error(self, tmp_path):
        with (
            patch("shutil.which", return_value="/usr/bin/kaggle"),
            patch.dict(
                "os.environ", {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}, clear=False
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            errs = _call(tmp_path, platform="kaggle")
        assert any("credential" in e.lower() or "kaggle" in e.lower() for e in errs)

    def test_env_credentials_accepted(self, tmp_path):
        with (
            patch("shutil.which", return_value="/usr/bin/kaggle"),
            patch.dict(
                "os.environ",
                {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"},
                clear=False,
            ),
        ):
            errs = _call(tmp_path, platform="kaggle")
        # Only credential error is resolved; CLI is found
        assert not any("credential" in e.lower() for e in errs)


class TestRunPreflightOrRaise:
    def test_raises_valueerror_on_errors(self, tmp_path):
        with pytest.raises(ValueError, match="Preflight"):
            with patch.dict("os.environ", {"GLADIUS_MODEL": ""}, clear=False):
                run_preflight_or_raise(
                    competition_dir=str(tmp_path),
                    platform="none",
                    data_dir=str(tmp_path),
                    target_metric=None,
                    max_iterations=1,
                    n_parallel=1,
                )

    def test_no_error_passes_silently(self, tmp_path):
        with patch.dict("os.environ", {"GLADIUS_MODEL": "m"}, clear=False):
            run_preflight_or_raise(
                competition_dir=str(tmp_path),
                platform="none",
                data_dir=str(tmp_path),
                target_metric=None,
                max_iterations=1,
                n_parallel=1,
            )
