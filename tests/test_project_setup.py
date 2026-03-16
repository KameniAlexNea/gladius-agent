"""Tests for gladius.project_setup.load_config.

Covers:
- Valid configs are loaded and defaults are applied correctly
- Missing required fields raise ConfigError
- Invalid enum values for platform / topology / direction raise ConfigError
- metric and direction must both be present or both absent
- data_dir is resolved relative to project_dir when not absolute
- All max_turns defaults are populated
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gladius.project_setup import ConfigError, load_config


def write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "project.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestRequiredFields:
    def test_missing_competition_id_raises(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            project_dir: {tmp_path}
            platform: none
            """,
        )
        with pytest.raises(ConfigError, match="competition_id"):
            load_config(cfg)

    def test_missing_project_dir_raises(self, tmp_path):
        cfg = write_config(tmp_path, "competition_id: x\n")
        with pytest.raises(ConfigError, match="project_dir"):
            load_config(cfg)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "does_not_exist.yaml")


class TestPlatformValidation:
    def test_valid_platforms_accepted(self, tmp_path):
        for platform in ("kaggle", "zindi", "fake", "none"):
            cfg = write_config(
                tmp_path,
                f"""\
                competition_id: x
                project_dir: {tmp_path}
                platform: {platform}
                """,
            )
            result = load_config(cfg)
            assert result["platform"] == platform

    def test_invalid_platform_raises(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: unknown_platform
            """,
        )
        with pytest.raises(ConfigError, match="platform"):
            load_config(cfg)


class TestMetricDirection:
    def test_metric_without_direction_raises(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            metric: f1_score
            """,
        )
        with pytest.raises(ConfigError, match="metric.*direction|direction.*metric"):
            load_config(cfg)

    def test_direction_without_metric_raises(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            direction: maximize
            """,
        )
        with pytest.raises(ConfigError, match="metric.*direction|direction.*metric"):
            load_config(cfg)

    def test_invalid_direction_raises(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            metric: rmse
            direction: sideways
            """,
        )
        with pytest.raises(ConfigError, match="direction"):
            load_config(cfg)

    def test_no_metric_no_direction_is_valid(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            """,
        )
        result = load_config(cfg)
        assert result["metric"] is None
        assert result["direction"] is None

    def test_direction_normalized_to_lowercase(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            metric: rmse
            direction: MINIMIZE
            """,
        )
        result = load_config(cfg)
        assert result["direction"] == "minimize"


class TestTopologyValidation:
    def test_valid_topologies_accepted(self, tmp_path):
        for topo in ("functional", "two-pizza", "platform", "autonomous", "matrix"):
            cfg = write_config(
                tmp_path,
                f"""\
                competition_id: x
                project_dir: {tmp_path}
                platform: none
                topology: {topo}
                """,
            )
            result = load_config(cfg)
            assert result["topology"] == topo

    def test_invalid_topology_raises(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            topology: star-wars
            """,
        )
        with pytest.raises(ConfigError, match="topology"):
            load_config(cfg)

    def test_topology_defaults_to_functional(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            """,
        )
        result = load_config(cfg)
        assert result["topology"] == "functional"


class TestDataDirResolution:
    def test_data_dir_defaults_to_project_data(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            """,
        )
        result = load_config(cfg)
        assert result["data_dir"] == str(tmp_path / "data")

    def test_relative_data_dir_resolved_against_project(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            data_dir: my_data
            """,
        )
        result = load_config(cfg)
        assert result["data_dir"] == str(tmp_path / "my_data")

    def test_absolute_data_dir_kept(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            data_dir: /abs/path/data
            """,
        )
        result = load_config(cfg)
        assert result["data_dir"] == "/abs/path/data"


class TestDefaults:
    def test_max_turns_all_populated(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            """,
        )
        result = load_config(cfg)
        mt = result["max_turns"]
        for key in ("coordinator", "full_stack", "platform_layer", "product_layer",
                    "validator", "memory_keeper", "reviewer", "domain_fix"):
            assert key in mt, f"max_turns['{key}'] not set"
            assert mt[key] > 0

    def test_permissions_deny_includes_rm(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            """,
        )
        result = load_config(cfg)
        deny = result["settings"]["permissions_deny"]
        assert any("rm" in rule for rule in deny)

    def test_permissions_deny_includes_pip(self, tmp_path):
        cfg = write_config(
            tmp_path,
            f"""\
            competition_id: x
            project_dir: {tmp_path}
            platform: none
            """,
        )
        result = load_config(cfg)
        deny = result["settings"]["permissions_deny"]
        assert any("pip" in rule for rule in deny)
