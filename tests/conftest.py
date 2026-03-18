"""Shared fixtures for all test modules."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gladius.state import CompetitionState


@pytest.fixture()
def ml_state() -> CompetitionState:
    """Minimal ML-competition state (metric + direction set)."""
    return CompetitionState(
        competition_id="test-comp",
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        target_metric="f1_score",
        metric_direction="maximize",
        topology="functional",
    )


@pytest.fixture()
def open_state() -> CompetitionState:
    """Open-ended task state (no metric)."""
    return CompetitionState(
        competition_id="open-task",
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        target_metric=None,
        metric_direction=None,
        topology="two-pizza",
    )


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Minimal project directory with a data/ sub-folder."""
    (tmp_path / "data").mkdir()
    return tmp_path


@pytest.fixture()
def minimal_config(project_dir: Path) -> Path:
    """Write a valid minimal YAML config and return its path."""
    cfg = project_dir / "project.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            competition_id: test-comp
            project_dir: {project_dir}
            platform: none
            metric: f1_score
            direction: maximize
        """
        ),
        encoding="utf-8",
    )
    return cfg
