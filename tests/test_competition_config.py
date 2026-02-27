"""Tests for competition_config frontmatter parsing (uses pyyaml)."""

import textwrap
from pathlib import Path

import pytest

from gladius.utils.competition_config import CompetitionConfigError, load_competition_config


def write_readme(tmp_path: Path, content: str) -> Path:
    readme = tmp_path / "README.md"
    readme.write_text(textwrap.dedent(content), encoding="utf-8")
    return tmp_path


# ── Valid frontmatter ─────────────────────────────────────────────────────────

def test_minimal_valid(tmp_path):
    write_readme(tmp_path, """\
        ---
        competition_id: my-comp
        platform: fake
        metric: auc_roc
        direction: maximize
        ---
        # Competition
        Description here.
    """)
    cfg = load_competition_config(str(tmp_path))
    assert cfg["competition_id"] == "my-comp"
    assert cfg["platform"] == "fake"
    assert cfg["metric"] == "auc_roc"
    assert cfg["direction"] == "maximize"


def test_data_dir_default(tmp_path):
    write_readme(tmp_path, """\
        ---
        competition_id: comp
        platform: kaggle
        metric: rmse
        direction: minimize
        ---
    """)
    cfg = load_competition_config(str(tmp_path))
    # data_dir defaults to "data" resolved relative to competition_dir
    assert cfg["data_dir"].endswith("data")


def test_data_dir_custom(tmp_path):
    write_readme(tmp_path, """\
        ---
        competition_id: comp
        platform: zindi
        metric: logloss
        direction: minimize
        data_dir: datasets
        ---
    """)
    cfg = load_competition_config(str(tmp_path))
    assert cfg["data_dir"].endswith("datasets")


def test_colon_in_value(tmp_path):
    """YAML values containing ':' should be parsed correctly with pyyaml."""
    write_readme(tmp_path, """\
        ---
        competition_id: "my-comp:2025"
        platform: fake
        metric: auc_roc
        direction: maximize
        ---
    """)
    cfg = load_competition_config(str(tmp_path))
    assert cfg["competition_id"] == "my-comp:2025"


def test_inline_comment_ignored(tmp_path):
    """pyyaml handles inline # comments natively."""
    write_readme(tmp_path, """\
        ---
        competition_id: my-comp  # this is the slug
        platform: fake
        metric: auc_roc
        direction: maximize
        ---
    """)
    cfg = load_competition_config(str(tmp_path))
    assert cfg["competition_id"] == "my-comp"


# ── Missing README ────────────────────────────────────────────────────────────

def test_missing_readme(tmp_path):
    with pytest.raises(CompetitionConfigError, match="No README.md"):
        load_competition_config(str(tmp_path))


# ── Missing required fields ───────────────────────────────────────────────────

def test_missing_competition_id(tmp_path):
    write_readme(tmp_path, """\
        ---
        platform: fake
        metric: auc_roc
        direction: maximize
        ---
    """)
    with pytest.raises(CompetitionConfigError, match="missing required fields"):
        load_competition_config(str(tmp_path))


def test_invalid_platform(tmp_path):
    write_readme(tmp_path, """\
        ---
        competition_id: comp
        platform: codalab
        metric: auc_roc
        direction: maximize
        ---
    """)
    with pytest.raises(CompetitionConfigError, match="platform must be"):
        load_competition_config(str(tmp_path))


def test_invalid_direction(tmp_path):
    write_readme(tmp_path, """\
        ---
        competition_id: comp
        platform: kaggle
        metric: auc_roc
        direction: up
        ---
    """)
    with pytest.raises(CompetitionConfigError, match="direction must be"):
        load_competition_config(str(tmp_path))


# ── Malformed frontmatter ─────────────────────────────────────────────────────

def test_no_frontmatter(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# No frontmatter here\n", encoding="utf-8")
    with pytest.raises(CompetitionConfigError, match="'---'"):
        load_competition_config(str(tmp_path))


def test_unclosed_frontmatter(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\ncompetition_id: comp\n# never closed\n", encoding="utf-8")
    with pytest.raises(CompetitionConfigError, match="never closed"):
        load_competition_config(str(tmp_path))
