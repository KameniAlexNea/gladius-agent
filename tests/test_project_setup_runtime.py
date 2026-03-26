from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import gladius.project_setup as ps


def _write_readme(dir_path: Path, frontmatter: str) -> Path:
    p = dir_path / "README.md"
    p.write_text(frontmatter + "\n# Title\n", encoding="utf-8")
    return p


def test_parse_frontmatter_no_block(tmp_path: Path):
    r = tmp_path / "README.md"
    r.write_text("# No frontmatter", encoding="utf-8")
    assert ps._parse_frontmatter(r) == {}


def test_parse_frontmatter_invalid_yaml_raises(tmp_path: Path):
    r = tmp_path / "README.md"
    r.write_text("---\n: bad\n---\n", encoding="utf-8")
    with pytest.raises(ps.CompetitionConfigError):
        ps._parse_frontmatter(r)


def test_load_competition_config_from_readme_success(tmp_path: Path):
    _write_readme(
        tmp_path,
        textwrap.dedent(
            """\
            ---
            competition_id: c
            platform: none
            topology: functional
            metric: auc
            direction: maximize
            submission_threshold: 0.9
            data_dir: data
            ---
            """
        ),
    )
    cfg = ps.load_competition_config(str(tmp_path))
    assert cfg["competition_id"] == "c"
    assert cfg["submission_threshold"] == 0.9
    assert cfg["data_dir"].endswith("/data")


def test_load_competition_config_missing_readme_raises(tmp_path: Path):
    with pytest.raises(ps.CompetitionConfigError):
        ps.load_competition_config(str(tmp_path))


def test_load_competition_config_uses_load_config_when_config_path(monkeypatch, tmp_path: Path):
    _write_readme(tmp_path, "---\ncompetition_id: c\n---")
    cfg_file = tmp_path / "project.yaml"
    cfg_file.write_text("x: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        ps,
        "load_config",
        lambda p: {
            "competition_id": "from-file",
            "platform": "none",
            "metric": None,
            "direction": None,
            "data_dir": str(tmp_path / "data"),
            "topology": "functional",
            "submission_threshold": None,
        },
    )

    cfg = ps.load_competition_config(str(tmp_path), config_path=str(cfg_file))
    assert cfg["competition_id"] == "from-file"


def test_setup_calls_subsystems(monkeypatch, tmp_path: Path):
    cfg_file = tmp_path / "project.yaml"
    cfg_file.write_text("x: 1\n", encoding="utf-8")
    cfg = {
        "competition_id": "c",
        "project_dir": str(tmp_path / "proj"),
        "topology": "functional",
        "platform": "none",
        "metric": "auc",
        "direction": "maximize",
        "roles": "all",
        "model": "m",
        "small_model": "s",
        "gladius_skills": "all",
        "scientific_skills": False,
        "scientific_skills_path": "",
        "custom_skills_dir": "",
        "force": False,
        "mcp": {"platform_server": False, "extra": {}},
        "settings": {
            "permissions_allow": ["Bash(ls *)"],
            "permissions_deny": ["Bash(rm *)"],
            "additional_directories": [],
        },
        "default_mode": "acceptEdits",
        "data_dir": str(tmp_path / "proj" / "data"),
        "use_web_search": False,
    }
    monkeypatch.setattr(ps, "load_config", lambda p: cfg)

    called = {"roles": 0, "skills": 0, "tools": 0, "claude": 0}
    monkeypatch.setattr(ps.roles, "copy", lambda *a, **k: called.__setitem__("roles", called["roles"] + 1))
    monkeypatch.setattr(ps.skills, "copy", lambda *a, **k: called.__setitem__("skills", called["skills"] + 1))
    monkeypatch.setattr(ps.skills, "copy_scientific", lambda *a, **k: None)
    monkeypatch.setattr(ps.skills, "copy_custom", lambda *a, **k: None)
    monkeypatch.setattr(ps.tools, "write_mcp_json", lambda *a, **k: called.__setitem__("tools", called["tools"] + 1))
    monkeypatch.setattr(ps.claude_md, "write_from_project", lambda *a, **k: called.__setitem__("claude", called["claude"] + 1))

    root = ps.setup(cfg_file)
    assert root.exists()
    assert called["roles"] == 1
    assert called["skills"] == 1
    assert called["tools"] == 1
    assert called["claude"] == 1
