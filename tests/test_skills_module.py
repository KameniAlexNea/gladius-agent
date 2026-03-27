from __future__ import annotations

from pathlib import Path

from gladius import skills


def test_copy_specific_and_warn_missing(tmp_path: Path, monkeypatch):
    src_root = tmp_path / "srcskills"
    src_root.mkdir()
    (src_root / "a").mkdir()
    (src_root / "a" / "SKILL.md").write_text("a", encoding="utf-8")
    (src_root / "b").mkdir()
    (src_root / "b" / "SKILL.md").write_text("b", encoding="utf-8")

    monkeypatch.setattr(skills, "_SRC_SKILLS", src_root)
    dst = tmp_path / "dst"
    skills.copy(dst, ["a", "missing"], force=False)
    assert (dst / "a" / "SKILL.md").exists()
    assert not (dst / "missing").exists()


def test_copy_scientific_and_custom(tmp_path: Path):
    dst = tmp_path / "dst"
    sci = tmp_path / "sci"
    custom = tmp_path / "custom"
    (sci / "s1").mkdir(parents=True)
    (sci / "s1" / "SKILL.md").write_text("x", encoding="utf-8")
    (custom / "c1").mkdir(parents=True)
    (custom / "c1" / "SKILL.md").write_text("y", encoding="utf-8")

    skills.copy_scientific(dst, override_path=str(sci), force=False)
    assert (dst / "s1" / "SKILL.md").exists()

    skills.copy_custom(dst, str(custom))
    assert (dst / "c1" / "SKILL.md").exists()


def test_copy_tree_force_overwrite(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "x.txt").write_text("1", encoding="utf-8")
    assert skills._copy_tree(src, dst, force=False) is True
    (src / "x.txt").write_text("2", encoding="utf-8")
    assert skills._copy_tree(src, dst, force=True) is True
    assert (dst / "x.txt").read_text(encoding="utf-8") == "2"
