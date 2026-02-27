"""
Competition config reader.

Every competition directory must have a README.md with a YAML frontmatter
block at the very top:

    ---
    competition_id: my-competition
    platform: kaggle          # kaggle | zindi | fake
    metric: auc_roc
    direction: maximize       # maximize | minimize
    data_dir: data            # relative to competition dir, or absolute
    ---

Fields:
  competition_id  (required) — slug used for platform API calls
  platform        (required) — kaggle | zindi | fake
  metric          (required) — e.g. auc_roc, rmse, logloss
  direction       (required) — maximize | minimize
  data_dir        (optional, default: "data") — path to the data folder

The rest of the README is the human-readable competition description that
agents also read for context.
"""

from __future__ import annotations

from pathlib import Path


class CompetitionConfigError(ValueError):
    pass


def load_competition_config(competition_dir: str) -> dict:
    """
    Parse README.md frontmatter in competition_dir.

    Returns dict with keys:
        competition_id, platform, metric, direction, data_dir (absolute path)

    Raises CompetitionConfigError if README.md is missing, has no frontmatter,
    or is missing required fields.
    """
    readme = Path(competition_dir) / "README.md"
    if not readme.exists():
        raise CompetitionConfigError(
            f"No README.md in {competition_dir!r}. "
            "Add one with a YAML frontmatter block:\n\n"
            "    ---\n"
            "    competition_id: my-competition\n"
            "    platform: kaggle\n"
            "    metric: auc_roc\n"
            "    direction: maximize\n"
            "    data_dir: data\n"
            "    ---\n"
        )

    cfg = _parse_frontmatter(readme)

    missing = [
        k
        for k in ("competition_id", "platform", "metric", "direction")
        if not cfg.get(k)
    ]
    if missing:
        raise CompetitionConfigError(
            f"README.md frontmatter missing required fields: {missing}"
        )
    if cfg["platform"] not in ("kaggle", "zindi", "fake"):
        raise CompetitionConfigError(
            f"platform must be kaggle | zindi | fake, got {cfg['platform']!r}"
        )
    if cfg["direction"] not in ("maximize", "minimize"):
        raise CompetitionConfigError(
            f"direction must be maximize | minimize, got {cfg['direction']!r}"
        )

    # Resolve data_dir relative to competition_dir
    data_dir = cfg.get("data_dir") or "data"
    p = Path(data_dir)
    if not p.is_absolute():
        p = Path(competition_dir) / p
    cfg["data_dir"] = str(p.resolve())

    return cfg


def _parse_frontmatter(readme: Path) -> dict:
    """Parse YAML frontmatter from README.md using pyyaml."""
    import yaml

    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise CompetitionConfigError(
            f"{readme}: must start with '---' to open the YAML frontmatter block."
        )
    end = text.find("\n---", 3)
    if end == -1:
        raise CompetitionConfigError(
            f"{readme}: frontmatter block never closed with '---'."
        )
    frontmatter_text = text[3:end]
    try:
        cfg = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise CompetitionConfigError(
            f"{readme}: invalid YAML frontmatter: {exc}"
        ) from exc
    if not isinstance(cfg, dict):
        raise CompetitionConfigError(f"{readme}: frontmatter must be a YAML mapping.")
    return {k: str(v) for k, v in cfg.items()}
