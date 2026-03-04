"""
Competition config reader.

Every competition directory must have a README.md with a YAML frontmatter
block at the very top:

    ---
    competition_id: my-competition
    platform: kaggle          # kaggle | zindi | fake | none  (default: none)
    metric: auc_roc           # optional — omit for open/app tasks
    direction: maximize       # required when metric is set: maximize | minimize
    data_dir: data            # optional — relative to competition dir, or absolute
    ---

Fields:
  competition_id  (required) — slug used for platform API calls and logging
  platform        (optional, default: "none") — kaggle | zindi | fake | none
  metric          (optional) — e.g. auc_roc, rmse, logloss.
                               Omit for open-ended tasks (app building, etc.).
                               If provided, direction must also be provided.
  direction       (required when metric is set) — maximize | minimize
  data_dir        (optional, default: "data") — path to the data folder.
                                    Resolved to an absolute path; existence is validated later
                                    by runtime components that actually read the files.

The rest of the README is the human-readable task description that agents
also read for context. For open-ended tasks this is the primary source of
truth — agents derive goal, deliverables, and self-assessment criteria from it.
"""

from __future__ import annotations

from pathlib import Path


class CompetitionConfigError(ValueError):
    pass


def load_competition_config(competition_dir: str) -> dict:
    """
    Parse README.md frontmatter in competition_dir.

    Returns dict with keys:
        competition_id, platform, metric (may be None), direction (may be None),
        data_dir (absolute path)

    Raises CompetitionConfigError if README.md is missing, has no frontmatter,
    is missing competition_id, or provides metric without direction (or vice-versa).
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

    # competition_id is always required
    if not cfg.get("competition_id"):
        raise CompetitionConfigError(
            "README.md frontmatter missing required fields: competition_id"
        )

    # platform is optional — default "none" (local/artifact submission)
    platform = cfg.get("platform") or "none"
    if platform not in ("kaggle", "zindi", "fake", "none"):
        raise CompetitionConfigError(
            f"platform must be kaggle | zindi | fake | none, got {platform!r}"
        )
    cfg["platform"] = platform

    # metric + direction are both optional but must come together
    has_metric = bool(cfg.get("metric"))
    has_direction = bool(cfg.get("direction"))
    if has_metric != has_direction:
        raise CompetitionConfigError(
            "README.md frontmatter: 'metric' and 'direction' must both be provided "
            "or both omitted. Got metric="
            f"{cfg.get('metric')!r}, direction={cfg.get('direction')!r}"
        )
    if has_direction and cfg["direction"] not in ("maximize", "minimize"):
        raise CompetitionConfigError(
            f"direction must be maximize | minimize, got {cfg['direction']!r}"
        )

    # Normalise absent metric/direction to None (str coercion may have made them "None")
    cfg["metric"] = cfg.get("metric") or None
    cfg["direction"] = cfg.get("direction") or None

    # Resolve data_dir relative to competition_dir
    data_dir_explicit = "data_dir" in cfg  # True only when user set it in frontmatter
    p = Path(cfg.get("data_dir") or "data")
    if not p.is_absolute():
        p = Path(competition_dir) / p
    cfg["data_dir"] = str(p.resolve())

    # Existence of data_dir is intentionally not validated here.
    # Config parsing should remain lightweight and testable in isolation; file
    # presence checks happen in runtime setup/execution paths.

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
