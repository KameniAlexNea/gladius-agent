"""Startup preflight checks for the competition agent."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


def _build_preflight_errors(
    *,
    competition_dir: str,
    platform: str,
    data_dir: str,
    target_metric: str | None,
    max_iterations: int,
    n_parallel: int,
) -> list[str]:
    errors: list[str] = []
    load_dotenv(
        os.path.join(competition_dir, ".env")
    )  # re-load .env from project dir for preflight checks

    if max_iterations <= 0:
        errors.append("max_iterations must be > 0")
    if n_parallel <= 0:
        errors.append("--parallel must be >= 1")

    model_name = os.environ.get("GLADIUS_MODEL")
    if not model_name:
        errors.append("GLADIUS_MODEL is not set in environment/.env")

    cdir = Path(competition_dir)
    if not cdir.exists():
        errors.append(f"competition directory does not exist: {competition_dir}")

    if target_metric is not None:
        ddir = Path(data_dir)
        if not ddir.exists():
            errors.append(f"data_dir does not exist for metric task: {data_dir}")

    if platform == "kaggle" and shutil.which("kaggle") is None:
        errors.append(
            "kaggle CLI not found on PATH (required for kaggle platform checks/submission)"
        )
    if platform == "kaggle":
        has_env_creds = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        has_file_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
        if not (has_env_creds or has_file_creds):
            errors.append(
                "Kaggle credentials missing (set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)"
            )

    if platform == "zindi" and importlib.util.find_spec("zindi") is None:
        errors.append("zindi package is not installed (required for zindi platform)")
    if platform == "zindi":
        has_zindi_creds = bool(
            (os.getenv("ZINDI_USERNAME") or os.getenv("USER_NAME"))
            and (os.getenv("ZINDI_PASSWORD") or os.getenv("PASSWORD"))
        )
        if not has_zindi_creds:
            errors.append(
                "Zindi credentials missing (set ZINDI_USERNAME and ZINDI_PASSWORD)"
            )

    return errors


def run_preflight_or_raise(
    *,
    competition_dir: str,
    platform: str,
    data_dir: str,
    target_metric: str | None,
    max_iterations: int,
    n_parallel: int,
) -> None:
    errors = _build_preflight_errors(
        competition_dir=competition_dir,
        platform=platform,
        data_dir=data_dir,
        target_metric=target_metric,
        max_iterations=max_iterations,
        n_parallel=n_parallel,
    )
    if errors:
        msg = "Preflight checks failed:\n- " + "\n- ".join(errors)
        raise ValueError(msg)
