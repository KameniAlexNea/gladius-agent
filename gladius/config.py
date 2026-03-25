from __future__ import annotations

from pathlib import Path

GLADIUS_DIRNAME = ".gladius"
RUNTIME_DIRNAME = "runtime"
AGENT_MEMORY_DIRNAME = "agent-memory"
TEAM_LEAD_ROLE_NAME = "team-lead"

EXPERIMENT_STATE_FILENAME = "EXPERIMENT_STATE.json"
DATA_BRIEFING_FILENAME = "DATA_BRIEFING.md"
TEAM_LEAD_MEMORY_FILENAME = "MEMORY.md"
STATE_DB_FILENAME = "state.db"

GLADIUS_RELATIVE_PATH = GLADIUS_DIRNAME
RUNTIME_RELATIVE_PATH = f"{GLADIUS_DIRNAME}/{RUNTIME_DIRNAME}"
RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH = (
    f"{RUNTIME_RELATIVE_PATH}/{EXPERIMENT_STATE_FILENAME}"
)
RUNTIME_DATA_BRIEFING_RELATIVE_PATH = (
    f"{RUNTIME_RELATIVE_PATH}/{DATA_BRIEFING_FILENAME}"
)
TEAM_LEAD_MEMORY_RELATIVE_PATH = f"{RUNTIME_RELATIVE_PATH}/{AGENT_MEMORY_DIRNAME}/{TEAM_LEAD_ROLE_NAME}/{TEAM_LEAD_MEMORY_FILENAME}"
STATE_DB_RELATIVE_PATH = f"{GLADIUS_DIRNAME}/{STATE_DB_FILENAME}"


def gladius_workspace_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / GLADIUS_DIRNAME


def runtime_workspace_path(project_dir: str | Path) -> Path:
    return gladius_workspace_path(project_dir) / RUNTIME_DIRNAME


def runtime_experiment_state_path(project_dir: str | Path) -> Path:
    return runtime_workspace_path(project_dir) / EXPERIMENT_STATE_FILENAME


def runtime_data_briefing_path(project_dir: str | Path) -> Path:
    return runtime_workspace_path(project_dir) / DATA_BRIEFING_FILENAME


def team_lead_memory_path(project_dir: str | Path) -> Path:
    return (
        runtime_workspace_path(project_dir)
        / AGENT_MEMORY_DIRNAME
        / TEAM_LEAD_ROLE_NAME
        / TEAM_LEAD_MEMORY_FILENAME
    )


def state_db_path(project_dir: str | Path) -> Path:
    return gladius_workspace_path(project_dir) / STATE_DB_FILENAME
