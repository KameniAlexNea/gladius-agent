from gladius.config import (
    AGENT_MEMORY_DIRNAME,
    DATA_BRIEFING_FILENAME,
    EXPERIMENT_STATE_FILENAME,
    GLADIUS_DIRNAME,
    GLADIUS_RELATIVE_PATH,
    RUNTIME_DATA_BRIEFING_RELATIVE_PATH,
    RUNTIME_DIRNAME,
    RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH,
    RUNTIME_RELATIVE_PATH,
    STATE_DB_FILENAME,
    STATE_DB_RELATIVE_PATH,
    TEAM_LEAD_MEMORY_FILENAME,
    TEAM_LEAD_MEMORY_RELATIVE_PATH,
    TEAM_LEAD_ROLE_NAME,
    gladius_workspace_path,
    load_project_env,
    runtime_data_briefing_path,
    runtime_experiment_state_path,
    runtime_workspace_path,
    state_db_path,
    team_lead_memory_path,
)

__all__ = [
    "AGENT_MEMORY_DIRNAME",
    "DATA_BRIEFING_FILENAME",
    "EXPERIMENT_STATE_FILENAME",
    "GLADIUS_DIRNAME",
    "GLADIUS_RELATIVE_PATH",
    "RUNTIME_DATA_BRIEFING_RELATIVE_PATH",
    "RUNTIME_DIRNAME",
    "RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH",
    "RUNTIME_RELATIVE_PATH",
    "STATE_DB_FILENAME",
    "STATE_DB_RELATIVE_PATH",
    "TEAM_LEAD_MEMORY_FILENAME",
    "TEAM_LEAD_MEMORY_RELATIVE_PATH",
    "TEAM_LEAD_ROLE_NAME",
    "gladius_workspace_path",
    "runtime_data_briefing_path",
    "runtime_experiment_state_path",
    "runtime_workspace_path",
    "load_project_env",
    "state_db_path",
    "team_lead_memory_path",
]

if __name__ == "__main__":
    from gladius.cli import main

    main()
