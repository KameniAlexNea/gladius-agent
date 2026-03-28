from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_dir: str | Path, *, override: bool = True) -> Path | None:
    """Load .env from the given project directory.

    Returns the resolved .env path when loaded, otherwise None.
    """
    env_path = Path(project_dir).expanduser().resolve() / ".env"
    if not env_path.is_file():
        return None
    load_dotenv(dotenv_path=env_path, override=override)
    return env_path


@dataclass(frozen=True)
class WorkspaceLayout:
    """Defines workspace file/folder names and derived path helpers."""

    gladius_dirname: str = ".gladius"
    runtime_dirname: str = "runtime"
    agent_memory_dirname: str = "agent-memory"
    team_lead_role_name: str = "team-lead"
    experiment_state_filename: str = "EXPERIMENT_STATE.json"
    data_briefing_filename: str = "DATA_BRIEFING.md"
    team_lead_memory_filename: str = "MEMORY.md"
    state_db_filename: str = "state.db"

    @property
    def gladius_relative_path(self) -> str:
        return self.gladius_dirname

    @property
    def runtime_relative_path(self) -> str:
        return f"{self.gladius_dirname}/{self.runtime_dirname}"

    @property
    def runtime_experiment_state_relative_path(self) -> str:
        return f"{self.runtime_relative_path}/{self.experiment_state_filename}"

    @property
    def runtime_data_briefing_relative_path(self) -> str:
        return f"{self.runtime_relative_path}/{self.data_briefing_filename}"

    @property
    def team_lead_memory_relative_path(self) -> str:
        return (
            f"{self.runtime_relative_path}/{self.agent_memory_dirname}/"
            f"{self.team_lead_role_name}/{self.team_lead_memory_filename}"
        )

    @property
    def state_db_relative_path(self) -> str:
        return f"{self.gladius_dirname}/{self.state_db_filename}"

    def gladius_workspace_path(self, project_dir: str | Path) -> Path:
        return Path(project_dir) / self.gladius_dirname

    def runtime_workspace_path(self, project_dir: str | Path) -> Path:
        return self.gladius_workspace_path(project_dir) / self.runtime_dirname

    def runtime_experiment_state_path(self, project_dir: str | Path) -> Path:
        return self.runtime_workspace_path(project_dir) / self.experiment_state_filename

    def runtime_data_briefing_path(self, project_dir: str | Path) -> Path:
        return self.runtime_workspace_path(project_dir) / self.data_briefing_filename

    def team_lead_memory_path(self, project_dir: str | Path) -> Path:
        return (
            self.runtime_workspace_path(project_dir)
            / self.agent_memory_dirname
            / self.team_lead_role_name
            / self.team_lead_memory_filename
        )

    def state_db_path(self, project_dir: str | Path) -> Path:
        return self.gladius_workspace_path(project_dir) / self.state_db_filename


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime and dev-mode settings resolved from environment variables."""

    max_turns: int
    max_consecutive_errors: int
    max_redispatch: int
    start_iteration_env_var: str
    scientific_skills_path: str
    fake_answers_path: str
    fake_platform_dir: str
    persistent_artifacts: set[str]
    max_state_snippet_chars: int
    stagnation_threshold_metric: float
    stagnation_threshold_quality: float

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            max_turns=int(os.getenv("GLADIUS_MAX_TURNS", "50")),
            max_consecutive_errors=int(
                os.getenv("GLADIUS_MAX_CONSECUTIVE_ERRORS", "10")
            ),
            max_redispatch=int(os.getenv("GLADIUS_MAX_REDISPATCH", "10")),
            start_iteration_env_var="GLADIUS_START_ITERATION",
            scientific_skills_path=os.getenv(
                "GLADIUS_SCIENTIFIC_SKILLS_PATH", ""
            ).strip(),
            fake_answers_path=os.getenv("FAKE_ANSWERS_PATH", "data/.answers.csv"),
            fake_platform_dir=os.getenv("FAKE_PLATFORM_DIR", ".fake_platform"),
            persistent_artifacts={"best_params.json"},
            max_state_snippet_chars=int(
                os.getenv("GLADIUS_MAX_STATE_SNIPPET_CHARS", "6000")
            ),
            stagnation_threshold_metric=float(
                os.getenv("GLADIUS_STAGNATION_THRESHOLD_METRIC", "0.001")
            ),
            stagnation_threshold_quality=float(
                os.getenv("GLADIUS_STAGNATION_THRESHOLD_QUALITY", "3.0")
            ),
        )


PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH: Path = PROMPTS_DIR / "orchestrator_system_prompt.md"

LAYOUT = WorkspaceLayout()
SETTINGS = RuntimeSettings.from_env()
