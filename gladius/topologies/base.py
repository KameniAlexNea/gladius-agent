"""
Base classes for topology-driven agent coordination.

A topology encapsulates how a team of role-based agents coordinates for a
single competition iteration. The orchestrator instantiates the right topology
from TOPOLOGY_REGISTRY based on the competition config and calls
`run_iteration()` each iteration instead of managing phases explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gladius.state import CompetitionState


@dataclass
class IterationResult:
    """
    The single return value of BaseTopology.run_iteration().

    The orchestrator interprets this object to update CompetitionState,
    decide whether to submit, and determine if the run should stop.
    """

    # Overall outcome
    status: str  # "success" | "error" | "timeout" | "oom"

    # Scores (same semantics as old implementer + validator output)
    oof_score: float | None = None       # None for open-ended tasks
    quality_score: float | None = None  # 0-100 self-assessed; required

    # Artifacts
    submission_file: str = ""
    solution_files: list[str] = field(default_factory=list)

    # Narrative
    notes: str = ""
    error_message: str = ""

    # Validator outputs
    is_improvement: bool = False
    submit: bool = False
    format_ok: bool = True
    stop: bool = False
    next_directions: list[str] = field(default_factory=list)

    # Memory keeper output (written to MEMORY.md by orchestrator)
    memory_content: str | None = None
    memory_summary: str | None = None

    # Plan text (saved to .claude/plans/ by orchestrator)
    plan_text: str = ""
    approach_summary: str = ""

    # Updated session IDs for all roles used in this iteration
    # { role_name: session_id } — used by the orchestrator to resume sessions
    team_session_ids: dict[str, str] = field(default_factory=dict)


class BaseTopology(ABC):
    """
    Abstract base for all management topologies.

    Subclasses implement `run_iteration()`.  They receive the full
    CompetitionState (read-only) and project_dir, coordinate their
    role-based sub-agents, and return an IterationResult.

    Topology classes MUST NOT mutate state directly.  The orchestrator
    is the sole writer of CompetitionState.
    """

    @abstractmethod
    async def run_iteration(
        self,
        state: "CompetitionState",
        project_dir: str,
        platform: str,
        *,
        n_parallel: int = 1,
        consume_agent_call: object | None = None,
        check_budget: object | None = None,
    ) -> IterationResult:
        """
        Run one full iteration and return an IterationResult.

        Parameters
        ----------
        state:              Current (read-only) competition state.
        project_dir:        Absolute path to the competition directory.
        platform:           "kaggle" | "zindi" | "fake" | "none"
        n_parallel:         Number of parallel branches (for autonomous topology).
        consume_agent_call: Callable(label: str) -> bool — budget guardrail.
        check_budget:       Callable() -> bool — time/call budget check.
        """

    # ── Shared helpers for all topologies ────────────────────────────────────

    def _budget_ok(
        self,
        label: str,
        consume_agent_call=None,
        check_budget=None,
    ) -> bool:
        """Return False if either guardrail fires."""
        if check_budget is not None and not check_budget():
            return False
        if consume_agent_call is not None and not consume_agent_call(label):
            return False
        return True
