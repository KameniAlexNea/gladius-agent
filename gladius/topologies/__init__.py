"""
Topology catalog + runtime registry.

* TOPOLOGY_CATALOG (TopologyDefinition dicts) — parsed from *.md, no SDK deps.
  Re-exported here for convenience; also importable directly from _catalog.py
  when you want to avoid pulling in the SDK (e.g. project setup / CLAUDE.md
  rendering paths).

* TOPOLOGY_REGISTRY — maps topology name → runtime BaseTopology subclass.
  Importing this module brings in all topology classes and their SDK deps.
"""

from __future__ import annotations

from gladius.topologies._catalog import TOPOLOGY_CATALOG, TopologyDefinition

# ── Runtime topology registry (Python implementations) ───────────────────────
# These imports bring in the SDK and all topology classes.
# Use src.topologies._catalog directly if you only need TOPOLOGY_CATALOG.

from gladius.topologies.autonomous import AutonomousTopology
from gladius.topologies.base import BaseTopology, IterationResult
from gladius.topologies.functional import FunctionalTopology
from gladius.topologies.matrix import MatrixTopology
from gladius.topologies.mini_team import MiniTeamTopology
from gladius.topologies.platform import PlatformTopology
from gladius.topologies.two_pizza import TwoPizzaTopology

TOPOLOGY_REGISTRY: dict[str, type[BaseTopology]] = {
    "functional": FunctionalTopology,
    "two-pizza": TwoPizzaTopology,
    "platform": PlatformTopology,
    "autonomous": AutonomousTopology,
    "matrix": MatrixTopology,
}

__all__ = [
    "TOPOLOGY_CATALOG",
    "TopologyDefinition",
    "TOPOLOGY_REGISTRY",
    "BaseTopology",
    "IterationResult",
    "FunctionalTopology",
    "MiniTeamTopology",
    "TwoPizzaTopology",
    "PlatformTopology",
    "AutonomousTopology",
    "MatrixTopology",
]

