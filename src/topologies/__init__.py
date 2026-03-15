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

from src.topologies._catalog import TOPOLOGY_CATALOG, TopologyDefinition

# ── Runtime topology registry (Python implementations) ───────────────────────
# These imports bring in the SDK and all topology classes.
# Use src.topologies._catalog directly if you only need TOPOLOGY_CATALOG.

from src.topologies.autonomous import AutonomousTopology
from src.topologies.base import BaseTopology, IterationResult
from src.topologies.functional import FunctionalTopology
from src.topologies.matrix import MatrixTopology
from src.topologies.platform import PlatformTopology
from src.topologies.two_pizza import TwoPizzaTopology

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
    "TwoPizzaTopology",
    "PlatformTopology",
    "AutonomousTopology",
    "MatrixTopology",
]

