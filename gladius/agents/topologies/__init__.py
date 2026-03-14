"""Topology registry — maps topology name to implementation class."""

from gladius.agents.topologies.autonomous import AutonomousTopology
from gladius.agents.topologies.base import BaseTopology, IterationResult
from gladius.agents.topologies.functional import FunctionalTopology
from gladius.agents.topologies.matrix import MatrixTopology
from gladius.agents.topologies.platform import PlatformTopology
from gladius.agents.topologies.two_pizza import TwoPizzaTopology

TOPOLOGY_REGISTRY: dict[str, type[BaseTopology]] = {
    "functional": FunctionalTopology,
    "two-pizza": TwoPizzaTopology,
    "platform": PlatformTopology,
    "autonomous": AutonomousTopology,
    "matrix": MatrixTopology,
}

__all__ = [
    "TOPOLOGY_REGISTRY",
    "BaseTopology",
    "IterationResult",
    "FunctionalTopology",
    "TwoPizzaTopology",
    "PlatformTopology",
    "AutonomousTopology",
    "MatrixTopology",
]
