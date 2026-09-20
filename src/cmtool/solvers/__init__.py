"""Solvers producing a common SimulationResult.

Importing this package registers the bundled solvers.
"""

from cmtool.solvers import rigid as _rigid  # noqa: F401  (registers "rigid")
from cmtool.solvers.base import SOLVERS, MechanismState, SimulationResult, Solver

__all__ = ["SOLVERS", "MechanismState", "SimulationResult", "Solver"]
