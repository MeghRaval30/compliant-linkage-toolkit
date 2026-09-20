"""Solvers producing a common SimulationResult.

Importing this package registers the bundled solvers.
"""

from cmtool.solvers import beam_fea as _beam_fea  # noqa: F401  (registers "beam_fea")
from cmtool.solvers import prbm as _prbm  # noqa: F401  (registers "prbm")
from cmtool.solvers import rigid as _rigid  # noqa: F401  (registers "rigid")
from cmtool.solvers.base import SOLVERS, MechanismState, SimulationResult, Solver

__all__ = ["SOLVERS", "MechanismState", "SimulationResult", "Solver"]
