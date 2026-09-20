"""Kinematic solvers, dispatched by topology.

Importing this package registers the bundled solvers.
"""

from cmtool.kinematics import fourbar as _fourbar  # noqa: F401  (registers "four_bar")
from cmtool.kinematics.base import KINEMATICS, KinematicSolver, solver_for

__all__ = ["KINEMATICS", "KinematicSolver", "solver_for"]
