"""Kinematic solver interface and topology-based dispatch.

Kinematic solvers advertise the topologies they can handle via
:meth:`KinematicSolver.can_solve`. :func:`solver_for` then picks one by asking
the graph, never by reading a ``"four_bar"`` string off the linkage. Registering
a five-bar or six-bar solver later is therefore a pure addition.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cmtool.core.graph import Linkage
from cmtool.core.registry import Registry, RegistryError
from cmtool.core.units import FloatArray
from cmtool.solvers.base import MechanismState


@runtime_checkable
class KinematicSolver(Protocol):
    """Position-level solver for one mechanism topology."""

    name: str

    def can_solve(self, linkage: Linkage) -> bool:
        """Return whether this solver handles the linkage's topology."""
        ...

    def solve(
        self, linkage: Linkage, input_angles_rad: FloatArray, **kwargs: Any
    ) -> list[MechanismState]:
        """Return one state per input angle."""
        ...


#: Registry of kinematic solvers, keyed by topology name.
KINEMATICS: Registry[KinematicSolver] = Registry("kinematic solver")


def solver_for(linkage: Linkage) -> KinematicSolver:
    """Return the registered kinematic solver that handles ``linkage``.

    Raises
    ------
    RegistryError
        If no registered solver claims the topology. The message reports the
        graph's mobility and loop count, which is usually enough to see why.
    """
    for _, solver in KINEMATICS:
        if solver.can_solve(linkage):
            return solver
    raise RegistryError(
        f"no kinematic solver handles linkage {linkage.name!r} "
        f"(bodies={len(linkage.bodies)}, joints={len(linkage.joints)}, "
        f"mobility={linkage.mobility()}, loops={len(linkage.independent_loops())}); "
        f"registered: {KINEMATICS.names()}"
    )
