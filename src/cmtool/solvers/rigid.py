"""Rigid-body kinematic solver.

Wraps whichever registered kinematic solver claims the linkage's topology and
packages the result in the common :class:`~cmtool.solvers.base.SimulationResult`
form. It computes no forces and no strains, so ``input_torque_nmm`` and
``flexure_strain`` stay ``None``.

This solver uses no material data at all, so its results are always
``is_physical`` -- the rigid path is pure geometry.
"""

from __future__ import annotations

from typing import Any

from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance
from cmtool.core.units import FloatArray
from cmtool.kinematics.base import solver_for
from cmtool.solvers.base import SOLVERS, SimulationResult


class RigidSolver:
    """Position-level solution of the rigid linkage over an input arc."""

    name = "rigid"

    def can_solve(self, linkage: Linkage) -> bool:
        """Return whether some registered kinematic solver handles this topology."""
        try:
            solver_for(linkage)
        except Exception:
            # Any dispatch failure means this solver cannot handle the topology.
            return False
        return True

    def solve(
        self, linkage: Linkage, input_angles_rad: FloatArray, **kwargs: Any
    ) -> SimulationResult:
        """Solve the rigid linkage at each input angle."""
        kinematics = solver_for(linkage)
        states = kinematics.solve(linkage, input_angles_rad, **kwargs)

        diagnostics: dict[str, Any] = {"kinematics": kinematics.name}
        extra = getattr(kinematics, "diagnostics", None)
        if callable(extra) and states:
            diagnostics.update(extra(linkage, states))

        return SimulationResult(
            solver=self.name,
            linkage=linkage,
            states=states,
            provenance=Provenance(notes={"solver": self.name}),
            diagnostics=diagnostics,
        )


SOLVERS.add("rigid", RigidSolver())
