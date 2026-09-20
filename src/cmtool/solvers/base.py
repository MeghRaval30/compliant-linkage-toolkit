"""Solver interface and the common result type.

Every solver -- rigid kinematics, PRBM, 2D beam FEA, 3D solid FEA -- consumes a
:class:`~cmtool.core.graph.Linkage` and produces a :class:`SimulationResult`.
Keeping one result type means the metrics, plotting and dataset code never
branches on which solver produced the data, and a new solver is a plug-in rather
than a special case.

Fields that only some solvers can fill (input torque, flexure strain) are
optional and default to ``None``. ``None`` means "this solver does not compute
that", never "zero".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance
from cmtool.core.registry import Registry
from cmtool.core.units import FloatArray, wrap_to_pi


@dataclass(frozen=True)
class MechanismState:
    """The mechanism at one value of the input coordinate.

    Attributes
    ----------
    input_angle_rad
        Absolute orientation of the input link, measured from the world +x axis
        along the vector from the input joint to the input link's far joint.
    joint_positions_mm
        World position of every joint.
    body_angles_rad
        Absolute orientation of each body's local frame (see
        :meth:`cmtool.core.graph.Linkage.body_frame`). Ground is always 0.
    output_points_mm
        World position of every tracked output point.
    branch
        Assembly mode the state was solved on.
    """

    input_angle_rad: float
    joint_positions_mm: dict[str, FloatArray]
    body_angles_rad: dict[str, float]
    output_points_mm: dict[str, FloatArray]
    branch: int = 1


@dataclass
class SimulationResult:
    """The output of any solver over a swept input arc.

    Attributes
    ----------
    solver
        Registered name of the solver that produced this result.
    linkage
        The linkage that was solved, kept so plots and metrics are self-contained.
    states
        One :class:`MechanismState` per sampled input angle, in sweep order.
    provenance
        Reproducibility record. ``provenance.placeholders_used`` being non-empty
        means the numbers are **not** a physical prediction.
    input_torque_nmm
        Input torque at each state, in N*mm. ``None`` for kinematic solvers.
    flexure_strain
        Peak bending strain per flexure at each state. ``None`` for solvers that
        do not model flexures.
    diagnostics
        Free-form solver notes (iteration counts, residuals, warnings).
    """

    solver: str
    linkage: Linkage
    states: list[MechanismState]
    provenance: Provenance = field(default_factory=Provenance)
    input_torque_nmm: FloatArray | None = None
    flexure_strain: dict[str, FloatArray] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views

    @property
    def n_states(self) -> int:
        """Number of solved configurations."""
        return len(self.states)

    @property
    def input_angles_rad(self) -> FloatArray:
        """Input angle at each state, in rad."""
        return np.array([s.input_angle_rad for s in self.states], dtype=float)

    @property
    def input_angles_deg(self) -> FloatArray:
        """Input angle at each state, in degrees."""
        return np.degrees(self.input_angles_rad)

    @property
    def input_sweep_deg(self) -> FloatArray:
        """Input angle relative to the first (reference) state, in degrees.

        This is the quantity that matters to a compliant mechanism: the
        as-printed configuration is unstressed, so deformation is measured from
        it, not from the world x axis.
        """
        angles = self.input_angles_rad
        return np.degrees(wrap_to_pi(angles - angles[0]))

    def path(self, output: str | None = None) -> FloatArray:
        """Return the ``(N, 2)`` world path of a tracked output point, in mm.

        Parameters
        ----------
        output
            Name of the output point. Defaults to the linkage's only output.
        """
        name = output or self._sole_output()
        return np.array([s.output_points_mm[name] for s in self.states], dtype=float)

    def joint_path(self, joint: str) -> FloatArray:
        """Return the ``(N, 2)`` world path of a joint, in mm."""
        return np.array([s.joint_positions_mm[joint] for s in self.states], dtype=float)

    def _sole_output(self) -> str:
        names = list(self.linkage.outputs)
        if len(names) != 1:
            raise ValueError(f"linkage has {len(names)} output points {names}; name one explicitly")
        return names[0]

    # ------------------------------------------------- compliant-side measures

    def joint_rotation_deg(self, joint: str) -> FloatArray:
        """Relative rotation at ``joint`` versus the reference state, in degrees.

        The relative rotation of the two bodies meeting at the joint is exactly
        the angle a flexure placed there would have to bend through, so this is
        the input to the strain check (``docs/physics.md``).
        """
        first, second = self.linkage.joints[joint].bodies
        rel = np.array(
            [s.body_angles_rad[first] - s.body_angles_rad[second] for s in self.states],
            dtype=float,
        )
        return np.degrees(wrap_to_pi(rel - rel[0]))

    def joint_excursion_deg(self) -> dict[str, float]:
        """Peak-to-peak relative rotation at every joint over the whole sweep.

        Peak-to-peak (not max-from-reference) is the headline number because a
        flexure's total bend range is what the strain limit and fatigue life
        respond to.
        """
        return {j: float(np.ptp(self.joint_rotation_deg(j))) for j in self.linkage.joints}

    def max_joint_rotation_deg(self) -> dict[str, float]:
        """Largest absolute rotation from the reference state, per joint."""
        return {j: float(np.max(np.abs(self.joint_rotation_deg(j)))) for j in self.linkage.joints}

    # ------------------------------------------------------------- reporting

    @property
    def is_physical(self) -> bool:
        """Whether no placeholder inputs were used (see :class:`Provenance`)."""
        return self.provenance.is_physical

    def summary(self) -> dict[str, Any]:
        """Return a compact dictionary describing the run."""
        return {
            "solver": self.solver,
            "linkage": self.linkage.name,
            "n_states": self.n_states,
            "input_range_deg": [
                float(self.input_angles_deg[0]),
                float(self.input_angles_deg[-1]),
            ]
            if self.states
            else None,
            "joint_excursion_deg": self.joint_excursion_deg() if self.states else {},
            "is_physical": self.is_physical,
            "placeholders_used": list(self.provenance.placeholders_used),
            "diagnostics": self.diagnostics,
        }


@runtime_checkable
class Solver(Protocol):
    """What every cmtool solver must provide."""

    name: str

    def can_solve(self, linkage: Linkage) -> bool:
        """Return whether this solver handles the given linkage's topology."""
        ...

    def solve(
        self, linkage: Linkage, input_angles_rad: FloatArray, **kwargs: Any
    ) -> SimulationResult:
        """Solve the linkage at each of the given input angles."""
        ...


#: Registry of solvers. Populated by :mod:`cmtool.solvers` on import.
SOLVERS: Registry[Solver] = Registry("solver")
