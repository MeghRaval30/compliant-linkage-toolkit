r"""Pseudo-rigid-body quasi-static solver.

Replaces each flexure with a pin at its characteristic pivot plus a torsional
spring, then solves the mechanism over a prescribed input arc and reports the
coupler path, the input torque and the strain in every flexure.

What the springs do, and what they do not
-----------------------------------------
This is worth being precise about, because it is easy to assume otherwise.

The mechanism has one degree of freedom and the input angle is **prescribed**. So
with no external load, the configuration at each input angle is fixed by geometry
alone: **flexure stiffness does not change the path.** Doubling every ``K`` gives
the identical coupler curve and exactly twice the input torque.

That means the PRBM path differs from the rigid path only because the
characteristic pivots sit somewhere other than the original joints -- which, with
pivot-matched placement, is nowhere. The value the PRBM adds in Phase A is
therefore the **torque curve** and the **strain**, plus being the reference model
the beam FEA and the measurements are compared against.

The path *will* diverge once effects the PRBM cannot represent enter: a real
flexure stretches and shears as well as bending, so the links do not stay exactly
the length the PRBM assumes. That difference is precisely what the
``prbm_vs_fea`` metric measures, and why A4 exists.

Method
------
Every flexure is unstressed in the configuration the part was printed in. Writing
``dphi_j(theta)`` for the relative rotation at joint ``j`` measured from that
configuration, the strain energy is

.. math::

    U(\theta) = \sum_j \tfrac{1}{2} K_j \, \Delta\phi_j(\theta)^2

and, the mechanism being one degree of freedom, the input torque needed to hold
it at ``theta`` is

.. math::

    T(\theta) = \frac{\mathrm{d}U}{\mathrm{d}\theta}

The derivative is taken numerically along the swept arc. Both quantities are
reported, so the energy and the torque can be checked against each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cmtool.convert.base import CompliantMechanism
from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance
from cmtool.core.units import FloatArray, wrap_to_pi
from cmtool.flexures.base import FLEXURES
from cmtool.kinematics.base import solver_for
from cmtool.solvers.base import SOLVERS, SimulationResult


class PrbmSolver:
    """Quasi-static pseudo-rigid-body solution over a prescribed input arc."""

    name = "prbm"

    def can_solve(self, linkage: Linkage) -> bool:
        """Return whether a kinematic solver handles this topology.

        The solver additionally needs a :class:`CompliantMechanism`, supplied by
        :func:`cmtool.api.simulate` when it is given one.
        """
        try:
            solver_for(linkage)
        except Exception:
            # Any dispatch failure means this solver cannot handle the topology.
            return False
        return True

    def solve(
        self,
        linkage: Linkage,
        input_angles_rad: FloatArray,
        *,
        mechanism: CompliantMechanism | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """Solve the PRBM mechanism at each input angle.

        Parameters
        ----------
        linkage
            The effective linkage, with joints at the characteristic pivots.
        input_angles_rad
            Prescribed input orientations.
        mechanism
            The conversion result, carrying flexure sizes and spring constants.

        Raises
        ------
        ValueError
            If no ``mechanism`` is given; the PRBM has no springs without one.
        """
        if mechanism is None:
            raise ValueError(
                "the prbm solver needs the compliant mechanism (flexure sizes and "
                "stiffnesses); call simulate(converted_mechanism, solver='prbm')"
            )

        kinematics = solver_for(linkage)
        angles = np.atleast_1d(np.asarray(input_angles_rad, dtype=float))
        states = kinematics.solve(linkage, angles, **kwargs)

        # The part is unstressed in the configuration it was printed in, which is
        # not necessarily where the sweep starts.
        reference_angle = float(np.radians(mechanism.reference_input_deg))
        reference_state = kinematics.solve(linkage, np.array([reference_angle]))[0]

        stiffness = {name: s.stiffness_nmm_per_rad for name, s in mechanism.sizing.items()}
        rotations: dict[str, FloatArray] = {}
        for name, joint in linkage.joints.items():
            first, second = joint.bodies
            swept = np.array(
                [s.body_angles_rad[first] - s.body_angles_rad[second] for s in states],
                dtype=float,
            )
            at_rest = (
                reference_state.body_angles_rad[first] - reference_state.body_angles_rad[second]
            )
            rotations[name] = np.asarray(wrap_to_pi(swept - at_rest), dtype=float)

        energy = np.zeros_like(angles)
        for name, delta in rotations.items():
            energy = energy + 0.5 * stiffness.get(name, 0.0) * delta**2

        if angles.size >= 2:
            torque = np.gradient(energy, angles, edge_order=2)
        else:
            torque = np.zeros_like(angles)

        flexure = FLEXURES.get(mechanism.flexure_type)
        strain: dict[str, FloatArray] = {}
        for name, sized in mechanism.sizing.items():
            strain[name] = np.array(
                [
                    flexure.peak_strain(sized.geometry, float(angle)).peak_strain
                    for angle in rotations[name]
                ],
                dtype=float,
            )

        provenance = Provenance(
            config_hash=mechanism.provenance.config_hash,
            notes={"solver": self.name, **mechanism.provenance.notes},
        )
        provenance.merge(mechanism.provenance)

        peak_strain = {name: float(np.max(values)) for name, values in strain.items()}
        allowable = mechanism.feasibility.allowable_strain
        overall = max(peak_strain.values()) if peak_strain else 0.0

        diagnostics: dict[str, Any] = {
            "kinematics": kinematics.name,
            "prbm_models": mechanism.feasibility.prbm_models,
            "all_small_length": mechanism.feasibility.all_small_length,
            "stiffness_nmm_per_rad": stiffness,
            "total_stiffness_nmm_per_rad": float(sum(stiffness.values())),
            "strain_energy_nmm": energy,
            "peak_strain": peak_strain,
            "peak_strain_overall": overall,
            "allowable_strain": allowable,
            "strain_margin": (allowable / overall) if overall > 0.0 else float("inf"),
            "max_input_torque_nmm": float(np.max(np.abs(torque))) if torque.size else 0.0,
            "reference_input_deg": mechanism.reference_input_deg,
            "prbm_notes": mechanism.feasibility.prbm_notes(),
        }
        extra = getattr(kinematics, "diagnostics", None)
        if callable(extra) and states:
            diagnostics.update(extra(linkage, states))

        return SimulationResult(
            solver=self.name,
            linkage=linkage,
            states=states,
            provenance=provenance,
            input_torque_nmm=torque,
            flexure_strain=strain,
            diagnostics=diagnostics,
        )


SOLVERS.add("prbm", PrbmSolver())
