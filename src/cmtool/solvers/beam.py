"""Two-dimensional co-rotational beam elements and a Newton-Raphson solver.

Why co-rotational
-----------------
A flexure in this project rotates a long way while the material inside it barely
strains at all. A linear beam element cannot express that: it confuses the large
rigid rotation with deformation and stiffens catastrophically.

The co-rotational formulation separates the two. Each element carries a local
frame that rotates with it, so the element's *rigid* motion is subtracted out and
what remains -- an axial stretch and two end rotations relative to the rotated
frame -- stays small enough for ordinary linear beam theory to describe exactly.
Large displacement, small strain. That is the regime a flexure lives in.

Formulation
-----------
Each node has three degrees of freedom: ``u``, ``v``, ``theta``. For an element
of undeformed length ``L0`` and current length ``Ln`` at current angle ``beta``:

* rigid rotation ``alpha = beta - beta0``,
* local deformations ``[Ln - L0, theta1 - alpha, theta2 - alpha]``,
* local forces from the linear beam relations
  ``N = EA/L0 * du``, ``M1 = EI/L0 * (4 t1 + 2 t2)``, ``M2 = EI/L0 * (2 t1 + 4 t2)``.

The local-to-global transformation ``B`` maps deformations to global degrees of
freedom, giving internal forces ``B^T f_local`` and a consistent tangent
``B^T k_local B`` plus the geometric terms from differentiating ``B`` itself.
Including those geometric terms is what makes Newton converge quadratically
rather than crawling.

Equilibrium is solved by Newton-Raphson with load stepping: the load (or the
prescribed rotation) is applied in increments, each one starting from the last
converged state, because a good initial guess is what keeps Newton in its basin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cmtool.core.units import FloatArray, wrap_to_pi

#: Degrees of freedom per node: horizontal, vertical, rotation.
DOF_PER_NODE = 3


@dataclass(frozen=True)
class BeamSection:
    """Cross-section properties of one beam element.

    Attributes
    ----------
    area_mm2
        Cross-sectional area ``A``.
    second_moment_mm4
        In-plane second moment ``I``.
    youngs_modulus_mpa
        ``E``, in N/mm^2.
    """

    area_mm2: float
    second_moment_mm4: float
    youngs_modulus_mpa: float

    @classmethod
    def rectangular(
        cls, thickness_mm: float, width_mm: float, youngs_modulus_mpa: float
    ) -> BeamSection:
        """Build a section for a rectangle of in-plane ``t`` and out-of-plane ``w``."""
        return cls(
            area_mm2=thickness_mm * width_mm,
            second_moment_mm4=width_mm * thickness_mm**3 / 12.0,
            youngs_modulus_mpa=youngs_modulus_mpa,
        )

    @property
    def in_plane_thickness_mm(self) -> float:
        """In-plane depth ``t``, recovered as ``sqrt(12 I / A)``.

        Exact for a rectangle, since ``A = t w`` and ``I = w t^3 / 12`` give
        ``12 I / A = t^2``. Every section this solver builds is rectangular, and
        recovering ``t`` this way means the element carries its own thickness
        rather than the callers having to track it alongside.
        """
        return float(np.sqrt(12.0 * self.second_moment_mm4 / self.area_mm2))

    @property
    def ea(self) -> float:
        """Axial rigidity ``E A``."""
        return self.youngs_modulus_mpa * self.area_mm2

    @property
    def ei(self) -> float:
        """Bending rigidity ``E I``."""
        return self.youngs_modulus_mpa * self.second_moment_mm4


@dataclass
class BeamModel:
    """A planar frame of co-rotational beam elements.

    Attributes
    ----------
    nodes
        ``(n, 2)`` array of undeformed node coordinates, in mm.
    elements
        Pairs of node indices.
    sections
        One :class:`BeamSection` per element.
    """

    nodes: FloatArray
    elements: list[tuple[int, int]]
    sections: list[BeamSection]

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 2)
        if len(self.elements) != len(self.sections):
            raise ValueError("every element needs exactly one section")

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.nodes.shape[0])

    @property
    def n_dof(self) -> int:
        """Total number of degrees of freedom."""
        return self.n_nodes * DOF_PER_NODE

    def dof(self, node: int, component: int) -> int:
        """Global index of one degree of freedom (component 0=u, 1=v, 2=theta)."""
        return node * DOF_PER_NODE + component

    def deformed_nodes(self, displacement: FloatArray) -> FloatArray:
        """Node coordinates after applying a displacement vector."""
        offsets = np.asarray(displacement, dtype=float).reshape(-1, DOF_PER_NODE)[:, :2]
        return self.nodes + offsets

    def node_rotations(self, displacement: FloatArray) -> FloatArray:
        """Nodal rotations, in radians."""
        return np.asarray(displacement, dtype=float).reshape(-1, DOF_PER_NODE)[:, 2]

    # ------------------------------------------------------------- assembly

    def internal_force_and_tangent(self, displacement: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Assemble the internal force vector and tangent stiffness matrix."""
        force = np.zeros(self.n_dof)
        tangent = np.zeros((self.n_dof, self.n_dof))
        coords = self.deformed_nodes(displacement)
        rotations = self.node_rotations(displacement)

        for (first, second), section in zip(self.elements, self.sections, strict=True):
            index = [
                self.dof(first, 0),
                self.dof(first, 1),
                self.dof(first, 2),
                self.dof(second, 0),
                self.dof(second, 1),
                self.dof(second, 2),
            ]
            f_e, k_e = _element(
                self.nodes[first],
                self.nodes[second],
                coords[first],
                coords[second],
                float(rotations[first]),
                float(rotations[second]),
                section,
            )
            force[np.ix_(index)] += f_e
            tangent[np.ix_(index, index)] += k_e

        return force, tangent

    def element_curvatures(self, displacement: FloatArray) -> FloatArray:
        """Mean curvature of each element, in 1/mm.

        Taken from the element's local end rotations, so it is the curvature the
        element's own bending energy corresponds to. Peak bending strain in a
        section of thickness ``t`` is ``t/2`` times this.
        """
        coords = self.deformed_nodes(displacement)
        rotations = self.node_rotations(displacement)
        out = np.zeros(len(self.elements))

        for index, (first, second) in enumerate(self.elements):
            local, length0, _, _ = _local_deformations(
                self.nodes[first],
                self.nodes[second],
                coords[first],
                coords[second],
                float(rotations[first]),
                float(rotations[second]),
            )
            # Linear curvature field over the element; mean magnitude of its ends.
            kappa1 = (-4.0 * local[1] - 2.0 * local[2]) / length0
            kappa2 = (2.0 * local[1] + 4.0 * local[2]) / length0
            out[index] = max(abs(kappa1), abs(kappa2))
        return out


def _local_deformations(
    node0_a: FloatArray,
    node0_b: FloatArray,
    pos_a: FloatArray,
    pos_b: FloatArray,
    rot_a: float,
    rot_b: float,
) -> tuple[FloatArray, float, float, float]:
    """Return ``(local_deformations, L0, Ln, beta)`` for one element."""
    ref = np.asarray(node0_b, dtype=float) - np.asarray(node0_a, dtype=float)
    cur = np.asarray(pos_b, dtype=float) - np.asarray(pos_a, dtype=float)
    length0 = float(np.linalg.norm(ref))
    length = float(np.linalg.norm(cur))
    if length0 <= 0.0 or length <= 0.0:
        raise ValueError("beam element has zero length")

    beta0 = float(np.arctan2(ref[1], ref[0]))
    beta = float(np.arctan2(cur[1], cur[0]))
    alpha = float(wrap_to_pi(beta - beta0))

    local = np.array(
        [
            length - length0,
            float(wrap_to_pi(rot_a - alpha)),
            float(wrap_to_pi(rot_b - alpha)),
        ]
    )
    return local, length0, length, beta


def _element(
    node0_a: FloatArray,
    node0_b: FloatArray,
    pos_a: FloatArray,
    pos_b: FloatArray,
    rot_a: float,
    rot_b: float,
    section: BeamSection,
) -> tuple[FloatArray, FloatArray]:
    """Return the internal force and tangent stiffness of one co-rotational beam element."""
    local, length0, length, beta = _local_deformations(node0_a, node0_b, pos_a, pos_b, rot_a, rot_b)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    k_local = np.array(
        [
            [section.ea / length0, 0.0, 0.0],
            [0.0, 4.0 * section.ei / length0, 2.0 * section.ei / length0],
            [0.0, 2.0 * section.ei / length0, 4.0 * section.ei / length0],
        ]
    )
    f_local = k_local @ local

    # Derivatives of the local deformations with respect to the global DOFs.
    r = np.array([-cos_b, -sin_b, 0.0, cos_b, sin_b, 0.0])
    o = np.array([sin_b, -cos_b, 0.0, -sin_b, cos_b, 0.0])
    z = o / length
    e1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    b_matrix = np.vstack([r, e1 - z, e2 - z])

    force = b_matrix.T @ f_local

    axial, moment_a, moment_b = f_local
    # Geometric stiffness: d(B)/d(d) contracted with the local forces. Without
    # these terms Newton still converges, but linearly instead of quadratically.
    tangent = b_matrix.T @ k_local @ b_matrix
    tangent += axial * np.outer(o, o) / length
    tangent += (moment_a + moment_b) * (np.outer(r, o) + np.outer(o, r)) / length**2
    return force, tangent


@dataclass
class SolveOutcome:
    """Result of a Newton-Raphson solve."""

    displacement: FloatArray
    reaction: FloatArray
    converged: bool
    iterations: list[int] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    steps: int = 0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "converged": self.converged,
            "steps": self.steps,
            "total_iterations": int(sum(self.iterations)),
            "max_iterations_in_a_step": max(self.iterations) if self.iterations else 0,
            "final_residual": self.residuals[-1] if self.residuals else None,
        }


class ConvergenceError(RuntimeError):
    """Raised when Newton-Raphson fails to converge within its iteration budget."""


def solve(
    model: BeamModel,
    *,
    fixed_dofs: dict[int, float] | None = None,
    prescribed_dofs: dict[int, float] | None = None,
    loads: dict[int, float] | None = None,
    steps: int = 10,
    max_iterations: int = 40,
    tolerance: float = 1e-9,
    initial: FloatArray | None = None,
) -> SolveOutcome:
    """Solve for equilibrium by Newton-Raphson with load stepping.

    Parameters
    ----------
    fixed_dofs
        Degrees of freedom held at a fixed value for the whole solve, as
        ``{dof_index: value}``. Usually zeros, for a clamped end.
    prescribed_dofs
        Degrees of freedom ramped from their current value to a target across the
        load steps. This is how a **prescribed input rotation** is applied.
    loads
        External forces and moments, ramped over the same steps.
    steps
        Number of load increments. More steps cost time but widen the basin of
        convergence; a flexure bent through a large angle needs several.
    tolerance
        Convergence test on the free-DOF residual norm, relative to the applied
        load scale.

    Returns
    -------
    SolveOutcome
        Includes the reaction vector, which is where the **input torque** comes
        from when the input rotation is prescribed.

    Raises
    ------
    ConvergenceError
        If any load step fails to converge.
    """
    fixed_dofs = dict(fixed_dofs or {})
    prescribed_dofs = dict(prescribed_dofs or {})
    loads = dict(loads or {})
    if set(fixed_dofs) & set(prescribed_dofs):
        raise ValueError("a degree of freedom cannot be both fixed and prescribed")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    n_dof = model.n_dof
    displacement = np.zeros(n_dof) if initial is None else np.array(initial, dtype=float).copy()
    for index, value in fixed_dofs.items():
        displacement[index] = value

    constrained = sorted(set(fixed_dofs) | set(prescribed_dofs))
    free = np.array([i for i in range(n_dof) if i not in set(constrained)], dtype=int)

    start = {index: float(displacement[index]) for index in prescribed_dofs}
    external_full = np.zeros(n_dof)
    for index, value in loads.items():
        external_full[index] = value

    outcome = SolveOutcome(
        displacement=displacement, reaction=np.zeros(n_dof), converged=False, steps=steps
    )
    scale = max(float(np.linalg.norm(external_full)), 1.0)

    for step in range(1, steps + 1):
        fraction = step / steps
        for index, target in prescribed_dofs.items():
            displacement[index] = start[index] + fraction * (target - start[index])
        external = external_full * fraction

        converged = False
        for iteration in range(1, max_iterations + 1):
            internal, tangent = model.internal_force_and_tangent(displacement)
            residual = external - internal
            if free.size == 0:
                converged = True
                outcome.iterations.append(iteration)
                outcome.residuals.append(0.0)
                break

            norm = float(np.linalg.norm(residual[free])) / scale
            if norm < tolerance:
                converged = True
                outcome.iterations.append(iteration)
                outcome.residuals.append(norm)
                break

            sub = tangent[np.ix_(free, free)]
            try:
                delta = np.linalg.solve(sub, residual[free])
            except np.linalg.LinAlgError as exc:
                raise ConvergenceError(
                    f"singular tangent stiffness at load step {step}/{steps}; the model is "
                    "under-constrained or has reached a limit point"
                ) from exc
            displacement[free] += delta

        if not converged:
            raise ConvergenceError(
                f"Newton-Raphson did not converge at load step {step}/{steps} "
                f"(residual {norm:.3e}, tolerance {tolerance:.1e}); try more steps"
            )

    internal, _ = model.internal_force_and_tangent(displacement)
    outcome.displacement = displacement
    outcome.reaction = internal - external_full
    outcome.converged = True
    return outcome
