"""Planar four-bar kinematics.

This is the **only** module (with its tests and the ``Linkage.four_bar``
convenience constructor) that is allowed to assume a four-bar. It claims the
topology through :meth:`FourBarSolver.can_solve` and everything downstream --
conversion, PRBM, FEA, metrics, plotting -- works through the generic graph and
:class:`~cmtool.solvers.base.SimulationResult` interfaces.

Method
------
Position analysis is done geometrically rather than through Freudenstein's
equation: the driven joint ``B`` is placed on its circle about ``A``, then ``C``
is the intersection of the circle of radius ``|BC|`` about ``B`` with the circle
of radius ``|CD|`` about ``D``. The two intersections are the two assembly
branches. This formulation extends unchanged to any dyad, which is what a
five-bar or six-bar solver will need, and it makes toggle positions explicit:
the branches merge exactly where the circles become tangent.

(The tests cross-check this against an independent Freudenstein implementation.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cmtool.core.graph import Linkage, LinkageError
from cmtool.core.units import FloatArray, cross2, wrap_to_pi
from cmtool.kinematics.base import KINEMATICS
from cmtool.solvers.base import MechanismState

#: Relative tolerance for declaring a dyad singular (circles tangent).
TOGGLE_TOL = 1e-9


class AssemblyError(RuntimeError):
    """Raised when a linkage cannot be assembled at a requested input angle."""


# --------------------------------------------------------------------- dyad


def solve_dyad(
    center_a: FloatArray,
    radius_a: float,
    center_b: FloatArray,
    radius_b: float,
    *,
    branch: int = 1,
) -> FloatArray | None:
    """Intersect two circles and return the point on the requested branch.

    Parameters
    ----------
    center_a, radius_a
        First circle (in the four-bar: joint ``B`` and the coupler length).
    center_b, radius_b
        Second circle (in the four-bar: joint ``D`` and the output length).
    branch
        ``+1`` selects the intersection lying to the **left** of the directed
        line ``center_a -> center_b``; ``-1`` selects the right-hand one. This
        sign is exactly :func:`dyad_branch`, so branches are consistent between
        assembly, sweeping and serialisation.

    Returns
    -------
    numpy.ndarray or None
        The intersection point, or ``None`` when the circles do not intersect
        (the dyad cannot be assembled at this configuration).
    """
    pos_a = np.asarray(center_a, dtype=float)
    pos_b = np.asarray(center_b, dtype=float)
    separation = pos_b - pos_a
    distance = float(np.linalg.norm(separation))

    if distance <= 0.0:
        return None
    if distance > radius_a + radius_b:
        return None
    if distance < abs(radius_a - radius_b):
        return None

    along = (radius_a**2 - radius_b**2 + distance**2) / (2.0 * distance)
    height_sq = radius_a**2 - along**2
    height = float(np.sqrt(max(height_sq, 0.0)))

    direction = separation / distance
    normal = np.array([-direction[1], direction[0]])
    return pos_a + along * direction + float(np.sign(branch) or 1.0) * height * normal


def dyad_branch(center_a: FloatArray, center_b: FloatArray, point: FloatArray) -> int:
    """Return which branch ``point`` lies on relative to ``center_a -> center_b``."""
    separation = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
    offset = np.asarray(point, dtype=float) - np.asarray(center_a, dtype=float)
    return 1 if cross2(separation, offset) >= 0.0 else -1


def dyad_clearance(
    center_a: FloatArray, radius_a: float, center_b: FloatArray, radius_b: float
) -> float:
    """Return how far the dyad is from a toggle, in mm.

    Zero means the two circles are tangent: the assembly branches merge and the
    mechanism is at a singular (toggle) position. Negative means the dyad cannot
    close at all.
    """
    distance = float(np.linalg.norm(np.asarray(center_b) - np.asarray(center_a)))
    return float(min(radius_a + radius_b - distance, distance - abs(radius_a - radius_b)))


# ------------------------------------------------------------------ topology


@dataclass(frozen=True)
class FourBarRoles:
    """Which graph element plays which four-bar role.

    Roles are discovered from the graph, so a four-bar written with any body or
    joint names is handled identically.
    """

    ground: str
    input_body: str
    coupler: str
    output_body: str
    joint_a: str  # ground - input  (the driven joint)
    joint_b: str  # input - coupler
    joint_c: str  # coupler - output
    joint_d: str  # output - ground

    @property
    def joints(self) -> tuple[str, str, str, str]:
        """The four joints in loop order A, B, C, D."""
        return (self.joint_a, self.joint_b, self.joint_c, self.joint_d)


def identify_four_bar(linkage: Linkage) -> FourBarRoles:
    """Work out the four-bar roles of a linkage's bodies and joints.

    Raises
    ------
    LinkageError
        If the graph is not a single-loop four-bar driven at a ground joint.
    """
    if len(linkage.bodies) != 4 or len(linkage.joints) != 4:
        raise LinkageError(
            f"four-bar needs 4 bodies and 4 joints, got "
            f"{len(linkage.bodies)} and {len(linkage.joints)}"
        )
    if any(len(linkage.joints_of(b)) != 2 for b in linkage.bodies):
        raise LinkageError("every body of a four-bar must carry exactly two joints")

    ground = linkage.ground
    joint_a = linkage.input_joint
    if joint_a not in linkage.joints:
        raise LinkageError(f"input joint {joint_a!r} is not a joint of the linkage")
    if ground not in linkage.joints[joint_a].bodies:
        raise LinkageError(
            f"input joint {joint_a!r} must be a ground joint for the four-bar solver"
        )

    input_body = linkage.input_body
    ground_joints = linkage.joints_of(ground)
    others = [j for j in ground_joints if j != joint_a]
    if len(others) != 1:
        raise LinkageError(f"ground must carry exactly two joints, found {ground_joints}")
    joint_d = others[0]
    output_body = linkage.joints[joint_d].other(ground)
    if output_body == input_body:
        raise LinkageError("input and output links must be different bodies")

    remaining = set(linkage.bodies) - {ground, input_body, output_body}
    if len(remaining) != 1:
        raise LinkageError(f"could not identify a unique coupler, candidates {remaining}")
    coupler = remaining.pop()

    joint_b = _joint_between(linkage, input_body, coupler)
    joint_c = _joint_between(linkage, coupler, output_body)

    return FourBarRoles(
        ground=ground,
        input_body=input_body,
        coupler=coupler,
        output_body=output_body,
        joint_a=joint_a,
        joint_b=joint_b,
        joint_c=joint_c,
        joint_d=joint_d,
    )


def _joint_between(linkage: Linkage, first: str, second: str) -> str:
    matches = [j.name for j in linkage.joints.values() if set(j.bodies) == {first, second}]
    if len(matches) != 1:
        raise LinkageError(f"expected exactly one joint between {first!r} and {second!r}")
    return matches[0]


@dataclass(frozen=True)
class FourBarLengths:
    """The four link lengths, in mm, in the standard naming."""

    ground: float
    input: float
    coupler: float
    output: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(ground, input, coupler, output)``."""
        return (self.ground, self.input, self.coupler, self.output)


def link_lengths_of(linkage: Linkage, roles: FourBarRoles | None = None) -> FourBarLengths:
    """Return the four link lengths from the reference configuration."""
    roles = roles or identify_four_bar(linkage)
    pos = {name: j.position_mm for name, j in linkage.joints.items()}

    def dist(first: str, second: str) -> float:
        return float(np.linalg.norm(pos[second] - pos[first]))

    return FourBarLengths(
        ground=dist(roles.joint_a, roles.joint_d),
        input=dist(roles.joint_a, roles.joint_b),
        coupler=dist(roles.joint_b, roles.joint_c),
        output=dist(roles.joint_c, roles.joint_d),
    )


# ------------------------------------------------------------------- Grashof


@dataclass(frozen=True)
class GrashofResult:
    """Grashof classification of a four-bar.

    Attributes
    ----------
    condition
        ``"grashof"``, ``"non_grashof"`` or ``"change_point"``.
    classification
        The mechanism type given which link is grounded.
    shortest
        Role name of the shortest link.
    input_fully_rotates
        Whether the **input** link can make complete revolutions. Relevant here
        because a flexure cannot: a fully rotating input has no direct compliant
        equivalent, so such designs must be driven over a limited arc
        (``docs/physics.md``).
    """

    condition: str
    classification: str
    shortest: str
    s_plus_l: float
    p_plus_q: float
    input_fully_rotates: bool

    @property
    def is_grashof(self) -> bool:
        """Whether ``s + l < p + q``."""
        return self.condition == "grashof"


def classify_grashof(lengths: FourBarLengths, *, tol: float = 1e-9) -> GrashofResult:
    """Classify a four-bar by the Grashof criterion.

    With links sorted by length, Grashof's condition is ``s + l < p + q``: the
    shortest link can then fully rotate relative to all others, and which link is
    grounded decides the mechanism type. Equality is a change point, where the
    mechanism can pass through a configuration with all links collinear and the
    assembly branch becomes indeterminate.
    """
    roles = {
        "ground": lengths.ground,
        "input": lengths.input,
        "coupler": lengths.coupler,
        "output": lengths.output,
    }
    ordered = sorted(roles.items(), key=lambda kv: kv[1])
    shortest_role, shortest = ordered[0]
    longest = ordered[-1][1]
    middle_sum = ordered[1][1] + ordered[2][1]
    s_plus_l = shortest + longest

    if abs(s_plus_l - middle_sum) <= tol * max(1.0, middle_sum):
        return GrashofResult(
            condition="change_point",
            classification="change_point",
            shortest=shortest_role,
            s_plus_l=s_plus_l,
            p_plus_q=middle_sum,
            # At a change point the input does traverse full revolutions, but the
            # branch is indeterminate at the collinear configuration.
            input_fully_rotates=True,
        )

    if s_plus_l > middle_sum:
        return GrashofResult(
            condition="non_grashof",
            classification="triple_rocker",
            shortest=shortest_role,
            s_plus_l=s_plus_l,
            p_plus_q=middle_sum,
            input_fully_rotates=False,
        )

    classification = {
        "ground": "double_crank",
        "input": "crank_rocker",
        "coupler": "double_rocker",
        "output": "rocker_crank",
    }[shortest_role]
    return GrashofResult(
        condition="grashof",
        classification=classification,
        shortest=shortest_role,
        s_plus_l=s_plus_l,
        p_plus_q=middle_sum,
        input_fully_rotates=classification in {"double_crank", "crank_rocker"},
    )


# ------------------------------------------------------- reachability, angles


def reachable_input_arc(
    lengths: FourBarLengths,
) -> tuple[float, float]:
    r"""Return ``(phi_min, phi_max)``: the reachable range of ``|theta2 - theta1|``.

    ``theta1`` is the direction of the ground link ``A -> D``. The dyad ``B-C-D``
    closes only while ``|r3 - r4| <= |BD| <= r3 + r4``, and

    .. math:: |BD|^2 = r_1^2 + r_2^2 - 2 r_1 r_2 \cos(\theta_2 - \theta_1)

    so the condition becomes a pair of bounds on ``cos(theta2 - theta1)``. The
    reachable set is ``phi_min <= |theta2 - theta1| <= phi_max`` with both in
    ``[0, pi]``. A fully rotating input gives ``(0, pi)``.

    Returns
    -------
    tuple of float
        Angles in radians. ``phi_min > phi_max`` means the linkage never closes.
    """
    r1, r2, r3, r4 = lengths.ground, lengths.input, lengths.coupler, lengths.output
    denom = 2.0 * r1 * r2
    if denom <= 0.0:
        raise ValueError("ground and input lengths must both be positive")

    # |BD| <= r3 + r4  ->  cos(phi) >= cos_lower_bound
    cos_lo = (r1**2 + r2**2 - (r3 + r4) ** 2) / denom
    # |BD| >= |r3 - r4| ->  cos(phi) <= cos_upper_bound
    cos_hi = (r1**2 + r2**2 - (r3 - r4) ** 2) / denom

    phi_max = float(np.arccos(np.clip(cos_lo, -1.0, 1.0))) if cos_lo > -1.0 else np.pi
    phi_min = float(np.arccos(np.clip(cos_hi, -1.0, 1.0))) if cos_hi < 1.0 else 0.0
    return phi_min, phi_max


def transmission_angle(pos_b: FloatArray, pos_c: FloatArray, pos_d: FloatArray) -> float:
    """Return the transmission angle at joint ``C``, in radians, within ``[0, pi]``.

    The transmission angle is the angle between the coupler ``C -> B`` and the
    output link ``C -> D``. It measures how effectively force in the coupler
    drives the output; values near 0 or pi mean the mechanism is near a toggle
    and transmits force badly.
    """
    to_b = np.asarray(pos_b, dtype=float) - np.asarray(pos_c, dtype=float)
    to_d = np.asarray(pos_d, dtype=float) - np.asarray(pos_c, dtype=float)
    norm = float(np.linalg.norm(to_b) * np.linalg.norm(to_d))
    if norm <= 0.0:
        raise ValueError("degenerate configuration: zero-length coupler or output link")
    cosine = float(np.dot(to_b, to_d)) / norm
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


# -------------------------------------------------------------------- solver


class FourBarSolver:
    """Position solver for a single-loop planar four-bar driven at a ground joint."""

    name = "four_bar"

    def can_solve(self, linkage: Linkage) -> bool:
        """Return whether ``linkage`` is a four-bar this solver can drive."""
        if len(linkage.bodies) != 4 or len(linkage.joints) != 4:
            return False
        if linkage.mobility() != 1:
            return False
        if len(linkage.independent_loops()) != 1:
            return False
        try:
            identify_four_bar(linkage)
        except LinkageError:
            return False
        return True

    def reference_branch(self, linkage: Linkage, roles: FourBarRoles | None = None) -> int:
        """Return the assembly branch of the reference (as-defined) configuration."""
        roles = roles or identify_four_bar(linkage)
        pos = {name: j.position_mm for name, j in linkage.joints.items()}
        return dyad_branch(pos[roles.joint_b], pos[roles.joint_d], pos[roles.joint_c])

    def reference_input_angle(self, linkage: Linkage, roles: FourBarRoles | None = None) -> float:
        """Return the input angle (rad) of the reference configuration."""
        roles = roles or identify_four_bar(linkage)
        pos = {name: j.position_mm for name, j in linkage.joints.items()}
        along = pos[roles.joint_b] - pos[roles.joint_a]
        return float(np.arctan2(along[1], along[0]))

    def solve(
        self,
        linkage: Linkage,
        input_angles_rad: FloatArray,
        *,
        branch: int | None = None,
        strict: bool = True,
        **_: Any,
    ) -> list[MechanismState]:
        """Solve the four-bar at each input angle.

        Parameters
        ----------
        input_angles_rad
            Absolute orientations of the input link, in radians.
        branch
            Assembly mode to stay on. Defaults to the reference configuration's
            branch, which is what keeps a swept path on one continuous curve.
        strict
            When ``True`` (default), an input angle at which the linkage cannot
            be assembled raises :class:`AssemblyError`. When ``False`` those
            angles are skipped, which is useful when scanning an unknown range.

        Raises
        ------
        AssemblyError
            If ``strict`` and the linkage cannot close at some input angle.
        """
        roles = identify_four_bar(linkage)
        lengths = link_lengths_of(linkage, roles)
        branch = self.reference_branch(linkage, roles) if branch is None else branch

        pos_a = linkage.joints[roles.joint_a].position_mm
        pos_d = linkage.joints[roles.joint_d].position_mm
        local_outputs = {
            name: linkage.to_local(out.body, out.position_mm)
            for name, out in linkage.outputs.items()
        }

        states: list[MechanismState] = []
        for theta2 in np.atleast_1d(np.asarray(input_angles_rad, dtype=float)):
            pos_b = pos_a + lengths.input * np.array([np.cos(theta2), np.sin(theta2)])
            pos_c = solve_dyad(pos_b, lengths.coupler, pos_d, lengths.output, branch=branch)
            if pos_c is None:
                if strict:
                    raise AssemblyError(
                        f"four-bar {linkage.name!r} does not assemble at "
                        f"theta2={np.degrees(theta2):.3f} deg; the reachable arc about the "
                        f"ground direction is "
                        f"{tuple(np.degrees(reachable_input_arc(lengths)))} deg"
                    )
                continue

            positions = {
                roles.joint_a: pos_a,
                roles.joint_b: pos_b,
                roles.joint_c: pos_c,
                roles.joint_d: pos_d,
            }
            body_angles = {
                name: (0.0 if body.is_ground else linkage.body_frame(name, positions)[1])
                for name, body in linkage.bodies.items()
            }
            outputs = {
                name: linkage.to_world(linkage.outputs[name].body, local, positions)
                for name, local in local_outputs.items()
            }
            states.append(
                MechanismState(
                    input_angle_rad=float(theta2),
                    joint_positions_mm=positions,
                    body_angles_rad=body_angles,
                    output_points_mm=outputs,
                    branch=branch,
                )
            )
        return states

    def diagnostics(self, linkage: Linkage, states: list[MechanismState]) -> dict[str, Any]:
        """Return per-sweep quality measures: transmission angle, toggle margin, branch.

        Both matter for design filtering. A branch flip during the sweep means
        the coupler curve jumps to the other assembly mode, which is a different
        mechanism, not a different pose.
        """
        roles = identify_four_bar(linkage)
        lengths = link_lengths_of(linkage, roles)
        grashof = classify_grashof(lengths)

        mu = np.array(
            [
                transmission_angle(
                    s.joint_positions_mm[roles.joint_b],
                    s.joint_positions_mm[roles.joint_c],
                    s.joint_positions_mm[roles.joint_d],
                )
                for s in states
            ],
            dtype=float,
        )
        clearance = np.array(
            [
                dyad_clearance(
                    s.joint_positions_mm[roles.joint_b],
                    lengths.coupler,
                    s.joint_positions_mm[roles.joint_d],
                    lengths.output,
                )
                for s in states
            ],
            dtype=float,
        )
        branches = {s.branch for s in states}
        phi_min, phi_max = reachable_input_arc(lengths)

        return {
            "roles": roles.__dict__,
            "link_lengths_mm": lengths.__dict__,
            "grashof": grashof.__dict__,
            "transmission_angle_deg": np.degrees(mu),
            "transmission_angle_min_deg": float(np.degrees(mu.min())) if mu.size else None,
            "transmission_angle_max_deg": float(np.degrees(mu.max())) if mu.size else None,
            "toggle_margin_mm": float(clearance.min()) if clearance.size else None,
            "near_toggle": bool(clearance.size and clearance.min() <= TOGGLE_TOL),
            "branch_flip": len(branches) > 1,
            "branch": sorted(branches),
            "reachable_arc_deg": [float(np.degrees(phi_min)), float(np.degrees(phi_max))],
        }


def input_sweep(
    linkage: Linkage, n_steps: int = 61, input_range_deg: tuple[float, float] | None = None
) -> FloatArray:
    """Return the sampled input angles (rad) for a linkage's input arc.

    The arc comes from ``input_range_deg`` if given, otherwise from the
    linkage's own ``input_range_deg``. A compliant mechanism always has a
    limited arc, so there is deliberately no "full revolution" default.
    """
    arc = input_range_deg or linkage.input_range_deg
    if arc is None:
        raise ValueError(
            f"linkage {linkage.name!r} has no input_range_deg; a compliant mechanism "
            "must be driven over a limited arc, so the range has to be stated"
        )
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    return np.radians(np.linspace(float(arc[0]), float(arc[1]), int(n_steps)))


def joint_rotations_rad(linkage: Linkage, states: list[MechanismState]) -> dict[str, FloatArray]:
    """Relative rotation at each joint versus the first state, in radians."""
    out: dict[str, FloatArray] = {}
    for name, joint in linkage.joints.items():
        first, second = joint.bodies
        rel = np.array(
            [s.body_angles_rad[first] - s.body_angles_rad[second] for s in states], dtype=float
        )
        out[name] = wrap_to_pi(rel - rel[0]) if rel.size else rel
    return out


KINEMATICS.add("four_bar", FourBarSolver())
