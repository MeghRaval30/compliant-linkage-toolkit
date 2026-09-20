"""Choosing an input arc that keeps every joint within a flexure's reach.

A rigid four-bar is happy to sweep 60 degrees. The flexures replacing its joints
are not: the pilot example's joint B rotates 71.9 degrees over a 60 degree input
arc, far more than a small-length flexural pivot can take at a printable
thickness.

So the input arc becomes a **design variable**, not a given. :func:`fit_input_arc`
shrinks the arc about a chosen centre until the worst joint excursion meets a
target, and reports which joint was the worst.

Note that joint excursions are generally larger than the input excursion -- a
four-bar amplifies rotation at some joints -- so the arc that satisfies a 22
degree joint limit is usually considerably narrower than 22 degrees of input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import cmtool.solvers  # noqa: F401  (registers the rigid solver)
from cmtool.core.graph import Linkage
from cmtool.solvers.base import SOLVERS, SimulationResult


class ArcFitError(RuntimeError):
    """Raised when no usable input arc exists for the requested limit."""


def sweep(
    linkage: Linkage, input_range_deg: tuple[float, float], n_steps: int = 41
) -> SimulationResult:
    """Run the rigid solver over an explicit arc."""
    angles = np.radians(np.linspace(float(input_range_deg[0]), float(input_range_deg[1]), n_steps))
    return SOLVERS.get("rigid").solve(linkage, angles)


def joint_excursions_deg(
    linkage: Linkage, input_range_deg: tuple[float, float], n_steps: int = 41
) -> dict[str, float] | None:
    """Peak-to-peak joint rotations over an arc, or ``None`` if it cannot be swept."""
    try:
        result = sweep(linkage, input_range_deg, n_steps)
    except Exception:
        # Not assemblable across the whole arc, or it crosses a toggle.
        return None
    if result.diagnostics.get("branch_flip"):
        return None
    return result.joint_excursion_deg()


@dataclass
class ArcFit:
    """The outcome of fitting an input arc to a joint-excursion limit."""

    input_range_deg: tuple[float, float]
    centre_deg: float
    half_width_deg: float
    target_excursion_deg: float
    excursions_deg: dict[str, float] = field(default_factory=dict)

    @property
    def max_excursion_deg(self) -> float:
        """Worst joint excursion over the fitted arc."""
        return max(self.excursions_deg.values()) if self.excursions_deg else 0.0

    @property
    def binding_joint(self) -> str:
        """The joint that limits the arc."""
        return max(self.excursions_deg, key=lambda k: self.excursions_deg[k])

    @property
    def input_excursion_deg(self) -> float:
        """Total input rotation over the fitted arc."""
        return 2.0 * self.half_width_deg

    @property
    def amplification(self) -> float:
        """Worst joint excursion divided by the input excursion."""
        if self.input_excursion_deg <= 0.0:
            return float("inf")
        return self.max_excursion_deg / self.input_excursion_deg

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "input_range_deg": list(self.input_range_deg),
            "centre_deg": self.centre_deg,
            "input_excursion_deg": self.input_excursion_deg,
            "target_excursion_deg": self.target_excursion_deg,
            "max_excursion_deg": self.max_excursion_deg,
            "binding_joint": self.binding_joint if self.excursions_deg else None,
            "amplification": self.amplification,
            "excursions_deg": dict(self.excursions_deg),
        }


def fit_input_arc(
    linkage: Linkage,
    *,
    max_joint_excursion_deg: float = 22.0,
    centre_deg: float | None = None,
    n_steps: int = 41,
    max_half_width_deg: float = 90.0,
    tol_deg: float = 0.05,
) -> ArcFit:
    """Find the widest symmetric input arc keeping every joint within a limit.

    Parameters
    ----------
    linkage
        The rigid linkage.
    max_joint_excursion_deg
        Target for the largest peak-to-peak joint rotation.
    centre_deg
        Centre of the arc. Defaults to the midpoint of the linkage's own
        ``input_range_deg``, or its reference input angle.
    n_steps
        Samples used when evaluating a candidate arc.
    max_half_width_deg
        Upper bound for the search.
    tol_deg
        Bisection tolerance on the half-width.

    Returns
    -------
    ArcFit

    Raises
    ------
    ArcFitError
        If even a very narrow arc cannot be swept, which means the linkage does
        not assemble near the centre or sits on a toggle there.

    Notes
    -----
    The search assumes joint excursion grows monotonically with arc width about
    the centre, which holds away from toggle positions. The returned arc is
    verified against the limit rather than assumed to satisfy it.
    """
    if max_joint_excursion_deg <= 0.0:
        raise ValueError("max_joint_excursion_deg must be positive")

    if centre_deg is None:
        if linkage.input_range_deg is not None:
            centre_deg = float(np.mean(linkage.input_range_deg))
        else:
            from cmtool.kinematics.fourbar import FourBarSolver

            centre_deg = float(np.degrees(FourBarSolver().reference_input_angle(linkage)))

    def excursion_at(half_width: float) -> float | None:
        arc = (centre_deg - half_width, centre_deg + half_width)
        values = joint_excursions_deg(linkage, arc, n_steps)
        return max(values.values()) if values else None

    seed = min(0.25, max_half_width_deg / 4.0)
    if excursion_at(seed) is None:
        raise ArcFitError(
            f"linkage {linkage.name!r} cannot be swept even +/-{seed} deg about "
            f"{centre_deg:.2f} deg; it does not assemble there or is at a toggle"
        )

    lo = seed
    hi = max_half_width_deg
    wide = excursion_at(hi)
    if wide is not None and wide <= max_joint_excursion_deg:
        lo = hi
    else:
        while hi - lo > tol_deg:
            mid = 0.5 * (lo + hi)
            value = excursion_at(mid)
            if value is not None and value <= max_joint_excursion_deg:
                lo = mid
            else:
                hi = mid

    arc = (centre_deg - lo, centre_deg + lo)
    excursions = joint_excursions_deg(linkage, arc, n_steps)
    if excursions is None:  # pragma: no cover - guarded by the seed check above
        raise ArcFitError(f"failed to evaluate the fitted arc {arc} for {linkage.name!r}")

    return ArcFit(
        input_range_deg=arc,
        centre_deg=float(centre_deg),
        half_width_deg=float(lo),
        target_excursion_deg=float(max_joint_excursion_deg),
        excursions_deg=excursions,
    )
