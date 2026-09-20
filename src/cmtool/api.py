"""The public, stable entry points of cmtool.

Everything here is deliberately small and solver-agnostic::

    mech   = Linkage.from_json("examples/fourbar.json")
    result = simulate(mech, solver="rigid", input_range_deg=(30, 80))
    result.path()

Later phases add ``convert(...)`` (rigid -> compliant) and further solver names;
the shapes of :func:`simulate` and :class:`~cmtool.solvers.base.SimulationResult`
do not change when they do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cmtool.core.graph import Linkage
from cmtool.core.units import FloatArray
from cmtool.solvers.base import SOLVERS, SimulationResult


def simulate(
    linkage: Linkage,
    *,
    solver: str = "rigid",
    input_range_deg: tuple[float, float] | None = None,
    n_steps: int = 61,
    input_angles_deg: FloatArray | None = None,
    **solver_kwargs: Any,
) -> SimulationResult:
    """Simulate a linkage over a limited input arc.

    Parameters
    ----------
    linkage
        The mechanism to solve.
    solver
        Registered solver name. ``"rigid"`` is the only one available in Phase A
        milestone A1; ``"prbm"``, ``"beam_fea"`` and ``"solid_fea"`` follow.
    input_range_deg
        ``(start, end)`` absolute orientation of the input link, in degrees.
        Falls back to the linkage's own ``input_range_deg``.
    n_steps
        Number of samples across the arc, inclusive of both ends.
    input_angles_deg
        Explicit sample angles, overriding ``input_range_deg`` and ``n_steps``.
        Useful for evaluating simulation and measurement at matched angles.
    **solver_kwargs
        Forwarded to the solver (for example ``branch`` for the four-bar).

    Returns
    -------
    SimulationResult
        Always check ``result.is_physical``: ``False`` means a placeholder value
        was used and the numbers are not a physical prediction.

    Notes
    -----
    There is no default "full revolution" sweep, by design. Flexures cannot
    rotate continuously, so every simulation states its arc explicitly.
    """
    engine = SOLVERS.get(solver)
    if not engine.can_solve(linkage):
        raise ValueError(
            f"solver {solver!r} cannot handle linkage {linkage.name!r} "
            f"(bodies={len(linkage.bodies)}, joints={len(linkage.joints)}, "
            f"mobility={linkage.mobility()})"
        )

    if input_angles_deg is not None:
        angles = np.radians(np.asarray(input_angles_deg, dtype=float))
    else:
        arc = input_range_deg or linkage.input_range_deg
        if arc is None:
            raise ValueError(
                f"no input arc given: pass input_range_deg, or set it on linkage {linkage.name!r}"
            )
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        angles = np.radians(np.linspace(float(arc[0]), float(arc[1]), int(n_steps)))

    return engine.solve(linkage, angles, **solver_kwargs)


def available_solvers() -> list[str]:
    """Return the names of all registered solvers."""
    return SOLVERS.names()
