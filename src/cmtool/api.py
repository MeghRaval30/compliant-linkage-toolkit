"""The public, stable entry points of cmtool.

Everything here is deliberately small and solver-agnostic::

    mech   = Linkage.from_json("examples/fourbar.json")
    result = simulate(mech, solver="rigid", input_range_deg=(30, 80))
    result.path()

``convert(...)`` turns a rigid linkage into a compliant one::

    cm = convert(mech, material="PLA", printer="bambu_a1")
    cm.feasibility.binding_joint

Further solver names arrive in A3/A4; the shapes of :func:`simulate` and
:class:`~cmtool.solvers.base.SimulationResult` do not change when they do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cmtool.convert.base import STRATEGIES, CompliantMechanism
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


def convert(
    linkage: Linkage,
    *,
    flexures: str | dict[str, str] = "small_length_pivot",
    material: str = "PLA",
    printer: str = "bambu_a1",
    strategy: str = "naive",
    **options: Any,
) -> CompliantMechanism:
    """Convert a rigid linkage into a compliant (flexure-based) mechanism.

    Parameters
    ----------
    linkage
        The rigid mechanism.
    flexures
        A flexure type name for every joint, or a per-joint mapping. Mixed
        flexure types arrive with the notch and cross-axis pivots in Phase B; for
        now a mapping must name one type.
    material, printer
        Config names under ``configs/``. Every physical number used comes from
        those files, with provenance.
    strategy
        Registered design strategy. ``"naive"`` sizes each flexure to the
        shortest length that survives its bend.
    **options
        Passed to the strategy: ``placement``, ``unstressed_at``,
        ``input_range_deg``, ``thickness_mm``, ``flexure_length_mm``, ...

    Returns
    -------
    CompliantMechanism
        Check ``result.feasibility.feasible`` and, when it is ``False``,
        ``result.feasibility.binding_joint`` and ``.reasons()``.

    Notes
    -----
    Conversion always reads material and printer data, and those values are
    currently placeholders, so the result will report ``is_physical = False``
    until the coupon and modulus tests replace them.
    """
    engine = STRATEGIES.get(strategy)
    flexure_type = _sole_flexure_type(flexures)
    return engine.convert(
        linkage,
        flexure_type=flexure_type,
        material=material,
        printer=printer,
        **options,
    )


def _sole_flexure_type(flexures: str | dict[str, str]) -> str:
    """Return the single flexure type named, or explain why a mapping is not yet allowed."""
    if isinstance(flexures, str):
        return flexures
    distinct = set(flexures.values())
    if len(distinct) == 1:
        return distinct.pop()
    raise NotImplementedError(
        f"mixed flexure types per joint are not implemented yet (got {sorted(distinct)}); "
        "the notch and cross-axis pivots arrive in Phase B"
    )


def available_solvers() -> list[str]:
    """Return the names of all registered solvers."""
    return SOLVERS.names()


def available_flexures() -> list[str]:
    """Return the names of all registered flexure types."""
    from cmtool.flexures import FLEXURES

    return FLEXURES.names()


def available_strategies() -> list[str]:
    """Return the names of all registered design strategies."""
    return STRATEGIES.names()
