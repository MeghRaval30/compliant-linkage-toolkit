r"""Closed-form design limits: how much bend a flexure joint can actually take.

Two constraints act on a small-length flexural pivot at once, and they pull in
opposite directions:

* **Strain** wants the flexure *long*: ``L >= t * theta * SF / (2 eps_allow)``.
* **PRBM validity** wants it *short*: ``L <= r * l``, where ``l`` is the shorter
  adjacent link and ``r`` the small-length ratio limit.

Eliminating ``L`` between them gives the bound that actually governs design:

.. math::

    \theta_{max} = \frac{2 r \, l \, \varepsilon_{allow}}{t \, SF}

This is worth stating explicitly because it is counter-intuitive: the largest
usable joint rotation is set by the **length of the adjacent link**, not by
anything about the flexure alone. Short links cannot carry large rotations, at
any flexure thickness, without leaving the pseudo-rigid-body model's envelope.

It also says exactly which levers exist, and their order of usefulness:

1. longer adjacent links (linear)
2. thinner flexures (inverse)
3. a higher measured allowable strain (linear)
4. accepting a larger ``r``, which trades PRBM accuracy for range -- a legitimate
   research choice for this project, but one that must be recorded per sample
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignLimit:
    """The governing bound on joint rotation, and the inputs that produced it."""

    max_bend_deg: float
    shortest_link_mm: float
    thickness_mm: float
    allowable_strain: float
    length_ratio_limit: float
    strain_safety_factor: float

    @property
    def max_excursion_deg(self) -> float:
        """Peak-to-peak excursion when the part is printed unstressed at mid-arc.

        Printing at mid-arc means the flexure swings symmetrically about zero, so
        the usable peak-to-peak range is twice the one-sided bound.
        """
        return 2.0 * self.max_bend_deg


def max_bend_deg(
    shortest_link_mm: float,
    thickness_mm: float,
    allowable_strain: float,
    *,
    length_ratio_limit: float = 0.1,
    strain_safety_factor: float = 1.5,
) -> DesignLimit:
    """Largest one-sided bend a joint can take within both constraints."""
    for name, value in (
        ("shortest_link_mm", shortest_link_mm),
        ("thickness_mm", thickness_mm),
        ("allowable_strain", allowable_strain),
        ("length_ratio_limit", length_ratio_limit),
        ("strain_safety_factor", strain_safety_factor),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")

    theta = (2.0 * length_ratio_limit * shortest_link_mm * allowable_strain) / (
        thickness_mm * strain_safety_factor
    )
    return DesignLimit(
        max_bend_deg=float(np.degrees(theta)),
        shortest_link_mm=float(shortest_link_mm),
        thickness_mm=float(thickness_mm),
        allowable_strain=float(allowable_strain),
        length_ratio_limit=float(length_ratio_limit),
        strain_safety_factor=float(strain_safety_factor),
    )


def required_link_length_mm(
    bend_deg: float,
    thickness_mm: float,
    allowable_strain: float,
    *,
    length_ratio_limit: float = 0.1,
    strain_safety_factor: float = 1.5,
) -> float:
    """Shortest adjacent link that can support a required one-sided bend.

    The inverse of :func:`max_bend_deg`. Use it when sizing a mechanism to a
    motion specification: it says how long the links have to be before the
    requested rotation is achievable at all.
    """
    if bend_deg <= 0.0:
        raise ValueError("bend_deg must be positive")
    theta = float(np.radians(bend_deg))
    return (theta * thickness_mm * strain_safety_factor) / (
        2.0 * length_ratio_limit * allowable_strain
    )


def min_link_length_for_excursion_mm(
    excursion_deg: float,
    thickness_mm: float,
    allowable_strain: float,
    *,
    unstressed_at: str = "mid_arc",
    length_ratio_limit: float = 0.1,
    strain_safety_factor: float = 1.5,
) -> float:
    """Shortest adjacent link supporting a peak-to-peak excursion.

    Printing unstressed at mid-arc halves the one-sided bend and therefore halves
    the link length required -- which is why it is the default in
    :mod:`cmtool.convert.naive`.
    """
    bend = excursion_deg / 2.0 if unstressed_at == "mid_arc" else excursion_deg
    return required_link_length_mm(
        bend,
        thickness_mm,
        allowable_strain,
        length_ratio_limit=length_ratio_limit,
        strain_safety_factor=strain_safety_factor,
    )
