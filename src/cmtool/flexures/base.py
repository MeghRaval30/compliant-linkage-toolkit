"""Flexure types: geometry, PRBM stiffness and strain.

A flexure type owns three things that must stay together:

* the geometry that gets printed,
* the pseudo-rigid-body model that represents it (stiffness and characteristic
  pivot location),
* **its own strain model**.

The third is why this is a plug-in interface rather than a set of free functions.
The leaf formula ``eps = t*theta/(2L)`` is right for a prismatic strip and wrong
for a notch hinge, which concentrates strain at its thinnest section. Making
``peak_strain`` a method of the type means a new flexure cannot silently inherit
a model that does not apply to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from cmtool.core.registry import Registry


@dataclass(frozen=True)
class FlexureGeometry:
    """Dimensions of one flexure, in mm.

    Attributes
    ----------
    thickness_mm
        In-plane thickness ``t``: the dimension that bends. The small one.
    length_mm
        Length ``L`` along the neutral axis.
    width_mm
        Out-of-plane width ``w``, equal to the printed part thickness.
    """

    thickness_mm: float
    length_mm: float
    width_mm: float

    def __post_init__(self) -> None:
        for name in ("thickness_mm", "length_mm", "width_mm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"flexure {name} must be positive and finite, got {value}")

    @property
    def slenderness(self) -> float:
        """Ratio ``L / t``. Higher means a longer, gentler flexure."""
        return self.length_mm / self.thickness_mm

    @property
    def aspect_ratio(self) -> float:
        """Ratio ``w / t``.

        Out-of-plane and in-plane second moments differ by the square of this, so
        it is the number that decides whether the planar assumption holds
        (``docs/physics.md`` section 7).
        """
        return self.width_mm / self.thickness_mm


@dataclass(frozen=True)
class StrainEstimate:
    """Peak strain in a flexure, with the model that produced it named."""

    peak_strain: float
    model: str
    bend_angle_rad: float
    includes_axial: bool = False
    includes_stress_concentration: bool = False

    @property
    def bend_angle_deg(self) -> float:
        """Bend angle in degrees."""
        return float(np.degrees(self.bend_angle_rad))


@dataclass(frozen=True)
class PrbmValidity:
    """How far inside the pseudo-rigid-body model's validity envelope a flexure sits.

    Recorded for every sample rather than assumed, so that Phase C's fidelity map
    (where PRBM suffices and where FEA is needed) comes out of data already
    collected. See ``docs/physics.md`` section 5.
    """

    length_ratio: float
    length_ratio_limit: float
    bend_angle_deg: float
    slenderness: float

    @property
    def length_ratio_ok(self) -> bool:
        """Whether the flexure is short enough relative to its links."""
        return self.length_ratio <= self.length_ratio_limit

    @property
    def utilisation(self) -> float:
        """Length ratio as a fraction of its limit; > 1 means outside the envelope."""
        return self.length_ratio / self.length_ratio_limit


@runtime_checkable
class FlexureType(Protocol):
    """What every flexure type must provide."""

    name: str

    def second_moment_mm4(self, geometry: FlexureGeometry) -> float:
        """In-plane second moment of area, in mm^4."""
        ...

    def stiffness_nmm_per_rad(self, geometry: FlexureGeometry, youngs_modulus_mpa: float) -> float:
        """PRBM torsional spring constant ``K``, in N*mm/rad."""
        ...

    def peak_strain(self, geometry: FlexureGeometry, bend_angle_rad: float) -> StrainEstimate:
        """Peak bending strain at a given relative rotation."""
        ...

    def max_bend_angle_rad(self, geometry: FlexureGeometry, allowable_strain: float) -> float:
        """Largest bend angle that keeps peak strain within the allowable."""
        ...

    def min_length_mm(
        self, thickness_mm: float, bend_angle_rad: float, allowable_strain: float
    ) -> float:
        """Shortest flexure that can take ``bend_angle_rad`` within the allowable strain."""
        ...

    def characteristic_pivot_fraction(self) -> float:
        """Where the PRBM pivot sits along the flexure, as a fraction of its length."""
        ...


#: Registry of flexure types.
FLEXURES: Registry[FlexureType] = Registry("flexure type")
