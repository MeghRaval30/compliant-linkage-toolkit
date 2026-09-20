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
from typing import Any, Protocol, runtime_checkable

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
    """Which PRBM model applies to a flexure, and how far it is from that model's edge.

    **This is metadata, never a filter.** The length ratio decides which
    pseudo-rigid-body model is used and is recorded on every sample; it does not
    decide whether a design is allowed. Phase C's fidelity map is a map of where
    the simple model stops working, so designs outside the small-length regime are
    exactly the ones worth keeping. Strain is the only hard physical limit.
    """

    length_ratio: float
    small_length_limit: float
    model: str
    bend_angle_deg: float
    slenderness: float
    max_angle_deg: float | None = None
    model_verified: bool = False

    @property
    def is_small_length(self) -> bool:
        """Whether the simple centre-pivot model applies."""
        return self.length_ratio <= self.small_length_limit

    @property
    def small_length_utilisation(self) -> float:
        """Length ratio as a fraction of the small-length limit.

        Above 1.0 the long-segment model is used instead. Reported so Phase A can
        *prefer* small-length designs for its pilot prints without excluding the
        others from the dataset.
        """
        return self.length_ratio / self.small_length_limit

    @property
    def near_model_boundary(self) -> bool:
        """Whether the length ratio sits close to the model-switch threshold.

        The two PRBM models are not continuous across it: the small-length model
        gives ``K = E I / L`` and the long-segment model ``gamma K_Theta E I / L``,
        about 2.25 times larger. Physical stiffness does not jump, so neither model
        is trustworthy right at the boundary and these samples are flagged rather
        than blended -- blending would be inventing a model. Milestone A4's beam
        FEA is what resolves this region.
        """
        return 0.7 <= self.small_length_utilisation <= 1.4

    @property
    def within_model_angle(self) -> bool:
        """Whether the bend stays inside the model's accurate angular range."""
        return self.max_angle_deg is None or self.bend_angle_deg <= self.max_angle_deg

    def notes(self) -> list[str]:
        """Human-readable caveats about this flexure's PRBM representation."""
        out: list[str] = []
        if not self.is_small_length:
            out.append(
                f"length ratio {self.length_ratio:.3f} exceeds the small-length limit "
                f"{self.small_length_limit:g}; using the {self.model!r} model, so the beam "
                f"FEA is the reference here rather than the PRBM"
            )
        if not self.within_model_angle and self.max_angle_deg is not None:
            out.append(
                f"bend {self.bend_angle_deg:.1f} deg exceeds the {self.model!r} model's "
                f"accurate range of about {self.max_angle_deg:.1f} deg"
            )
        if self.near_model_boundary:
            out.append(
                f"length ratio {self.length_ratio:.3f} sits near the model-switch "
                f"threshold {self.small_length_limit:g}; the two PRBM models differ by "
                "about 2.25x in stiffness there, so treat either with caution and use "
                "the beam FEA as the reference"
            )
        if not self.model_verified:
            out.append(
                f"the {self.model!r} model constants have not yet been checked against "
                "our own beam FEA (milestone A4)"
            )
        return out

    def to_dict(self) -> dict[str, float | str | bool | None]:
        """Return a JSON-serialisable summary."""
        return {
            "length_ratio": self.length_ratio,
            "small_length_limit": self.small_length_limit,
            "is_small_length": self.is_small_length,
            "small_length_utilisation": self.small_length_utilisation,
            "model": self.model,
            "bend_angle_deg": self.bend_angle_deg,
            "slenderness": self.slenderness,
            "max_angle_deg": self.max_angle_deg,
            "within_model_angle": self.within_model_angle,
            "near_model_boundary": self.near_model_boundary,
            "model_verified": self.model_verified,
        }


@runtime_checkable
class FlexureType(Protocol):
    """What every flexure type must provide."""

    name: str

    def second_moment_mm4(self, geometry: FlexureGeometry) -> float:
        """In-plane second moment of area, in mm^4."""
        ...

    def stiffness_nmm_per_rad(
        self,
        geometry: FlexureGeometry,
        youngs_modulus_mpa: float,
        *,
        model: Any = None,
        provenance: Any = None,
    ) -> float:
        """PRBM torsional spring constant ``K``, in N*mm/rad, for the given model."""
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

    def characteristic_pivot_fraction(self, *, model: Any = None, provenance: Any = None) -> float:
        """Where the PRBM pivot sits along the flexure, as a fraction from its root."""
        ...


#: Registry of flexure types.
FLEXURES: Registry[FlexureType] = Registry("flexure type")
