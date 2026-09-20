"""Pseudo-rigid-body models: where the pivot sits, and how stiff the spring is.

A prismatic flexure is one piece of geometry, but it has **two** pseudo-rigid-body
models depending on how long it is relative to the links it joins:

``small_length``
    A short flexure behaves as a pin at its own **centre**, with ``K = E I / L``.
    No fitted constant is involved; it follows from uniform curvature over a short
    segment.

``long_segment``
    Howell's model for a flexible cantilever carrying an end force. The segment is
    replaced by a rigid link of length ``gamma * L`` pinned at ``(1 - gamma) * L``
    from the root, with ``K = gamma * K_Theta * E I / L``.

The distinction matters because the length ratio ``L_flexure / L_link`` is
**metadata, not a filter**. Designs past the small-length limit are kept
deliberately: Phase C's fidelity map is a map of where the simple model fails, and
filtering those designs out would hide the result it is meant to show. What the
ratio does is *select the model* and get recorded on every sample.

Model constants live in ``configs/models/prbm.yaml`` with citations, and are
marked unverified until milestone A4's beam FEA checks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from cmtool.core.config import dig, load_config, quantity
from cmtool.core.provenance import Provenance
from cmtool.core.registry import Registry
from cmtool.flexures.base import FlexureGeometry


@runtime_checkable
class PrbmModel(Protocol):
    """A pseudo-rigid-body representation of one flexure."""

    @property
    def name(self) -> str:
        """Registered name of the model."""
        ...

    def characteristic_pivot_fraction(self, provenance: Provenance | None = None) -> float:
        """Where the pivot sits along the flexure, as a fraction from its root."""
        ...

    def stiffness_nmm_per_rad(
        self,
        geometry: FlexureGeometry,
        youngs_modulus_mpa: float,
        second_moment_mm4: float,
        provenance: Provenance | None = None,
    ) -> float:
        """Torsional spring constant ``K``, in N*mm/rad."""
        ...

    def max_angle_deg(self, provenance: Provenance | None = None) -> float | None:
        """Angle beyond which the model is reported as out of its accurate range."""
        ...


#: Registry of PRBM models.
PRBM_MODELS: Registry[PrbmModel] = Registry("PRBM model")


def _config() -> dict[str, Any]:
    """Load ``configs/models/prbm.yaml``."""
    data, _ = load_config("models", "prbm")
    return data


@dataclass(frozen=True)
class SmallLengthModel:
    """Pin at the flexure centre with ``K = E I / L``.

    Exact for a flexure short enough that the segments it joins are effectively
    rigid and its own curvature is uniform.
    """

    name: str = "small_length"

    def characteristic_pivot_fraction(self, provenance: Provenance | None = None) -> float:
        """Return 0.5: the pivot is at the centre of the flexure."""
        return quantity(_config(), "small_length.characteristic_pivot_fraction", prefix="prbm").get(
            provenance
        )

    def stiffness_nmm_per_rad(
        self,
        geometry: FlexureGeometry,
        youngs_modulus_mpa: float,
        second_moment_mm4: float,
        provenance: Provenance | None = None,
    ) -> float:
        """Return ``K = E I / L``."""
        coefficient = quantity(_config(), "small_length.stiffness_coefficient", prefix="prbm").get(
            provenance
        )
        return coefficient * youngs_modulus_mpa * second_moment_mm4 / geometry.length_mm

    def max_angle_deg(self, provenance: Provenance | None = None) -> float | None:
        """Return ``None``: the small-length model has no separate angle limit.

        Its accuracy is bounded by the strain limit rather than by a fitted range.
        """
        return None


@dataclass(frozen=True)
class LongSegmentModel:
    """Howell's PRBM for a flexible segment with an end force.

    Pivot at ``(1 - gamma) L`` from the root, ``K = gamma * K_Theta * E I / L``.

    Boundary conditions
    -------------------
    The model is derived for a cantilever: fixed at the root, loaded at the free
    end. A flexure joining two moving links is not exactly that, so which end
    counts as the root matters and the asymmetry is real. This is the single
    biggest assumption in the long-segment path, and it is precisely what the
    beam FEA in A4 exists to check.
    """

    name: str = "long_segment"

    def gamma(self, provenance: Provenance | None = None) -> float:
        """Characteristic radius factor."""
        return quantity(_config(), "long_segment.gamma", prefix="prbm").get(provenance)

    def stiffness_coefficient(self, provenance: Provenance | None = None) -> float:
        """Stiffness coefficient ``K_Theta``."""
        return quantity(_config(), "long_segment.stiffness_coefficient", prefix="prbm").get(
            provenance
        )

    def characteristic_pivot_fraction(self, provenance: Provenance | None = None) -> float:
        """Return ``1 - gamma``: the pivot sits that far from the flexure root."""
        return 1.0 - self.gamma(provenance)

    def stiffness_nmm_per_rad(
        self,
        geometry: FlexureGeometry,
        youngs_modulus_mpa: float,
        second_moment_mm4: float,
        provenance: Provenance | None = None,
    ) -> float:
        """Return ``K = gamma * K_Theta * E I / L``."""
        gamma = self.gamma(provenance)
        k_theta = self.stiffness_coefficient(provenance)
        return gamma * k_theta * youngs_modulus_mpa * second_moment_mm4 / geometry.length_mm

    def max_angle_deg(self, provenance: Provenance | None = None) -> float | None:
        """Return the PRBM angle beyond which the model is reported as strained."""
        return quantity(_config(), "long_segment.max_angle_deg", prefix="prbm").get(provenance)


PRBM_MODELS.add("small_length", SmallLengthModel())
PRBM_MODELS.add("long_segment", LongSegmentModel())


def small_length_ratio_limit(provenance: Provenance | None = None) -> float:
    """Length ratio at which the model switches from small-length to long-segment."""
    return quantity(_config(), "small_length.length_ratio_limit", prefix="prbm").get(provenance)


def max_length_fraction_of_link(provenance: Provenance | None = None) -> float:
    """Hard geometric cap on flexure length, as a fraction of the shorter link.

    This is the real limit: a flexure is a necked-down part of a link, so some
    rigid material has to remain at each end. It replaced the small-length ratio
    as the feasibility constraint when PRBM validity became metadata.
    """
    return quantity(_config(), "geometry.max_length_fraction_of_link", prefix="prbm").get(
        provenance
    )


def model_is_verified() -> bool:
    """Whether the PRBM constants have been checked against our own beam FEA."""
    return bool(dig(_config(), "verification.verified_against_fea"))


def select_model(length_ratio: float, *, provenance: Provenance | None = None) -> PrbmModel:
    """Pick the PRBM model appropriate to a flexure's length ratio.

    Below the small-length limit the flexure is modelled as a pin at its centre;
    above it, Howell's long-segment model applies. Selection is automatic but the
    choice is always recorded, and either model can be forced.
    """
    limit = small_length_ratio_limit(provenance)
    return PRBM_MODELS.get("small_length" if length_ratio <= limit else "long_segment")
