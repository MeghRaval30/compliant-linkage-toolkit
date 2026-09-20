"""Pseudo-rigid-body models: where the pivot sits, and how stiff the spring is.

A prismatic flexure is one piece of geometry, but which pseudo-rigid-body model
represents it depends on how long it is relative to the links it joins:

``small_length``
    A short flexure behaves as a pin at its own **centre**, with ``K = E I / L``.
    No fitted constant is involved; it follows from uniform curvature over a short
    segment.

``long_segment``
    Howell's model for a flexible cantilever. The segment is replaced by a rigid
    link of length ``gamma * L`` pinned at ``(1 - gamma) * L`` **from the root**.

Two things about the long-segment model are easy to get wrong, and both are
checked in ``tests/validation/test_prbm.py``:

**The pivot is at ``(1 - gamma) L`` from the root, not ``gamma L``.** Fitting the
exact circular-arc solution with the pivot at ``(1-gamma)L`` reproduces the tip
path to 2e-4 L; the other convention is off by 0.15 L, and for ``gamma = 0.85``
the rigid link is then too short to reach the tip at all.

**The stiffness formula differs by loading case.** End force uses
``K = gamma * K_Theta * E I / L``; end moment uses ``K = K_Theta * E I / L``, with
no ``gamma``. Multiplying the two end-moment constants together gives ~1.11, which
looks as though it would erase the stiffness step at the model boundary. It does
not: the correct end-moment stiffness is ~1.50 E I / L, so the step falls from
2.25x to 1.52x.

The length ratio ``L_flexure / L_link`` is **metadata, not a filter**. Designs past
the small-length limit are kept deliberately: Phase C's fidelity map is a map of
where the simple model fails, and filtering those designs out would hide the
result it is meant to show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from cmtool.core.config import ConfigError, dig, load_config, quantity
from cmtool.core.provenance import Provenance
from cmtool.core.registry import Registry
from cmtool.flexures.base import FlexureGeometry

#: Stiffness formulas a variant can declare.
STIFFNESS_FORMULAS = {
    "k_theta_ei_over_l": "K = K_Theta * E I / L",
    "gamma_k_theta_ei_over_l": "K = gamma * K_Theta * E I / L",
}


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
    """Howell's PRBM for a flexible segment, in one of its loading variants.

    Parameters
    ----------
    variant
        ``"end_force"`` or ``"end_moment"``. The two differ in both ``gamma`` and
        the stiffness formula, so they are separate registered models rather than
        one model with a switch.

    Boundary conditions
    -------------------
    The model is derived for a cantilever: fixed at the root, loaded at the free
    end. A flexure joining two moving links is neither purely force-loaded nor
    purely moment-loaded, so which variant applies is an empirical question --
    which is what the A4 beam FEA study answers.
    """

    variant: str = "end_force"

    @property
    def name(self) -> str:
        """Registered name, for example ``"long_segment_end_force"``."""
        return f"long_segment_{self.variant}"

    def _entry(self, key: str) -> str:
        return f"long_segment.variants.{self.variant}.{key}"

    def gamma(self, provenance: Provenance | None = None) -> float:
        """Characteristic radius factor."""
        return quantity(_config(), self._entry("gamma"), prefix="prbm").get(provenance)

    def stiffness_coefficient(self, provenance: Provenance | None = None) -> float:
        """Stiffness coefficient ``K_Theta``."""
        return quantity(_config(), self._entry("stiffness_coefficient"), prefix="prbm").get(
            provenance
        )

    def stiffness_formula(self) -> str:
        """Which formula this variant uses; a key of :data:`STIFFNESS_FORMULAS`."""
        formula = str(dig(_config(), self._entry("stiffness_formula")))
        if formula not in STIFFNESS_FORMULAS:
            raise ConfigError(f"unknown stiffness formula {formula!r} for {self.name}")
        return formula

    def characteristic_pivot_fraction(self, provenance: Provenance | None = None) -> float:
        """Return ``1 - gamma``: the pivot sits that far from the flexure root.

        Not ``gamma``. The rigid link of length ``gamma L`` runs from the pivot to
        the tip, so the pivot is ``(1 - gamma) L`` back from the root.
        """
        return 1.0 - self.gamma(provenance)

    def stiffness_multiple(self, provenance: Provenance | None = None) -> float:
        """``K`` as a multiple of ``E I / L``, per this variant's stiffness formula."""
        k_theta = self.stiffness_coefficient(provenance)
        if self.stiffness_formula() == "gamma_k_theta_ei_over_l":
            return self.gamma(provenance) * k_theta
        return k_theta

    def stiffness_nmm_per_rad(
        self,
        geometry: FlexureGeometry,
        youngs_modulus_mpa: float,
        second_moment_mm4: float,
        provenance: Provenance | None = None,
    ) -> float:
        """Return ``K`` in N*mm/rad, using this variant's stiffness formula."""
        multiple = self.stiffness_multiple(provenance)
        return multiple * youngs_modulus_mpa * second_moment_mm4 / geometry.length_mm

    def max_angle_deg(self, provenance: Provenance | None = None) -> float | None:
        """Return the PRBM angle beyond which the model is reported as strained."""
        return quantity(_config(), "long_segment.max_angle_deg", prefix="prbm").get(provenance)


PRBM_MODELS.add("small_length", SmallLengthModel())
PRBM_MODELS.add("long_segment_end_force", LongSegmentModel("end_force"))
PRBM_MODELS.add("long_segment_end_moment", LongSegmentModel("end_moment"))


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


def active_long_segment_variant() -> str:
    """Which long-segment variant the toolkit currently applies."""
    return str(dig(_config(), "long_segment.active_variant"))


def active_long_segment_model() -> LongSegmentModel:
    """Return the registered long-segment model for the active variant."""
    model = PRBM_MODELS.get(f"long_segment_{active_long_segment_variant()}")
    assert isinstance(model, LongSegmentModel)
    return model


def model_is_verified() -> bool:
    """Whether the PRBM constants have been checked against our own beam FEA."""
    return bool(dig(_config(), "verification.verified_against_fea"))


def stiffness_step_at_boundary(provenance: Provenance | None = None) -> float:
    """Ratio of long-segment to small-length stiffness at the model switch.

    Physical stiffness is continuous, so a value far from 1.0 marks a region where
    neither model can be trusted. About 2.25 for the end-force variant and 1.52
    for end-moment.
    """
    return active_long_segment_model().stiffness_multiple(provenance)


def select_model(length_ratio: float, *, provenance: Provenance | None = None) -> PrbmModel:
    """Pick the PRBM model appropriate to a flexure's length ratio.

    Below the small-length limit the flexure is modelled as a pin at its centre;
    above it, the active long-segment variant applies. Selection is automatic but
    the choice is always recorded, and either model can be forced.
    """
    limit = small_length_ratio_limit(provenance)
    if length_ratio <= limit:
        return PRBM_MODELS.get("small_length")
    return active_long_segment_model()
