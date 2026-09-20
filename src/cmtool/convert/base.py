"""Conversion data types: per-joint sizing, feasibility, and the compliant mechanism.

The feasibility model is deliberately simple and explicit, because its job is to
explain *why* a design fails, not just that it did.

For each joint, two lengths are computed:

``L_strain``
    the shortest flexure that can take the required bend within the allowable
    strain, ``t * theta / (2 eps_allow)``, times a safety factor.

``L_prbm``
    the longest flexure that still counts as "small-length",
    ``ratio_limit * (shorter adjacent link)``.

A joint is feasible when ``L_strain <= L_prbm``. Their ratio is the joint's
**utilisation**, and the joint with the highest utilisation is the one limiting
the design. That is the number to report to a designer: it says which joint to
fix and by how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from cmtool.core.graph import Joint, Linkage
from cmtool.core.provenance import Provenance
from cmtool.core.registry import Registry
from cmtool.core.units import rotation_matrix
from cmtool.flexures.base import FlexureGeometry, PrbmValidity, StrainEstimate


@dataclass(frozen=True)
class JointSizing:
    """Sizing and verdict for the flexure at one joint."""

    joint: str
    flexure_type: str
    geometry: FlexureGeometry
    max_bend_rad: float
    excursion_deg: float
    shortest_adjacent_link_mm: float
    min_length_strain_mm: float
    max_length_prbm_mm: float
    strain: StrainEstimate
    validity: PrbmValidity
    stiffness_nmm_per_rad: float
    host_body: str
    pivot_offset_mm: float

    @property
    def max_bend_deg(self) -> float:
        """Largest bend from the unstressed configuration, in degrees."""
        return float(np.degrees(abs(self.max_bend_rad)))

    @property
    def utilisation(self) -> float:
        """``L_strain / L_prbm``. Above 1.0 the joint cannot be built as specified."""
        if self.max_length_prbm_mm <= 0.0:
            return float("inf")
        return self.min_length_strain_mm / self.max_length_prbm_mm

    @property
    def strain_ok(self) -> bool:
        """Whether the chosen length keeps peak strain within the allowable."""
        return self.geometry.length_mm >= self.min_length_strain_mm - 1e-12

    @property
    def prbm_ok(self) -> bool:
        """Whether the chosen length stays inside the PRBM validity envelope."""
        return self.geometry.length_mm <= self.max_length_prbm_mm + 1e-12

    @property
    def feasible(self) -> bool:
        """Whether this joint satisfies both constraints at once."""
        return self.strain_ok and self.prbm_ok

    @property
    def limit_reason(self) -> str | None:
        """Why this joint fails, or ``None`` if it does not."""
        if self.feasible:
            return None
        if not self.prbm_ok:
            return (
                f"needs L >= {self.min_length_strain_mm:.2f} mm for "
                f"{self.max_bend_deg:.1f} deg of bend, but PRBM validity caps it at "
                f"{self.max_length_prbm_mm:.2f} mm "
                f"({self.validity.length_ratio_limit:g} x the "
                f"{self.shortest_adjacent_link_mm:.1f} mm adjacent link)"
            )
        return (
            f"peak strain {self.strain.peak_strain:.4f} exceeds the allowable at "
            f"L = {self.geometry.length_mm:.2f} mm"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "joint": self.joint,
            "type": self.flexure_type,
            "t_mm": self.geometry.thickness_mm,
            "length_mm": self.geometry.length_mm,
            "width_mm": self.geometry.width_mm,
            "max_bend_deg": self.max_bend_deg,
            "excursion_deg": self.excursion_deg,
            "peak_strain": self.strain.peak_strain,
            "strain_model": self.strain.model,
            "min_length_strain_mm": self.min_length_strain_mm,
            "max_length_prbm_mm": self.max_length_prbm_mm,
            "prbm_length_ratio": self.validity.length_ratio,
            "stiffness_nmm_per_rad": self.stiffness_nmm_per_rad,
            "utilisation": self.utilisation,
            "feasible": self.feasible,
            "limit_reason": self.limit_reason,
        }


@dataclass
class FeasibilityReport:
    """Whether a converted design can be built, and which joint decides that."""

    joints: dict[str, JointSizing]
    allowable_strain: float
    strain_safety_factor: float
    length_ratio_limit: float

    @property
    def feasible(self) -> bool:
        """Whether every joint is feasible."""
        return all(j.feasible for j in self.joints.values())

    @property
    def binding_joint(self) -> str:
        """The joint with the highest utilisation: the one limiting this design."""
        return max(self.joints, key=lambda name: self.joints[name].utilisation)

    @property
    def max_utilisation(self) -> float:
        """Utilisation of the binding joint."""
        return self.joints[self.binding_joint].utilisation

    def reasons(self) -> list[str]:
        """One line per infeasible joint."""
        return [
            f"{name}: {sizing.limit_reason}"
            for name, sizing in self.joints.items()
            if sizing.limit_reason
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "feasible": self.feasible,
            "binding_joint": self.binding_joint,
            "max_utilisation": self.max_utilisation,
            "allowable_strain": self.allowable_strain,
            "strain_safety_factor": self.strain_safety_factor,
            "prbm_length_ratio_limit": self.length_ratio_limit,
            "joints": {name: s.to_dict() for name, s in self.joints.items()},
            "reasons": self.reasons(),
        }


@dataclass
class CompliantMechanism:
    """A rigid linkage plus the flexures that replace its joints."""

    base: Linkage
    sizing: dict[str, JointSizing]
    feasibility: FeasibilityReport
    material_name: str
    printer_name: str
    flexure_type: str
    placement: str
    unstressed_at: str
    input_range_deg: tuple[float, float]
    reference_input_deg: float
    provenance: Provenance = field(default_factory=Provenance)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Name of the underlying linkage."""
        return self.base.name

    @property
    def is_physical(self) -> bool:
        """Whether no placeholder inputs were used."""
        return self.provenance.is_physical

    def effective_linkage(self) -> Linkage:
        """Return the linkage whose joints sit at the PRBM characteristic pivots.

        For ``pivot_matched`` placement this is geometrically identical to the
        rigid linkage: that is the whole point of matching. For ``unmatched``
        placement each pivot is displaced by half a flexure length along its host
        link, which changes the effective link lengths and shifts the coupler
        path *before any physics is involved*.

        Simulating this linkage rigidly and comparing with the original rigid
        path isolates the conversion artefact from the simulation-to-reality gap.
        """
        joints = dict(self.base.joints)
        for name, sized in self.sizing.items():
            if sized.pivot_offset_mm == 0.0:
                continue
            joint = self.base.joints[name]
            direction = self._host_direction(joint, sized.host_body)
            shifted = joint.position_mm + sized.pivot_offset_mm * direction
            joints[name] = Joint(
                name=joint.name,
                bodies=joint.bodies,
                xy=(float(shifted[0]), float(shifted[1])),
                kind="flexure",
            )
        for name, joint in joints.items():
            if joint.kind != "flexure":
                joints[name] = Joint(joint.name, joint.bodies, joint.xy, kind="flexure")

        effective = Linkage(
            bodies=dict(self.base.bodies),
            joints=joints,
            outputs=dict(self.base.outputs),
            input_joint=self.base.input_joint,
            input_range_deg=self.input_range_deg,
            name=f"{self.base.name}_compliant",
            meta={**self.base.meta, "placement": self.placement},
        )
        effective.validate()
        return effective

    def _host_direction(self, joint: Joint, host_body: str) -> Any:
        """Return the unit vector along the host link, pointing away from this joint."""
        far = [j for j in self.base.joints_of(host_body) if j != joint.name]
        if not far:
            return rotation_matrix(0.0) @ np.array([1.0, 0.0])
        along = self.base.joints[far[0]].position_mm - joint.position_mm
        norm = float(np.linalg.norm(along))
        return along / norm if norm > 0.0 else np.array([1.0, 0.0])

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the conversion."""
        return {
            "name": self.name,
            "flexure_type": self.flexure_type,
            "material": self.material_name,
            "printer": self.printer_name,
            "placement": self.placement,
            "unstressed_at": self.unstressed_at,
            "input_range_deg": list(self.input_range_deg),
            "reference_input_deg": self.reference_input_deg,
            "feasibility": self.feasibility.to_dict(),
            "is_physical": self.is_physical,
            "placeholders_used": list(self.provenance.placeholders_used),
            "diagnostics": self.diagnostics,
        }


@runtime_checkable
class DesignStrategy(Protocol):
    """Turns a rigid linkage into a compliant one."""

    name: str

    def convert(self, linkage: Linkage, **kwargs: Any) -> CompliantMechanism:
        """Return the compliant counterpart of ``linkage``."""
        ...


#: Registry of design strategies (naive, optimiser, surrogate, LLM agent).
STRATEGIES: Registry[DesignStrategy] = Registry("design strategy")
