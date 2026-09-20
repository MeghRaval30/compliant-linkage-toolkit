"""Naive rigid-to-compliant replacement, with pivot-matched placement by default.

Every joint becomes a flexure of the same type, sized by the smallest length that
survives the required bend. There is no optimisation here -- that is a later
strategy. This one exists to be the honest baseline, and to get the two
placement variants on the table.

Two decisions carry most of the physics:

**Pivot-matched placement.** A small-length flexural pivot's characteristic pivot
sits at the *centre* of the flexure. Dropping flexures in without accounting for
that moves every effective link length by about half a flexure length, which
displaces the coupler path before any physics is involved. The default therefore
centres each flexure on its original joint. ``placement="unmatched"`` keeps the
offset deliberately, so the size of that artefact can be measured rather than
guessed at.

**Unstressed at mid-arc.** The mechanism is printed in one configuration and that
configuration is unstressed. Printing it at the *middle* of the input arc halves
the largest bend any flexure sees, compared with printing it at one end -- which
halves peak strain, and roughly halves the flexure length needed. This is free
and it is the default; ``unstressed_at="start"`` is available for comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cmtool.convert.arc import sweep
from cmtool.convert.base import (
    STRATEGIES,
    CompliantMechanism,
    FeasibilityReport,
    JointSizing,
)
from cmtool.core.config import config_hash
from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance, git_commit
from cmtool.core.units import wrap_to_pi
from cmtool.flexures.base import FLEXURES, FlexureGeometry
from cmtool.flexures.prbm_models import max_length_fraction_of_link, select_model
from cmtool.flexures.slfp import prbm_validity
from cmtool.materials.loader import Material, Printer

#: Margin applied on top of the strain-limited minimum flexure length.
#: A design choice, not a measurement: the allowable strain is itself uncertain,
#: and printed flexures have surface defects that a nominal geometry does not.
DEFAULT_STRAIN_SAFETY_FACTOR = 1.5

#: Shortest flexure worth printing, in mm. Below roughly this, a "flexure" is a
#: few extrusion widths long and neither prints nor models predictably.
DEFAULT_MIN_FLEXURE_LENGTH_MM = 1.0


class NaiveStrategy:
    """Replace every joint with the same flexure type, sized by strain."""

    name = "naive"

    def convert(
        self,
        linkage: Linkage,
        *,
        flexure_type: str = "small_length_pivot",
        material: str = "PLA",
        printer: str = "kobra2_neo",
        placement: str = "pivot_matched",
        unstressed_at: str = "mid_arc",
        input_range_deg: tuple[float, float] | None = None,
        n_steps: int = 61,
        thickness_mm: float | None = None,
        flexure_length_mm: float | dict[str, float] | None = None,
        strain_safety_factor: float = DEFAULT_STRAIN_SAFETY_FACTOR,
        max_length_fraction: float | None = None,
        min_flexure_length_mm: float = DEFAULT_MIN_FLEXURE_LENGTH_MM,
        **_: Any,
    ) -> CompliantMechanism:
        """Convert ``linkage`` into a compliant mechanism.

        Parameters
        ----------
        flexure_type
            Registered flexure type for every joint.
        material, printer
            Config names. All physical numbers come from those files.
        placement
            ``"pivot_matched"`` (default) or ``"unmatched"``.
        unstressed_at
            ``"mid_arc"`` (default) or ``"start"``: which configuration is
            printed, and therefore unstressed.
        input_range_deg
            Arc to design for. Falls back to the linkage's own arc.
        thickness_mm
            Flexure thickness. Defaults to the printer's minimum printable
            thickness, which is currently a placeholder on every printer.
        flexure_length_mm
            Fixed length for every joint, or a per-joint mapping. When omitted,
            each flexure is sized to the shortest length that survives its bend.
        strain_safety_factor, min_flexure_length_mm
            Design choices; see the module constants.
        max_length_fraction
            Hard geometric cap on flexure length as a fraction of the shorter
            adjacent link. Defaults to the value in ``configs/models/prbm.yaml``.

        Notes
        -----
        Feasibility is decided by **strain and geometry only**. The PRBM length
        ratio is recorded on every joint and selects which model represents it,
        but it never rejects a design.
        """
        if placement not in {"pivot_matched", "unmatched"}:
            raise ValueError(f"unknown placement {placement!r}")
        if unstressed_at not in {"mid_arc", "start"}:
            raise ValueError(f"unknown unstressed_at {unstressed_at!r}")
        if n_steps < 3:
            raise ValueError("n_steps must be at least 3")

        arc = input_range_deg or linkage.input_range_deg
        if arc is None:
            raise ValueError(
                f"linkage {linkage.name!r} has no input arc; pass input_range_deg "
                "(a compliant mechanism is always driven over a limited arc)"
            )

        flexure = FLEXURES.get(flexure_type)
        material_cfg = Material.load(material)
        printer_cfg = Printer.load(printer)

        provenance = Provenance(
            config_hash=config_hash(material_cfg.raw, printer_cfg.raw),
            code_commit=git_commit(),
            notes={
                "strategy": self.name,
                "placement": placement,
                "unstressed_at": unstressed_at,
                "strain_safety_factor": strain_safety_factor,
            },
        )

        length_fraction = (
            float(max_length_fraction)
            if max_length_fraction is not None
            else max_length_fraction_of_link(provenance)
        )

        # Odd sample count so "the middle of the arc" is an actual sample.
        steps = n_steps if n_steps % 2 == 1 else n_steps + 1
        result = sweep(linkage, (float(arc[0]), float(arc[1])), steps)
        reference_index = steps // 2 if unstressed_at == "mid_arc" else 0

        thickness = (
            float(thickness_mm)
            if thickness_mm is not None
            else printer_cfg.min_flexure_thickness_mm(provenance)
        )
        width = printer_cfg.part_thickness_mm(provenance)
        allowable_strain = material_cfg.allowable_strain(provenance)
        modulus = material_cfg.youngs_modulus_mpa(provenance)

        sizing: dict[str, JointSizing] = {}
        for joint_name in linkage.joints:
            rotations = _joint_rotations_rad(result, joint_name, reference_index)
            max_bend = float(np.max(np.abs(rotations)))
            excursion = float(np.degrees(np.ptp(rotations)))

            adjacent = _shortest_adjacent_link_mm(linkage, joint_name)
            min_length_strain = max(
                flexure.min_length_mm(thickness, max_bend, allowable_strain) * strain_safety_factor,
                min_flexure_length_mm,
            )
            max_length_geometric = length_fraction * adjacent

            length = _chosen_length(flexure_length_mm, joint_name, min_length_strain)
            geometry = FlexureGeometry(thickness_mm=thickness, length_mm=length, width_mm=width)
            host = _host_body(linkage, joint_name)

            # The length ratio selects the model; it does not gate the design.
            model = select_model(length / adjacent, provenance=provenance)
            pivot_fraction = flexure.characteristic_pivot_fraction(
                model=model, provenance=provenance
            )

            sizing[joint_name] = JointSizing(
                joint=joint_name,
                flexure_type=flexure_type,
                geometry=geometry,
                max_bend_rad=max_bend,
                excursion_deg=excursion,
                shortest_adjacent_link_mm=adjacent,
                min_length_strain_mm=min_length_strain,
                max_length_geometric_mm=max_length_geometric,
                strain=flexure.peak_strain(geometry, max_bend),
                validity=prbm_validity(
                    geometry, adjacent, max_bend, model=model, provenance=provenance
                ),
                stiffness_nmm_per_rad=flexure.stiffness_nmm_per_rad(
                    geometry, modulus, model=model, provenance=provenance
                ),
                host_body=host,
                pivot_offset_mm=(0.0 if placement == "pivot_matched" else pivot_fraction * length),
                pivot_fraction=pivot_fraction,
            )

        report = FeasibilityReport(
            joints=sizing,
            allowable_strain=allowable_strain,
            strain_safety_factor=strain_safety_factor,
            max_length_fraction=length_fraction,
        )

        return CompliantMechanism(
            base=linkage,
            sizing=sizing,
            feasibility=report,
            material_name=material_cfg.name,
            printer_name=printer_cfg.name,
            flexure_type=flexure_type,
            placement=placement,
            unstressed_at=unstressed_at,
            input_range_deg=(float(arc[0]), float(arc[1])),
            reference_input_deg=float(result.input_angles_deg[reference_index]),
            provenance=provenance,
            diagnostics={
                "n_steps": steps,
                "reference_index": reference_index,
                "thickness_mm": thickness,
                "width_mm": width,
                "youngs_modulus_MPa": modulus,
                "grashof": result.diagnostics.get("grashof", {}).get("classification"),
                "transmission_angle_min_deg": result.diagnostics.get("transmission_angle_min_deg"),
                "branch_flip": result.diagnostics.get("branch_flip"),
            },
        )


def _joint_rotations_rad(result: Any, joint: str, reference_index: int) -> np.ndarray:
    """Relative rotation at ``joint`` measured from a chosen reference state."""
    first, second = result.linkage.joints[joint].bodies
    rel = np.array(
        [s.body_angles_rad[first] - s.body_angles_rad[second] for s in result.states],
        dtype=float,
    )
    return np.asarray(wrap_to_pi(rel - rel[reference_index]), dtype=float)


def _shortest_adjacent_link_mm(linkage: Linkage, joint: str) -> float:
    """Length of the shorter of the two links meeting at ``joint``.

    "Small-length" is relative to what the flexure connects, so the shorter of
    the two is the one that decides.
    """
    lengths = [
        linkage.link_length(body)
        for body in linkage.joints[joint].bodies
        if len(linkage.joints_of(body)) >= 2
    ]
    if not lengths:
        raise ValueError(f"joint {joint!r} has no adjacent link with a defined length")
    return float(min(lengths))


def _host_body(linkage: Linkage, joint: str) -> str:
    """Pick the link the flexure lies along.

    Prefer a moving body over ground, then the shorter link. Deterministic, so
    that a design converts identically on every run.
    """
    bodies = list(linkage.joints[joint].bodies)
    moving = [b for b in bodies if not linkage.bodies[b].is_ground]
    candidates = moving or bodies
    return min(candidates, key=lambda b: (linkage.link_length(b), b))


def _chosen_length(requested: float | dict[str, float] | None, joint: str, sized: float) -> float:
    """Return the flexure length to use at a joint."""
    if requested is None:
        return sized
    if isinstance(requested, dict):
        return float(requested.get(joint, sized))
    return float(requested)


STRATEGIES.add("naive", NaiveStrategy())
