"""Searching for four-bar designs that are actually buildable as compliant mechanisms.

The pilot example inherited from milestone A1 is not buildable: its 18 mm input
link cannot host a flexure long enough to survive even a 9 degree bend while
staying inside the pseudo-rigid-body envelope. The governing bound
(:mod:`cmtool.convert.limits`) says why —

    theta_max = 2 r l eps_allow / (t SF)

— the usable rotation is proportional to the **shorter adjacent link**. So a
compliant four-bar wants links of comparable, substantial length. A classic
crank-rocker with a stubby crank is exactly the wrong shape.

This module samples candidates with that in mind and filters them on:

1. assembly across the whole arc, with no branch flip,
2. transmission angle inside its window,
3. a fitted input arc meeting the joint-excursion target,
4. per-joint flexure feasibility (strain *and* PRBM validity),
5. the swept footprint fitting the print envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cmtool.convert.arc import ArcFitError, fit_input_arc, sweep
from cmtool.convert.base import CompliantMechanism
from cmtool.convert.limits import min_link_length_for_excursion_mm
from cmtool.convert.naive import DEFAULT_STRAIN_SAFETY_FACTOR, NaiveStrategy
from cmtool.core.graph import Linkage, LinkageError
from cmtool.core.provenance import Provenance
from cmtool.core.units import FloatArray
from cmtool.flexures.prbm_models import max_length_fraction_of_link
from cmtool.kinematics.fourbar import (
    AssemblyError,
    identify_four_bar,
    transmission_angle,
)
from cmtool.materials.loader import Material, Printer

#: Transmission angle window, in degrees. Outside it a linkage transmits force badly.
DEFAULT_TRANSMISSION_WINDOW_DEG = (40.0, 140.0)

#: Space the printed part needs around the bare linkage, in mm: base plate depth
#: plus its margin, which is set by the fiducial pad size. Measured from
#: :class:`~cmtool.cad.mechanism.MechanismCadSpec` defaults rather than guessed --
#: the first mechanism CAD came out 205 mm tall against a 180 mm bed because this
#: allowance was too small.
DEFAULT_CAD_OVERHEAD_MM = 60.0


@dataclass
class Candidate:
    """One design that passed (or failed) the filters."""

    linkage: Linkage
    compliant: CompliantMechanism | None = None
    arc_deg: tuple[float, float] | None = None
    excursions_deg: dict[str, float] = field(default_factory=dict)
    footprint_mm: tuple[float, float] | None = None
    transmission_range_deg: tuple[float, float] | None = None
    rejected: str | None = None
    rejected_code: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this candidate passed every filter."""
        return self.rejected is None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        out: dict[str, Any] = {
            "name": self.linkage.name,
            "ok": self.ok,
            "rejected": self.rejected,
            "rejected_code": self.rejected_code,
            "arc_deg": list(self.arc_deg) if self.arc_deg else None,
            "excursions_deg": {k: round(v, 3) for k, v in self.excursions_deg.items()},
            "footprint_mm": list(self.footprint_mm) if self.footprint_mm else None,
            "transmission_range_deg": (
                [round(v, 2) for v in self.transmission_range_deg]
                if self.transmission_range_deg
                else None
            ),
        }
        if self.compliant is not None:
            out["feasibility"] = self.compliant.feasibility.to_dict()
        return out


def swept_footprint_mm(
    linkage: Linkage, arc_deg: tuple[float, float], n_steps: int = 41
) -> tuple[float, float]:
    """Bounding-box size of everything the mechanism occupies over its arc.

    Includes joints and tracked output points at every sampled configuration.
    """
    result = sweep(linkage, arc_deg, n_steps)
    points: list[FloatArray] = []
    for state in result.states:
        points.extend(state.joint_positions_mm.values())
        points.extend(state.output_points_mm.values())
    stacked = np.vstack(points)
    lo = np.asarray(stacked.min(axis=0), dtype=float)
    hi = np.asarray(stacked.max(axis=0), dtype=float)
    return (float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def static_footprint_mm(linkage: Linkage) -> tuple[float, float]:
    """Bounding-box size of the linkage in its reference configuration alone.

    A lower bound on the swept footprint, and far cheaper: one configuration
    rather than a whole sweep.
    """
    points = [j.position_mm for j in linkage.joints.values()]
    points += [o.position_mm for o in linkage.outputs.values()]
    stacked = np.vstack(points)
    lo = np.asarray(stacked.min(axis=0), dtype=float)
    hi = np.asarray(stacked.max(axis=0), dtype=float)
    return (float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def reference_transmission_angle_deg(linkage: Linkage) -> float | None:
    """Transmission angle at the reference configuration, or ``None`` if undefined."""
    try:
        roles = identify_four_bar(linkage)
    except LinkageError:
        return None
    return float(
        np.degrees(
            transmission_angle(
                linkage.joints[roles.joint_b].position_mm,
                linkage.joints[roles.joint_c].position_mm,
                linkage.joints[roles.joint_d].position_mm,
            )
        )
    )


def evaluate(
    linkage: Linkage,
    *,
    max_joint_excursion_deg: float = 22.0,
    min_joint_excursion_deg: float = 12.0,
    envelope_mm: tuple[float, float] = (180.0, 180.0),
    envelope_margin_mm: float = DEFAULT_CAD_OVERHEAD_MM,
    transmission_window_deg: tuple[float, float] = DEFAULT_TRANSMISSION_WINDOW_DEG,
    n_steps: int = 41,
    **convert_options: Any,
) -> Candidate:
    """Fit an arc to a linkage, convert it, and apply every filter.

    Parameters
    ----------
    max_joint_excursion_deg
        Target ceiling for the worst joint excursion; the arc is shrunk to meet it.
    min_joint_excursion_deg
        Floor below which the design is rejected as not worth printing: a
        mechanism that barely moves cannot show a measurable sim-to-real gap.
    envelope_mm, envelope_margin_mm
        Print envelope, and the space the printed part needs around the bare
        linkage for its base mount, fiducial pads and input lever.
    """
    candidate = Candidate(linkage=linkage)
    usable = (envelope_mm[0] - envelope_margin_mm, envelope_mm[1] - envelope_margin_mm)

    # Two cheap checks first, both sound: a sweep can only grow the footprint, and
    # the reference pose's transmission angle is part of the swept range. Rejecting
    # here costs one configuration instead of the dozen sweeps an arc fit needs.
    static = static_footprint_mm(linkage)
    if static[0] > usable[0] or static[1] > usable[1]:
        candidate.footprint_mm = static
        candidate.rejected = (
            f"static footprint {static[0]:.0f}x{static[1]:.0f} mm already exceeds the usable "
            f"{usable[0]:.0f}x{usable[1]:.0f} mm envelope"
        )
        candidate.rejected_code = "footprint"
        return candidate

    mu_reference = reference_transmission_angle_deg(linkage)
    if mu_reference is not None and not (
        transmission_window_deg[0] <= mu_reference <= transmission_window_deg[1]
    ):
        candidate.rejected = (
            f"transmission angle {mu_reference:.1f} deg at the reference pose is already "
            f"outside the {transmission_window_deg[0]:.0f}..{transmission_window_deg[1]:.0f} "
            "deg window"
        )
        candidate.rejected_code = "transmission_angle"
        return candidate

    try:
        fit = fit_input_arc(
            linkage, max_joint_excursion_deg=max_joint_excursion_deg, n_steps=n_steps
        )
    except (ArcFitError, AssemblyError, LinkageError) as exc:
        candidate.rejected = f"no usable arc: {exc}"
        candidate.rejected_code = "no_usable_arc"
        return candidate

    candidate.arc_deg = fit.input_range_deg
    candidate.excursions_deg = fit.excursions_deg

    if fit.max_excursion_deg < min_joint_excursion_deg:
        candidate.rejected = (
            f"largest joint excursion {fit.max_excursion_deg:.1f} deg is below the "
            f"{min_joint_excursion_deg:.1f} deg floor"
        )
        candidate.rejected_code = "excursion_too_small"
        return candidate

    result = sweep(linkage, fit.input_range_deg, n_steps)
    if result.diagnostics.get("branch_flip"):
        candidate.rejected = "branch flip across the arc"
        candidate.rejected_code = "branch_flip"
        return candidate

    mu_lo = result.diagnostics.get("transmission_angle_min_deg")
    mu_hi = result.diagnostics.get("transmission_angle_max_deg")
    if mu_lo is None or mu_hi is None:
        candidate.rejected = "transmission angle could not be evaluated"
        candidate.rejected_code = "transmission_angle"
        return candidate

    candidate.transmission_range_deg = (float(mu_lo), float(mu_hi))
    if mu_lo < transmission_window_deg[0] or mu_hi > transmission_window_deg[1]:
        candidate.rejected = (
            f"transmission angle {mu_lo:.1f}..{mu_hi:.1f} deg leaves the "
            f"{transmission_window_deg[0]:.0f}..{transmission_window_deg[1]:.0f} deg window"
        )
        candidate.rejected_code = "transmission_angle"
        return candidate

    footprint = swept_footprint_mm(linkage, fit.input_range_deg, n_steps)
    candidate.footprint_mm = footprint
    if footprint[0] > usable[0] or footprint[1] > usable[1]:
        candidate.rejected = (
            f"swept footprint {footprint[0]:.0f}x{footprint[1]:.0f} mm exceeds the usable "
            f"{usable[0]:.0f}x{usable[1]:.0f} mm envelope"
        )
        candidate.rejected_code = "footprint"
        return candidate

    compliant = NaiveStrategy().convert(
        linkage, input_range_deg=fit.input_range_deg, n_steps=n_steps, **convert_options
    )
    candidate.compliant = compliant
    if not compliant.feasibility.feasible:
        binding = compliant.feasibility.binding_joint
        candidate.rejected = (
            f"joint {binding} infeasible "
            f"(utilisation {compliant.feasibility.max_utilisation:.2f}): "
            f"{compliant.sizing[binding].limit_reason}"
        )
        candidate.rejected_code = f"flexure_infeasible_{binding}"
        return candidate

    return candidate


@dataclass
class SearchReport:
    """Outcome of a design search."""

    kept: list[Candidate]
    reasons: dict[str, int]
    link_floor_mm: float
    n_candidates: int
    seed: int

    @property
    def n_rejected(self) -> int:
        """How many candidates were rejected."""
        return sum(self.reasons.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "link_floor_mm": self.link_floor_mm,
            "n_kept": len(self.kept),
            "n_rejected": self.n_rejected,
            "reasons": dict(self.reasons),
            "kept": [c.summary() for c in self.kept],
        }


def sampling_link_floor_mm(
    max_joint_excursion_deg: float,
    *,
    material: str = "PLA",
    printer: str = "bambu_a1",
    thickness_mm: float | None = None,
    unstressed_at: str = "mid_arc",
    max_length_fraction: float | None = None,
    strain_safety_factor: float = DEFAULT_STRAIN_SAFETY_FACTOR,
    provenance: Provenance | None = None,
) -> float:
    """Shortest link worth sampling at all, from the closed-form design bound.

    A four-bar whose shortest link is below this cannot reach the target joint
    excursion at any flexure length, so sampling one is wasted work. Deriving the
    floor instead of hard-coding a range also means it tracks the material and
    printer numbers as those move from placeholder to measured.
    """
    material_cfg = Material.load(material)
    printer_cfg = Printer.load(printer)
    thickness = (
        float(thickness_mm)
        if thickness_mm is not None
        else printer_cfg.min_flexure_thickness_mm(provenance)
    )
    return min_link_length_for_excursion_mm(
        max_joint_excursion_deg,
        thickness,
        material_cfg.allowable_strain(provenance),
        unstressed_at=unstressed_at,
        max_length_fraction=(
            max_length_fraction
            if max_length_fraction is not None
            else max_length_fraction_of_link(provenance)
        ),
        strain_safety_factor=strain_safety_factor,
    )


def search(
    *,
    n_candidates: int = 400,
    seed: int = 0,
    link_range_mm: tuple[float, float] = (35.0, 85.0),
    coupler_offset_range: tuple[float, float] = (0.25, 0.75),
    coupler_height_range_mm: tuple[float, float] = (10.0, 45.0),
    max_joint_excursion_deg: float = 22.0,
    envelope_mm: tuple[float, float] = (180.0, 180.0),
    envelope_margin_mm: float = DEFAULT_CAD_OVERHEAD_MM,
    keep: int = 3,
    prefer_prbm_valid: bool = True,
    **evaluate_options: Any,
) -> SearchReport:
    """Sample four-bars and return the best feasible ones plus rejection counts.

    Link lengths are sampled from a band of comparable, substantial lengths,
    because the governing bound makes short links useless for compliant
    conversion. The lower end of that band is not a magic number: it is raised to
    :func:`sampling_link_floor_mm`, the shortest link that could reach the target
    excursion at all. Sampling below it only generates candidates that the
    feasibility check would reject after a dozen wasted sweeps.

    Returns
    -------
    SearchReport
        Kept candidates ranked by how much motion they deliver per unit of
        flexure utilisation, the tally of why the others were rejected, and the
        link-length floor that was applied.
    """
    rng = np.random.default_rng(seed)
    kept: list[Candidate] = []
    reasons: dict[str, int] = {}

    floor = sampling_link_floor_mm(
        max_joint_excursion_deg,
        material=evaluate_options.get("material", "PLA"),
        printer=evaluate_options.get("printer", "bambu_a1"),
        thickness_mm=evaluate_options.get("thickness_mm"),
        unstressed_at=evaluate_options.get("unstressed_at", "mid_arc"),
    )
    link_range_mm = (max(link_range_mm[0], floor), max(link_range_mm[1], floor * 1.05))

    for index in range(n_candidates):
        ground, crank, coupler, output = rng.uniform(*link_range_mm, size=4)
        offset = rng.uniform(*coupler_offset_range)
        height = rng.uniform(*coupler_height_range_mm)
        start_deg = float(rng.uniform(45.0, 135.0))

        try:
            linkage = Linkage.four_bar(
                ground_mm=float(ground),
                input_mm=float(crank),
                coupler_mm=float(coupler),
                output_mm=float(output),
                input_angle_deg=start_deg,
                input_range_deg=(start_deg - 1.0, start_deg + 1.0),
                name=f"fb_{seed:02d}_{index:04d}",
            )
        except (LinkageError, AssemblyError):
            reasons["does_not_assemble"] = reasons.get("does_not_assemble", 0) + 1
            continue

        linkage = _place_coupler_point(linkage, offset, height)

        candidate = evaluate(
            linkage,
            max_joint_excursion_deg=max_joint_excursion_deg,
            envelope_mm=envelope_mm,
            envelope_margin_mm=envelope_margin_mm,
            **evaluate_options,
        )
        if candidate.ok:
            kept.append(candidate)
        else:
            key = candidate.rejected_code or "unknown"
            reasons[key] = reasons.get(key, 0) + 1

    kept.sort(key=lambda c: _design_score(c, prefer_prbm_valid=prefer_prbm_valid), reverse=True)
    return SearchReport(
        kept=kept[:keep],
        reasons=reasons,
        link_floor_mm=float(link_range_mm[0]),
        n_candidates=n_candidates,
        seed=seed,
    )


def _design_score(candidate: Candidate, *, prefer_prbm_valid: bool = True) -> float:
    """Rank by delivered motion per unit of flexure utilisation.

    Prefers designs that move a lot while leaving margin in their flexures,
    because those are the ones most likely to survive a real print and still show
    a measurable path.

    ``prefer_prbm_valid`` adds a bonus for designs whose flexures all sit inside
    the small-length regime. It is a **preference, not a filter**: Phase A wants
    simple, well-understood pilot prints, while the dataset needs the harder cases
    where the simple model fails. Ranking gets the first without excluding the
    second.
    """
    if candidate.compliant is None:
        return -np.inf
    excursion = max(candidate.excursions_deg.values()) if candidate.excursions_deg else 0.0
    utilisation = max(candidate.compliant.feasibility.max_utilisation, 1e-6)
    score = excursion / utilisation
    if prefer_prbm_valid and candidate.compliant.feasibility.all_small_length:
        score *= 1.5
    return float(score)


def _place_coupler_point(linkage: Linkage, offset: float, height_mm: float) -> Linkage:
    """Move the tracked coupler point to a parameterised spot on the coupler link."""
    from cmtool.core.graph import OutputPoint

    pos_b = linkage.joints["B"].position_mm
    pos_c = linkage.joints["C"].position_mm
    along = pos_c - pos_b
    length = float(np.linalg.norm(along))
    unit = along / length
    normal = np.array([-unit[1], unit[0]])
    point = pos_b + offset * length * unit + height_mm * normal
    linkage.outputs["P"] = OutputPoint("P", "coupler", (float(point[0]), float(point[1])))
    return linkage
