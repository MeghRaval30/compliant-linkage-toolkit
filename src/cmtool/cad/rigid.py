"""The rigid half of the demo pair: a pin-jointed four-bar you can print.

The compliant part in :mod:`cmtool.cad.mechanism` is the thing this project is
about. This is its control: the *same* four-bar, the same link lengths, the same
coupler point, the same base footprint and mounting holes -- built the way it
would have been built before compliant mechanisms, with pin joints.

Put the two side by side and the argument makes itself: one part with four
moving joints and no assembly, against one with pins, clearances, friction and
backlash. Give both a pen hole at the coupler point and they draw their own
answer on the same sheet of paper.

Two joint styles
----------------
``print_in_place``
    Pins printed already assembled. The mechanism is built in **three levels**
    stacked in Z -- base plate, then the input and output links, then the coupler
    -- with a running clearance between each. Every pin rises from the level
    below it, passes through a hole in the level above, and is capped. Nothing is
    assembled and nothing can fall out.

    The clearance is the whole game. Too tight and the joint fuses on the first
    layer of the mating part; too loose and the backlash swamps the path error
    the demo exists to show. ``clearance_mm`` defaults to 0.35 mm radial for a
    0.4 mm nozzle at 0.2 mm layers, and it is a **design choice, not a
    measurement** -- the first print is what says whether it frees off.

``bolt``
    The fallback for when print-in-place does not free off: the same links with
    plain M3 clearance holes, printed flat as separate bodies, assembled with
    bolts. Slower to build and it needs hardware, but it cannot fail to print.

What this part is *not* for
---------------------------
It carries no marker pads and no torque lever. It is a demonstration piece and a
control, not an instrumented specimen: the measured path in this project comes
from the compliant part. Giving this one pads would invite the comparison to be
read as a measurement of pin-joint accuracy, which is a different experiment
needing its own error budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cadquery as cq
import numpy as np

from cmtool.cad.mechanism import _bar
from cmtool.convert.placement import normal_vector as _normal
from cmtool.convert.placement import unit_vector as _unit
from cmtool.core.graph import Linkage
from cmtool.core.units import FloatArray

#: Radial pin clearance for a 0.4 mm nozzle at 0.2 mm layers. A design choice
#: carried over from common print-in-place practice, not something we measured.
DEFAULT_CLEARANCE_MM = 0.35


@dataclass(frozen=True)
class RigidCadSpec:
    """Geometry choices for the pin-jointed part.

    Attributes
    ----------
    clearance_mm
        Radial gap between a pin and its hole. Also used as the vertical gap
        between stacked levels, where it is generous: one or two layers is
        enough there, and more only adds wobble out of plane.
    pen_hole_diameter_mm
        Through-hole at the coupler point. 5 mm takes a fineliner or a gel pen
        barrel; the pen rests on the paper through it and draws the coupler path.
    """

    joint_style: str = "print_in_place"
    clearance_mm: float = DEFAULT_CLEARANCE_MM
    link_width_mm: float = 10.0
    link_thickness_mm: float = 3.0
    base_thickness_mm: float = 3.0
    pin_diameter_mm: float = 5.0
    cap_overhang_mm: float = 1.3
    cap_thickness_mm: float = 1.2
    pen_hole_diameter_mm: float = 5.0
    base_depth_mm: float = 20.0
    base_margin_mm: float = 10.0
    bolt_diameter_mm: float = 3.4
    bolt_inset_mm: float = 7.0
    #: Gap between separately printed bodies in the ``bolt`` layout.
    part_gap_mm: float = 4.0

    @property
    def vertical_gap_mm(self) -> float:
        """Running clearance between stacked levels."""
        return min(self.clearance_mm, 0.3)

    def validate(self) -> None:
        """Refuse geometry that cannot work, rather than exporting it quietly."""
        if self.joint_style not in {"print_in_place", "bolt"}:
            raise ValueError(
                f"unknown joint_style {self.joint_style!r}; choose 'print_in_place' or 'bolt'"
            )
        if self.clearance_mm <= 0.0:
            raise ValueError("clearance_mm must be positive; a zero gap fuses the joint")
        if self.pen_hole_diameter_mm >= self.link_width_mm:
            raise ValueError(
                f"pen hole {self.pen_hole_diameter_mm} mm does not fit in a "
                f"{self.link_width_mm} mm link"
            )
        hole = self.pin_diameter_mm / 2.0 + self.clearance_mm
        if hole + 1.2 > self.link_width_mm / 2.0:
            raise ValueError(
                f"a {self.pin_diameter_mm} mm pin with {self.clearance_mm} mm clearance "
                f"leaves under 1.2 mm of wall in a {self.link_width_mm} mm link"
            )


@dataclass
class RigidLayout:
    """Where everything ended up, for the print sheet and the comparison sheet."""

    joint_style: str = "print_in_place"
    clearance_mm: float = DEFAULT_CLEARANCE_MM
    level_z_mm: dict[str, float] = field(default_factory=dict)
    pin_centres_mm: dict[str, tuple[float, float]] = field(default_factory=dict)
    bolt_holes_mm: list[tuple[float, float]] = field(default_factory=list)
    pen_hole_mm: tuple[float, float] | None = None
    pen_hole_z_mm: float = 0.0
    n_moving_joints: int = 0
    n_printed_bodies: int = 1
    n_fasteners: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "joint_style": self.joint_style,
            "clearance_mm": self.clearance_mm,
            "level_z_mm": dict(self.level_z_mm),
            "pin_centres_mm": {k: list(v) for k, v in self.pin_centres_mm.items()},
            "bolt_holes_mm": [list(p) for p in self.bolt_holes_mm],
            "pen_hole_mm": list(self.pen_hole_mm) if self.pen_hole_mm else None,
            "pen_hole_z_mm": self.pen_hole_z_mm,
            "n_moving_joints": self.n_moving_joints,
            "n_printed_bodies": self.n_printed_bodies,
            "n_fasteners": self.n_fasteners,
            "warnings": list(self.warnings),
        }


def _disc(centre: FloatArray, radius: float, z: float, height: float) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .workplane(offset=z)
        .moveTo(float(centre[0]), float(centre[1]))
        .circle(radius)
        .extrude(height)
    )


def _flat_bar(
    start: FloatArray, end: FloatArray, width: float, z: float, height: float
) -> cq.Workplane:
    """Return a rounded-end bar at height ``z``."""
    return _bar(start, end, width, height).translate((0.0, 0.0, z))


def _base_plate(
    linkage: Linkage, spec: RigidCadSpec, layout: RigidLayout
) -> tuple[cq.Workplane, FloatArray, FloatArray]:
    """Build the plate the ground pivots rise from, and its in-plane frame.

    Deliberately a strip along the ground link rather than a plate under the
    whole mechanism: the coupler has to overhang it so a pen through the coupler
    hole can reach the paper.
    """
    ground = linkage.ground
    first, second = linkage.joints_of(ground)
    start = linkage.joints[first].position_mm
    end = linkage.joints[second].position_mm
    along = _unit(end - start)
    normal = _normal(along)

    # Point the plate away from the mechanism.
    moving = np.mean(
        [
            linkage.joints[j].position_mm
            for j in linkage.joints
            if ground not in linkage.joints[j].bodies
        ]
        or [start],
        axis=0,
    )
    if float(np.dot(moving - start, normal)) > 0.0:
        normal = -normal

    span = float(np.linalg.norm(end - start))
    width = span + 2.0 * spec.base_margin_mm
    origin = start - along * spec.base_margin_mm

    corners = [
        origin + normal * spec.base_depth_mm,
        origin + along * width + normal * spec.base_depth_mm,
        origin + along * width,
        origin,
    ]
    plate = (
        cq.Workplane("XY")
        .polyline([(float(x), float(y)) for x, y in corners])
        .close()
        .extrude(spec.base_thickness_mm)
    )

    centre = origin + along * (width / 2.0) + normal * (spec.base_depth_mm / 2.0)
    for sign_u in (-1.0, 1.0):
        for sign_n in (-1.0, 1.0):
            hole = (
                centre
                + along * sign_u * (width / 2.0 - spec.bolt_inset_mm)
                + normal * sign_n * (spec.base_depth_mm / 2.0 - spec.bolt_inset_mm)
            )
            layout.bolt_holes_mm.append((float(hole[0]), float(hole[1])))
            plate = plate.cut(_disc(hole, spec.bolt_diameter_mm / 2.0, 0.0, spec.base_thickness_mm))

    return plate, along, normal


def build_rigid_mechanism(
    linkage: Linkage,
    *,
    spec: RigidCadSpec | None = None,
) -> tuple[cq.Workplane, RigidLayout]:
    """Build the printable pin-jointed part for a rigid linkage.

    Parameters
    ----------
    linkage
        The four-bar. Its joint coordinates are taken as-is, so the rigid part
        and the compliant one share the same link lengths and coupler point by
        construction rather than by anyone remembering to match them.

    Returns
    -------
    tuple
        The solid, and a :class:`RigidLayout` recording the pin positions, the
        pen hole, the level heights and the part/fastener counts the comparison
        sheet quotes.
    """
    spec = spec or RigidCadSpec()
    spec.validate()
    layout = RigidLayout(joint_style=spec.joint_style, clearance_mm=spec.clearance_mm)

    if spec.joint_style == "bolt":
        return _build_bolt_variant(linkage, spec, layout)
    return _build_print_in_place(linkage, spec, layout)


def _build_print_in_place(
    linkage: Linkage, spec: RigidCadSpec, layout: RigidLayout
) -> tuple[cq.Workplane, RigidLayout]:
    """Three levels stacked in Z, every pin captured by a cap."""
    ground = linkage.ground
    solid, _along, _normal_vec = _base_plate(linkage, spec, layout)

    gap = spec.vertical_gap_mm
    z_lower = spec.base_thickness_mm + gap
    z_upper = z_lower + spec.link_thickness_mm + gap
    layout.level_z_mm = {
        "base_top": spec.base_thickness_mm,
        "lower_links": z_lower,
        "coupler": z_upper,
        "coupler_top": z_upper + spec.link_thickness_mm,
    }

    pin_radius = spec.pin_diameter_mm / 2.0
    hole_radius = pin_radius + spec.clearance_mm
    cap_radius = hole_radius + spec.cap_overhang_mm

    # Which body sits on which level, and which body carries the pin at each
    # joint. The pin always rises from the lower of the two bodies it connects.
    ground_joints = set(linkage.joints_of(ground))
    lower_bodies = [
        b for b in linkage.bodies if b != ground and set(linkage.joints_of(b)) & ground_joints
    ]
    upper_bodies = [b for b in linkage.bodies if b != ground and b not in lower_bodies]
    if len(lower_bodies) != 2 or len(upper_bodies) != 1:
        raise ValueError(
            "the print-in-place layout expects a four-bar: two links on ground and one "
            f"coupler, got lower={lower_bodies} upper={upper_bodies}"
        )
    coupler = upper_bodies[0]

    # --- lower links, one level up from the base -------------------------
    for body in lower_bodies:
        joints = linkage.joints_of(body)
        ends = [linkage.joints[j].position_mm for j in joints]
        solid = solid.union(
            _flat_bar(ends[0], ends[1], spec.link_width_mm, z_lower, spec.link_thickness_mm)
        )

    # --- the coupler, one level above that -------------------------------
    coupler_joints = linkage.joints_of(coupler)
    coupler_ends = [linkage.joints[j].position_mm for j in coupler_joints]
    solid = solid.union(
        _flat_bar(
            coupler_ends[0], coupler_ends[1], spec.link_width_mm, z_upper, spec.link_thickness_mm
        )
    )

    # The coupler point usually sits off the line between the two joints, so the
    # coupler is a triangle rather than a bar.
    pen_point: FloatArray | None = None
    if linkage.outputs:
        output = next(iter(linkage.outputs.values()))
        pen_point = output.position_mm
        for end in coupler_ends:
            if float(np.linalg.norm(pen_point - end)) > 1e-6:
                solid = solid.union(
                    _flat_bar(end, pen_point, spec.link_width_mm, z_upper, spec.link_thickness_mm)
                )

    # --- pins -------------------------------------------------------------
    for joint_name, joint in linkage.joints.items():
        centre = joint.position_mm
        layout.pin_centres_mm[joint_name] = (float(centre[0]), float(centre[1]))
        on_ground = ground in joint.bodies

        if on_ground:
            # Rises from the base plate, through a lower link.
            root_z = 0.0
            hole_z = z_lower
        else:
            # Rises from a lower link, through the coupler.
            root_z = z_lower
            hole_z = z_upper

        hole_top = hole_z + spec.link_thickness_mm
        # Clear the hole's top face before the cap, or the cap welds the joint shut.
        pin_top = hole_top + gap

        solid = solid.cut(_disc(centre, hole_radius, hole_z - 0.01, spec.link_thickness_mm + 0.02))
        solid = solid.union(_disc(centre, pin_radius, root_z, pin_top - root_z))
        solid = solid.union(_disc(centre, cap_radius, pin_top, spec.cap_thickness_mm))

    layout.n_moving_joints = len(linkage.joints)
    layout.n_printed_bodies = 1
    layout.n_fasteners = 0

    if pen_point is not None:
        solid = solid.cut(
            _disc(
                pen_point,
                spec.pen_hole_diameter_mm / 2.0,
                z_upper - 0.01,
                spec.link_thickness_mm + 0.02,
            )
        )
        layout.pen_hole_mm = (float(pen_point[0]), float(pen_point[1]))
        layout.pen_hole_z_mm = z_upper

    layout.warnings.extend(_clearance_notes(spec))
    layout.warnings.extend(_cap_clearance_notes(linkage, spec, coupler))
    return solid, layout


def _segment_distance(point: FloatArray, start: FloatArray, end: FloatArray) -> float:
    """Shortest distance from a point to a segment, in mm."""
    span = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length_sq = float(span @ span)
    if length_sq <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((np.asarray(point, dtype=float) - start) @ span / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _cap_clearance_notes(linkage: Linkage, spec: RigidCadSpec, coupler: str) -> list[str]:
    """Check the ground pins' caps never meet the coupler, over the whole arc.

    The caps that retain the lower links sit at exactly the coupler's height. If
    the coupler ever swings over a ground pin they fuse into one solid on the
    build plate -- a part that looks right and does not move, discovered only
    after an hour of printing. Worth the sweep to find out beforehand.
    """
    from cmtool.convert.arc import sweep as sweep_linkage

    arc = linkage.input_range_deg
    if arc is None:
        return [
            "no input arc on this linkage, so the coupler-versus-cap clearance was not "
            "checked; set input_range_deg and re-export before printing"
        ]

    ground = linkage.ground
    ground_joints = [j for j in linkage.joints if ground in linkage.joints[j].bodies]
    coupler_joints = linkage.joints_of(coupler)
    output = next(iter(linkage.outputs.values()), None)

    cap_radius = spec.pin_diameter_mm / 2.0 + spec.clearance_mm + spec.cap_overhang_mm
    needed = cap_radius + spec.link_width_mm / 2.0

    try:
        result = sweep_linkage(linkage, arc, 37)
    except Exception:  # pragma: no cover - reported elsewhere
        return ["could not sweep the linkage to check cap clearance"]

    worst = float("inf")
    worst_joint = ""
    for state in result.states:
        ends = [state.joint_positions_mm[j] for j in coupler_joints]
        segments = [(ends[0], ends[1])]
        if output is not None:
            point = state.output_points_mm[output.name]
            segments.extend((end, point) for end in ends)
        for pin in ground_joints:
            centre = linkage.joints[pin].position_mm
            for start, end in segments:
                distance = _segment_distance(centre, start, end)
                if distance < worst:
                    worst, worst_joint = distance, pin

    if worst < needed:
        return [
            f"COLLISION: the coupler passes within {worst:.1f} mm of the pin cap at "
            f"{worst_joint}, which needs {needed:.1f} mm. They print as one solid and the "
            "mechanism will not move. Narrow the links, shrink the cap, or move the "
            "coupler point"
        ]
    return [
        f"coupler-to-cap clearance {worst:.1f} mm at {worst_joint} over the whole arc "
        f"(needs {needed:.1f} mm)"
    ]


def _build_bolt_variant(
    linkage: Linkage, spec: RigidCadSpec, layout: RigidLayout
) -> tuple[cq.Workplane, RigidLayout]:
    """Every link flat on the bed as a separate body, joined by M3 bolts.

    The fallback. It cannot fail to print, because there is nothing to free off.
    """
    ground = linkage.ground
    solid, along, normal = _base_plate(linkage, spec, layout)

    # Ground pivots become plain clearance holes in the base plate.
    for joint_name, joint in linkage.joints.items():
        if ground in joint.bodies:
            centre = joint.position_mm
            layout.pin_centres_mm[joint_name] = (float(centre[0]), float(centre[1]))
            solid = solid.cut(
                _disc(centre, spec.bolt_diameter_mm / 2.0, -0.01, spec.base_thickness_mm + 0.02)
            )

    # The moving links are laid out in a row beside the base plate, all flat, so
    # the whole thing is one plate of parts to lift off and bolt together.
    moving = [b for b in linkage.bodies if b != ground]
    offset = spec.base_depth_mm + spec.part_gap_mm
    for index, body in enumerate(moving):
        joints = linkage.joints_of(body)
        ends = [linkage.joints[j].position_mm for j in joints]
        shift = normal * (offset + index * (spec.link_width_mm + spec.part_gap_mm))
        start, end = ends[0] + shift, ends[1] + shift
        piece = _flat_bar(start, end, spec.link_width_mm, 0.0, spec.link_thickness_mm)

        extra = []
        if linkage.outputs:
            output = next(iter(linkage.outputs.values()))
            if output.body == body:
                pen_point = output.position_mm + shift
                for anchor in (start, end):
                    if float(np.linalg.norm(pen_point - anchor)) > 1e-6:
                        piece = piece.union(
                            _flat_bar(
                                anchor,
                                pen_point,
                                spec.link_width_mm,
                                0.0,
                                spec.link_thickness_mm,
                            )
                        )
                extra.append(pen_point)
                layout.pen_hole_mm = (float(pen_point[0]), float(pen_point[1]))
                layout.pen_hole_z_mm = 0.0

        for hole, diameter in [(start, spec.bolt_diameter_mm), (end, spec.bolt_diameter_mm)] + [
            (p, spec.pen_hole_diameter_mm) for p in extra
        ]:
            piece = piece.cut(_disc(hole, diameter / 2.0, -0.01, spec.link_thickness_mm + 0.02))
        solid = solid.union(piece)

    layout.n_moving_joints = len(linkage.joints)
    layout.n_printed_bodies = 1 + len(moving)
    # One bolt per joint, plus a nut and a washer each: what has to be bought.
    layout.n_fasteners = len(linkage.joints)
    layout.warnings.append(
        "bolt variant: the links print as separate bodies on one plate and need "
        f"{layout.n_fasteners} M3 bolts with nuts, plus washers between rubbing faces"
    )
    _ = along
    return solid, layout


def _clearance_notes(spec: RigidCadSpec) -> list[str]:
    """Say plainly that the clearance is a guess until a part is printed."""
    notes = [
        f"pin clearance {spec.clearance_mm:.2f} mm radial is a DESIGN CHOICE, not a "
        "measurement: if the joints seize, reprint at +0.1 mm; if they rattle, -0.1 mm",
    ]
    if spec.clearance_mm < 0.25:
        notes.append(
            f"{spec.clearance_mm:.2f} mm is tight for a 0.4 mm nozzle; expect to have to "
            "free the joints by hand on the first print"
        )
    if spec.clearance_mm > 0.5:
        notes.append(
            f"{spec.clearance_mm:.2f} mm is loose: backlash at the pins may be comparable "
            "to the path differences this pair exists to show"
        )
    return notes


def estimate_print(solid: cq.Workplane, *, density_kg_per_m3: float) -> dict[str, float]:
    """Rough print time and mass from the solid's volume.

    **An estimate, not a measurement, and it is labelled as one wherever it is
    printed.** It assumes the whole part is solid at the given density and a
    single effective deposition rate, so it ignores infill, perimeters, travel
    and the first-layer slowdown. The slicer's own number supersedes it the
    moment there is one; this exists so a print sheet can say roughly how long
    to expect before anyone opens OrcaSlicer.
    """
    # Sum over every shape, not ``val()``: a print-in-place part is deliberately
    # several disjoint solids -- the levels do not touch -- so the first one is a
    # fraction of the part.
    shapes = [item for item in solid.vals() if isinstance(item, cq.Shape)]
    if not shapes:
        raise ValueError("shape has no geometry to measure")
    volume_mm3 = float(sum(item.Volume() for item in shapes))
    #: Effective deposited volume per second, averaged over a whole print on a
    #: bed-slinger at 0.2 mm layers. A planning figure only.
    rate_mm3_per_s = 6.0
    return {
        "volume_mm3": volume_mm3,
        "mass_g": volume_mm3 * density_kg_per_m3 * 1e-6,
        "estimated_minutes": volume_mm3 / rate_mm3_per_s / 60.0,
        "assumed_rate_mm3_per_s": rate_mm3_per_s,
    }
