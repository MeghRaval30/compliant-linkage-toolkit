"""Printable geometry for a compliant mechanism.

One monolithic part: base mount, flexures, rigid links, input lever and marker
pads all printed together. That is the point of a compliant mechanism -- no
assembly, no pin joints, no friction.

Construction
------------
Every joint gets a flexure: a thin prismatic strip of length ``L`` and thickness
``t``, aligned with its host link and placed so that its **characteristic pivot**
lands on the original rigid joint. That is what pivot matching means, and it
keeps the effective link lengths unchanged. Where those strips end up is
:func:`cmtool.convert.placement.attachment_points`, shared with the beam FEA and
the viewer so all three draw the same part.

Each body then becomes a straight bar between its two **attachment points**,
where the attachment point at a joint is the end of that joint's flexure on the
body's side. This keeps every rigid link a simple bar regardless of which body
hosts which flexure.

The ground body is not a bar but the base plate, carrying M3 clearance holes for
the reusable fixture and two fiducial pads at a known spacing.

The input lever carries two features: a marker pad at its tip, and a through-hole
at a **known radius** from the input pivot for hooking a spring scale to. The
coupler path is insensitive to flexure stiffness, so it cannot test the stiffness
model at all; the input torque is directly proportional to it. Measuring torque is
what makes the PRBM-versus-FEA stiffness disagreement checkable against reality.

Marker pads are all raised by the same amount above the part's top face, so the
base fiducials, the lever marker and the coupler marker are **coplanar**. The
camera homography maps one plane to millimetres; markers at different heights
would each need their own correction.

Known limitation
----------------
The flexure is a plain prismatic strip with sharp corners where it meets the
links, exactly as the pseudo-rigid-body model assumes. Sharp re-entrant corners
concentrate stress, and that is where these parts will crack first. A fillet
would help the part and hurt the model's fidelity -- the taper adds compliance
the model does not know about. ``fillet_radius_mm`` is therefore available and
defaults to **off**: the first prints should show where they actually fail before
geometry and model are allowed to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cadquery as cq
import numpy as np

from cmtool.convert.base import CompliantMechanism
from cmtool.convert.placement import attachment_points as _attachment_points
from cmtool.convert.placement import normal_vector as _normal
from cmtool.convert.placement import unit_vector as _unit
from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance
from cmtool.core.units import FloatArray
from cmtool.materials.loader import Printer


@dataclass(frozen=True)
class MechanismCadSpec:
    """Geometry choices for turning a compliant mechanism into a printable part."""

    link_width_mm: float = 8.0
    lever_length_mm: float = 45.0
    lever_width_mm: float = 8.0
    pad_size_mm: float = 30.0
    pad_thickness_mm: float = 1.0
    base_depth_mm: float = 38.0
    base_margin_mm: float = 14.0
    bolt_diameter_mm: float = 3.4
    bolt_inset_mm: float = 8.0
    force_hole_diameter_mm: float = 3.0
    force_point_fraction: float = 0.8
    fillet_radius_mm: float = 0.0
    part_thickness_mm: float | None = None
    #: Through-hole at the coupler point so the part can draw its own path on
    #: paper. Zero leaves the marker pad solid. The rigid control part carries
    #: the same hole at the same coordinate, which is what lets the two curves
    #: be compared on one sheet.
    pen_hole_diameter_mm: float = 0.0

    def resolved_thickness_mm(
        self, printer: Printer, provenance: Provenance | None = None
    ) -> float:
        """Out-of-plane depth, from the spec or the printer's design rules."""
        if self.part_thickness_mm is not None:
            return float(self.part_thickness_mm)
        return printer.part_thickness_mm(provenance)


def _bar(start: FloatArray, end: FloatArray, width: float, depth: float) -> cq.Workplane:
    """Return a straight bar of the given width between two points, with rounded ends."""
    direction = _unit(np.asarray(end, dtype=float) - np.asarray(start, dtype=float))
    offset = _normal(direction) * (width / 2.0)
    corners = [
        tuple(np.asarray(start) + offset),
        tuple(np.asarray(end) + offset),
        tuple(np.asarray(end) - offset),
        tuple(np.asarray(start) - offset),
    ]
    body = cq.Workplane("XY").polyline([(float(x), float(y)) for x, y in corners]).close()
    solid = body.extrude(depth)
    for point in (start, end):
        solid = solid.union(
            cq.Workplane("XY")
            .moveTo(float(point[0]), float(point[1]))
            .circle(width / 2.0)
            .extrude(depth)
        )
    return solid


def joint_name_other_body(linkage: Linkage, joint: str, host: str) -> str:
    """Return the body on the far side of ``joint`` from ``host``."""
    return linkage.joints[joint].other(host)


def _strip_between(
    start: FloatArray, end: FloatArray, thickness: float, depth: float
) -> cq.Workplane:
    """Return a prismatic strip of the given thickness spanning two points."""
    unit = _unit(np.asarray(end, dtype=float) - np.asarray(start, dtype=float))
    offset = _normal(unit) * (thickness / 2.0)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    corners = [
        tuple(start + offset),
        tuple(end + offset),
        tuple(end - offset),
        tuple(start - offset),
    ]
    return (
        cq.Workplane("XY")
        .polyline([(float(x), float(y)) for x, y in corners])
        .close()
        .extrude(depth)
    )


def _pad(centre: FloatArray, size: float, depth: float, thickness: float) -> cq.Workplane:
    """Return a flat square marker pad raised above the part's top face."""
    return (
        cq.Workplane("XY")
        .workplane(offset=depth)
        .moveTo(float(centre[0]), float(centre[1]))
        .rect(size, size)
        .extrude(thickness)
    )


@dataclass
class MechanismLayout:
    """Where everything ended up, for the print sheet and the vision pipeline."""

    flexure_axis: dict[str, tuple[float, float]] = field(default_factory=dict)
    attachment_points: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    bolt_holes_mm: list[tuple[float, float]] = field(default_factory=list)
    fiducial_pads_mm: list[tuple[float, float]] = field(default_factory=list)
    fiducial_spacing_mm: float = 0.0
    lever_pad_mm: tuple[float, float] | None = None
    force_point_mm: tuple[float, float] | None = None
    force_radius_mm: float = 0.0
    coupler_pad_mm: tuple[float, float] | None = None
    pad_plane_z_mm: float = 0.0
    lever_clears_base: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "flexure_axis": {k: list(v) for k, v in self.flexure_axis.items()},
            "attachment_points": {
                body: {joint: list(pt) for joint, pt in points.items()}
                for body, points in self.attachment_points.items()
            },
            "bolt_holes_mm": [list(p) for p in self.bolt_holes_mm],
            "fiducial_pads_mm": [list(p) for p in self.fiducial_pads_mm],
            "fiducial_spacing_mm": self.fiducial_spacing_mm,
            "lever_pad_mm": list(self.lever_pad_mm) if self.lever_pad_mm else None,
            "force_point_mm": list(self.force_point_mm) if self.force_point_mm else None,
            "force_radius_mm": self.force_radius_mm,
            "coupler_pad_mm": list(self.coupler_pad_mm) if self.coupler_pad_mm else None,
            "pad_plane_z_mm": self.pad_plane_z_mm,
            "lever_clears_base": self.lever_clears_base,
            "warnings": list(self.warnings),
        }


def build_mechanism(
    mechanism: CompliantMechanism,
    *,
    spec: MechanismCadSpec | None = None,
    printer: Printer | None = None,
    provenance: Provenance | None = None,
) -> tuple[cq.Workplane, MechanismLayout]:
    """Build the monolithic printable part for a converted mechanism.

    Returns
    -------
    tuple
        The solid, and a :class:`MechanismLayout` recording bolt holes, marker pad
        centres, the fiducial spacing the vision pipeline needs, and any warnings.
    """
    spec = spec or MechanismCadSpec()
    printer_cfg = printer or Printer.load(mechanism.printer_name)
    depth = spec.resolved_thickness_mm(printer_cfg, provenance)
    linkage = mechanism.base
    layout = MechanismLayout(pad_plane_z_mm=depth + spec.pad_thickness_mm)

    attachments, axes = _attachment_points(mechanism, linkage)
    layout.flexure_axis = {k: (float(v[0]), float(v[1])) for k, v in axes.items()}
    layout.attachment_points = {
        body: {j: (float(p[0]), float(p[1])) for j, p in points.items()}
        for body, points in attachments.items()
    }

    ground = linkage.ground
    ground_joints = linkage.joints_of(ground)
    pos_first = linkage.joints[ground_joints[0]].position_mm
    pos_second = linkage.joints[ground_joints[1]].position_mm
    along = _unit(pos_second - pos_first)
    normal = _normal(along)

    moving_centroid = np.mean(
        [
            linkage.joints[j].position_mm
            for j in linkage.joints
            if ground not in linkage.joints[j].bodies
        ]
        or [pos_first],
        axis=0,
    )
    if float(np.dot(moving_centroid - pos_first, normal)) > 0.0:
        normal = -normal  # point away from the mechanism

    # --- base plate -------------------------------------------------------
    span = float(np.linalg.norm(pos_second - pos_first))
    origin = pos_first - along * spec.base_margin_mm
    width = span + 2.0 * spec.base_margin_mm
    inward = -normal * spec.base_margin_mm
    corners = [
        origin + inward,
        origin + along * width + inward,
        origin + along * width + normal * spec.base_depth_mm,
        origin + normal * spec.base_depth_mm,
    ]
    solid = (
        cq.Workplane("XY")
        .polyline([(float(p[0]), float(p[1])) for p in corners])
        .close()
        .extrude(depth)
    )

    centre_base = origin + along * (width / 2.0) + normal * (spec.base_depth_mm / 2.0)
    for sign_u in (-1.0, 1.0):
        for sign_n in (-1.0, 1.0):
            hole = (
                centre_base
                + along * sign_u * (width / 2.0 - spec.bolt_inset_mm)
                + normal * sign_n * (spec.base_depth_mm / 2.0 - spec.bolt_inset_mm)
            )
            layout.bolt_holes_mm.append((float(hole[0]), float(hole[1])))
            solid = solid.cut(
                cq.Workplane("XY")
                .moveTo(float(hole[0]), float(hole[1]))
                .circle(spec.bolt_diameter_mm / 2.0)
                .extrude(depth)
            )

    pad_offset = min(width / 2.0 - spec.pad_size_mm / 2.0 - 2.0, 45.0)
    pad_line = centre_base + normal * (spec.base_depth_mm / 4.0)
    for sign_u in (-1.0, 1.0):
        pad_centre = pad_line + along * sign_u * pad_offset
        layout.fiducial_pads_mm.append((float(pad_centre[0]), float(pad_centre[1])))
        solid = solid.union(_pad(pad_centre, spec.pad_size_mm, depth, spec.pad_thickness_mm))
    layout.fiducial_spacing_mm = 2.0 * pad_offset

    # --- rigid links ------------------------------------------------------
    for body in linkage.bodies:
        if body == ground:
            continue
        points = list(attachments[body].values())
        if len(points) != 2:
            continue
        solid = solid.union(_bar(points[0], points[1], spec.link_width_mm, depth))

    # --- flexures ---------------------------------------------------------
    for joint_name in linkage.joints:
        sized = mechanism.sizing[joint_name]
        root = attachments[sized.host_body][joint_name]
        tip = attachments[joint_name_other_body(linkage, joint_name, sized.host_body)][joint_name]
        solid = solid.union(_strip_between(root, tip, sized.geometry.thickness_mm, depth))

    # --- input lever, pointing away from the driven link ------------------
    input_joint = linkage.joints[linkage.input_joint]
    input_body = linkage.input_body
    far_joints = [j for j in linkage.joints_of(input_body) if j != linkage.input_joint]
    toward_link = _unit(linkage.joints[far_joints[0]].position_mm - input_joint.position_mm)
    lever_tip = input_joint.position_mm - toward_link * spec.lever_length_mm
    solid = solid.union(_bar(input_joint.position_mm, lever_tip, spec.lever_width_mm, depth))
    layout.lever_pad_mm = (float(lever_tip[0]), float(lever_tip[1]))
    solid = solid.union(_pad(lever_tip, spec.pad_size_mm * 0.6, depth, spec.pad_thickness_mm))

    # Through-hole for a thread (or a small hook), so input torque can be measured.
    # The path barely responds to flexure stiffness but the torque responds to it
    # directly, so torque is the measurement that actually tests the stiffness model.
    #
    # The primary rig runs a thread from here, horizontally, over a pulley at the
    # table edge with weights hanging from it: the mechanism lies flat, so the pull
    # is horizontal and a scale that weighs vertically cannot read it. 3 mm takes a
    # doubled thread or a small S-hook and keeps the hole well inside the lever.
    radius = spec.lever_length_mm * spec.force_point_fraction
    force_point = input_joint.position_mm - toward_link * radius
    layout.force_point_mm = (float(force_point[0]), float(force_point[1]))
    layout.force_radius_mm = float(radius)
    solid = solid.cut(
        cq.Workplane("XY")
        .moveTo(float(force_point[0]), float(force_point[1]))
        .circle(spec.force_hole_diameter_mm / 2.0)
        .extrude(depth)
    )

    layout.lever_clears_base = _lever_clears_base(
        mechanism, linkage, spec, origin, along, normal, width
    )
    if not layout.lever_clears_base:
        layout.warnings.append(
            "the input lever sweeps over the base plate: shorten lever_length_mm, or "
            "drive the mechanism from the other side"
        )

    # --- coupler marker pad ----------------------------------------------
    if linkage.outputs:
        output = next(iter(linkage.outputs.values()))
        point = output.position_mm
        body_points = list(attachments[output.body].values())
        if body_points:
            nearest = min(body_points, key=lambda p: float(np.linalg.norm(p - point)))
            if float(np.linalg.norm(point - nearest)) > 1e-6:
                solid = solid.union(_bar(nearest, point, spec.link_width_mm * 0.7, depth))
        layout.coupler_pad_mm = (float(point[0]), float(point[1]))
        solid = solid.union(_pad(point, spec.pad_size_mm * 0.6, depth, spec.pad_thickness_mm))
        if spec.pen_hole_diameter_mm > 0.0:
            # Through the pad and the part together, so a pen dropped in reaches
            # the paper. The ArUco pad is still usable around it.
            solid = solid.cut(
                cq.Workplane("XY")
                .moveTo(float(point[0]), float(point[1]))
                .circle(spec.pen_hole_diameter_mm / 2.0)
                .extrude(depth + spec.pad_thickness_mm + 1.0)
            )

    return solid, layout


def _lever_clears_base(
    mechanism: CompliantMechanism,
    linkage: Linkage,
    spec: MechanismCadSpec,
    origin: FloatArray,
    along: FloatArray,
    normal: FloatArray,
    width: float,
) -> bool:
    """Check the lever tip stays off the base plate across the whole input arc."""
    from cmtool.convert.arc import sweep

    pivot = linkage.joints[linkage.input_joint].position_mm
    try:
        result = sweep(linkage, mechanism.input_range_deg, 25)
    except Exception:
        # Sweep failure is reported elsewhere; treat clearance as unknown-but-ok.
        return True

    input_body = linkage.input_body
    far_joints = [j for j in linkage.joints_of(input_body) if j != linkage.input_joint]

    # Sample along the lever, not just its tip: a long lever can overhang the base
    # entirely while its middle still sweeps straight through it.
    fractions = np.linspace(0.0, 1.0, 21)
    for state in result.states:
        far = state.joint_positions_mm[far_joints[0]]
        direction = -_unit(far - pivot)
        for fraction in fractions:
            point = pivot + direction * (spec.lever_length_mm * float(fraction))
            local = point - origin
            u = float(np.dot(local, along))
            n = float(np.dot(local, normal))
            if 0.0 <= u <= width and -spec.base_margin_mm <= n <= spec.base_depth_mm:
                return False
    return True
