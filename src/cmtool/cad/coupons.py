"""Test coupons: the first things printed, before any mechanism.

Two unknowns currently block every feasibility verdict in the toolkit, and both
are placeholders in config:

``min_flexure_thickness_mm``
    How thin a flexure this printer can actually produce, consistently enough to
    measure. Guessing it wrong in either direction is costly: too thick and no
    design meets its joint-excursion target, too thin and parts snap.

``youngs_modulus_MPa`` and ``allowable_strain``
    The material properties that set stiffness and the bend limit.

The flexure coupon answers the first, the cantilever coupon the second. Printing
them before any mechanism means the mechanism designs are sized with measured
numbers rather than guesses.

Geometry conventions
--------------------
Everything is built in the XY plane and extruded along +Z by the part thickness,
matching how the parts are printed: mechanism plane flat on the bed, flexures
bending in-plane, layers stacked out-of-plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import cadquery as cq

#: Thicknesses on the flexure coupon, in mm. Spans from one nozzle width (where
#: a single extrusion is the whole wall) to well above it.
DEFAULT_FLEXURE_THICKNESSES_MM = (0.4, 0.5, 0.6, 0.8, 1.0)

#: Cantilever strip thicknesses, in mm. Two values deliberately: a thin strip is
#: almost all perimeter and a thicker one has infill, so comparing the two shows
#: whether the apparent modulus depends on wall structure. If it does, the
#: modulus measured here does not transfer to a 0.5 mm flexure unchanged, and
#: that is something to know before trusting any simulation.
DEFAULT_CANTILEVER_THICKNESSES_MM = (1.0, 2.0)


@dataclass(frozen=True)
class FlexureCouponSpec:
    """Plate of flexure strips of graded thickness, sharing one rigid rail."""

    thicknesses_mm: tuple[float, ...] = DEFAULT_FLEXURE_THICKNESSES_MM
    flexure_length_mm: float = 10.0
    part_thickness_mm: float = 6.0
    rail_height_mm: float = 12.0
    paddle_width_mm: float = 22.0
    paddle_height_mm: float = 18.0
    pitch_mm: float = 30.0
    pip_size_mm: float = 2.0

    @property
    def total_width_mm(self) -> float:
        """Overall X extent of the plate."""
        return self.pitch_mm * len(self.thicknesses_mm)

    @property
    def total_height_mm(self) -> float:
        """Overall Y extent of the plate."""
        return self.rail_height_mm + self.flexure_length_mm + self.paddle_height_mm

    def identification(self) -> dict[float, int]:
        """Map each thickness to its pip count (1 = leftmost, thinnest).

        Pips are through-holes rather than embossed text: no font dependency, and
        they stay legible at the size a 0.4 mm nozzle can resolve.
        """
        return {t: i + 1 for i, t in enumerate(self.thicknesses_mm)}


@dataclass(frozen=True)
class CantileverCouponSpec:
    """Plain rectangular strips for a cantilever deflection modulus test."""

    thicknesses_mm: tuple[float, ...] = DEFAULT_CANTILEVER_THICKNESSES_MM
    repeats: int = 3
    length_mm: float = 100.0
    width_mm: float = 12.0
    gap_mm: float = 6.0

    @property
    def strip_count(self) -> int:
        """Total number of strips on the plate."""
        return len(self.thicknesses_mm) * self.repeats

    @property
    def total_width_mm(self) -> float:
        """Overall X extent of the arrangement."""
        return self.strip_count * self.width_mm + (self.strip_count - 1) * self.gap_mm

    def layout(self) -> list[tuple[float, float, int]]:
        """Return ``(x_centre, thickness, repeat_index)`` for each strip."""
        out: list[tuple[float, float, int]] = []
        index = 0
        for thickness in self.thicknesses_mm:
            for repeat in range(self.repeats):
                x = index * (self.width_mm + self.gap_mm) + self.width_mm / 2.0
                out.append((x, thickness, repeat))
                index += 1
        return out


def build_flexure_coupon(spec: FlexureCouponSpec | None = None) -> cq.Workplane:
    """Build the flexure thickness coupon.

    Each strip hangs between a shared rigid rail and its own paddle. The strip is
    open on both faces so a caliper can reach it, which is the whole point: the
    measurement wanted is *as-printed* thickness against nominal.
    """
    spec = spec or FlexureCouponSpec()
    depth = spec.part_thickness_mm

    result = (
        cq.Workplane("XY")
        .moveTo(spec.total_width_mm / 2.0, spec.rail_height_mm / 2.0)
        .rect(spec.total_width_mm, spec.rail_height_mm)
        .extrude(depth)
    )

    pips = spec.identification()
    for index, thickness in enumerate(spec.thicknesses_mm):
        x = spec.pitch_mm * (index + 0.5)

        strip = (
            cq.Workplane("XY")
            .moveTo(x, spec.rail_height_mm + spec.flexure_length_mm / 2.0)
            .rect(thickness, spec.flexure_length_mm)
            .extrude(depth)
        )
        paddle_y = spec.rail_height_mm + spec.flexure_length_mm + spec.paddle_height_mm / 2.0
        paddle = (
            cq.Workplane("XY")
            .moveTo(x, paddle_y)
            .rect(spec.paddle_width_mm, spec.paddle_height_mm)
            .extrude(depth)
        )
        result = result.union(strip).union(paddle)

        count = pips[thickness]
        first = x - (count - 1) * spec.pip_size_mm * 1.5 / 2.0
        for pip in range(count):
            hole = (
                cq.Workplane("XY")
                .moveTo(first + pip * spec.pip_size_mm * 1.5, paddle_y)
                .rect(spec.pip_size_mm, spec.pip_size_mm)
                .extrude(depth)
            )
            result = result.cut(hole)

    return result


def build_cantilever_coupon(spec: CantileverCouponSpec | None = None) -> cq.Workplane:
    """Build the cantilever modulus strips as one multi-solid plate.

    The strips are separate bodies arranged on a single build plate, so they
    print in one job under identical conditions -- which matters, because the
    whole point is to compare them.
    """
    spec = spec or CantileverCouponSpec()
    strips = [
        cast(
            "cq.Shape",
            cq.Workplane("XY")
            .moveTo(x, spec.length_mm / 2.0)
            .rect(spec.width_mm, spec.length_mm)
            .extrude(thickness)
            .val(),
        )
        for x, thickness, _repeat in spec.layout()
    ]
    # A compound, not a union: the strips are separate parts that happen to share
    # a build plate, and must stay separate solids through STEP and STL export.
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(strips)])


def coupon_metadata(flexure: FlexureCouponSpec, cantilever: CantileverCouponSpec) -> dict[str, Any]:
    """Return a JSON-serialisable description of both coupons."""
    return {
        "flexure_coupon": {
            "thicknesses_mm": list(flexure.thicknesses_mm),
            "flexure_length_mm": flexure.flexure_length_mm,
            "part_thickness_mm": flexure.part_thickness_mm,
            "size_mm": [flexure.total_width_mm, flexure.total_height_mm],
            "identification": {
                f"{t} mm": f"{n} pip(s)" for t, n in flexure.identification().items()
            },
        },
        "cantilever_coupon": {
            "thicknesses_mm": list(cantilever.thicknesses_mm),
            "repeats": cantilever.repeats,
            "length_mm": cantilever.length_mm,
            "width_mm": cantilever.width_mm,
            "strip_count": cantilever.strip_count,
            "size_mm": [cantilever.total_width_mm, cantilever.length_mm],
        },
    }


@dataclass
class CouponSet:
    """Both coupons plus their metadata, ready to export."""

    flexure_spec: FlexureCouponSpec = field(default_factory=FlexureCouponSpec)
    cantilever_spec: CantileverCouponSpec = field(default_factory=CantileverCouponSpec)

    def build(self) -> dict[str, cq.Workplane]:
        """Build both coupons, keyed by output file stem."""
        return {
            "flexure_coupon": build_flexure_coupon(self.flexure_spec),
            "cantilever_coupon": build_cantilever_coupon(self.cantilever_spec),
        }

    def metadata(self) -> dict[str, Any]:
        """Return the combined metadata."""
        return coupon_metadata(self.flexure_spec, self.cantilever_spec)
