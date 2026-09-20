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


#: Mandrel radii, in mm. Chosen so a 0.4-0.6 mm strip spans roughly 0.008 to 0.06
#: strain: from comfortably safe to certainly failing.
DEFAULT_MANDREL_RADII_MM = (5.0, 7.5, 10.0, 15.0, 20.0, 25.0)

#: Strip thicknesses for the strain test, in mm.
DEFAULT_STRAIN_STRIP_THICKNESSES_MM = (0.4, 0.5, 0.6)


def bend_strain(thickness_mm: float, radius_mm: float) -> float:
    """Peak surface strain of a strip of thickness ``t`` wrapped on radius ``R``.

    The inner face sits at ``R`` and the neutral axis at ``R + t/2``, so the outer
    fibre strain is

    ``eps = (t / 2) / (R + t / 2)``

    The familiar ``t / (2R)`` is the thin-strip limit of this, and it overstates
    the strain by about 6 percent at the tightest radius here -- enough to matter
    when the number being measured *is* the strain limit.
    """
    if thickness_mm <= 0.0 or radius_mm <= 0.0:
        raise ValueError("thickness and radius must be positive")
    return (thickness_mm / 2.0) / (radius_mm + thickness_mm / 2.0)


@dataclass(frozen=True)
class StrainCouponSpec:
    """Mandrels and strips for measuring the allowable strain.

    Why the geometry is what it is
    ------------------------------
    A flexure bends **in-plane**: its thickness lies in XY and its width runs
    out-of-plane along Z, so bending puts stress along the extrusion direction,
    not across layer boundaries.

    A test strip must bend the same way or it measures the wrong thing. Printed
    flat and bent over a horizontal bar, a strip is loaded across its layers and
    what gets measured is interlayer adhesion -- a different, usually much lower,
    failure strain. So the strips here are printed as thin upright walls, exactly
    like a flexure, and the mandrels are **vertical posts** they wrap around in
    the plane of the bed.
    """

    mandrel_radii_mm: tuple[float, ...] = DEFAULT_MANDREL_RADII_MM
    strip_thicknesses_mm: tuple[float, ...] = DEFAULT_STRAIN_STRIP_THICKNESSES_MM
    repeats: int = 2
    strip_length_mm: float = 110.0
    strip_height_mm: float = 6.0
    tab_length_mm: float = 20.0
    tab_width_mm: float = 10.0
    post_height_mm: float = 10.0
    base_thickness_mm: float = 4.0
    gap_mm: float = 10.0
    strip_pitch_mm: float = 14.0

    @property
    def rows(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Split the mandrels into two rows so the block fits the bed."""
        half = len(self.mandrel_radii_mm) // 2
        return self.mandrel_radii_mm[:half], self.mandrel_radii_mm[half:]

    def strain_table(self) -> list[dict[str, float]]:
        """Nominal strain for every strip thickness on every mandrel."""
        return [
            {
                "thickness_mm": thickness,
                "radius_mm": radius,
                "nominal_strain": bend_strain(thickness, radius),
            }
            for thickness in self.strip_thicknesses_mm
            for radius in self.mandrel_radii_mm
        ]

    def strip_count(self) -> int:
        """Total number of test strips."""
        return len(self.strip_thicknesses_mm) * self.repeats


def build_strain_mandrels(spec: StrainCouponSpec | None = None) -> cq.Workplane:
    """Build the block of vertical mandrel posts."""
    spec = spec or StrainCouponSpec()
    row_a, row_b = spec.rows

    def row_width(radii: tuple[float, ...]) -> float:
        return sum(2.0 * r for r in radii) + spec.gap_mm * (len(radii) + 1)

    width = max(row_width(row_a), row_width(row_b))
    depth = (
        3.0 * spec.gap_mm
        + 2.0 * (max(row_a) if row_a else 0.0)
        + 2.0 * (max(row_b) if row_b else 0.0)
    )

    solid = (
        cq.Workplane("XY")
        .moveTo(width / 2.0, depth / 2.0)
        .rect(width, depth)
        .extrude(spec.base_thickness_mm)
    )

    y_a = spec.gap_mm + (max(row_a) if row_a else 0.0)
    y_b = depth - spec.gap_mm - (max(row_b) if row_b else 0.0)
    for radii, y in ((row_a, y_a), (row_b, y_b)):
        x = spec.gap_mm
        for radius in radii:
            x += radius
            solid = solid.union(
                cq.Workplane("XY")
                .moveTo(x, y)
                .circle(radius)
                .extrude(spec.base_thickness_mm + spec.post_height_mm)
            )
            x += radius + spec.gap_mm
    return solid


def build_strain_strips(spec: StrainCouponSpec | None = None) -> cq.Workplane:
    """Build the test strips as upright thin walls, each with a handling tab."""
    spec = spec or StrainCouponSpec()
    shapes: list[cq.Shape] = []
    index = 0
    for thickness in spec.strip_thicknesses_mm:
        for _repeat in range(spec.repeats):
            y = index * spec.strip_pitch_mm
            strip = (
                cq.Workplane("XY")
                .moveTo(spec.strip_length_mm / 2.0, y)
                .rect(spec.strip_length_mm, thickness)
                .extrude(spec.strip_height_mm)
            )
            tab = (
                cq.Workplane("XY")
                .moveTo(-spec.tab_length_mm / 2.0, y)
                .rect(spec.tab_length_mm, spec.tab_width_mm)
                .extrude(spec.strip_height_mm)
            )
            shapes.append(cast("cq.Shape", strip.union(tab).val()))
            index += 1
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(shapes)])


def strain_data_template(spec: StrainCouponSpec | None = None) -> str:
    """Return the CSV data-entry template for the strain test.

    One row per strip thickness and mandrel radius, pre-filled with the nominal
    strain so the only thing to write down at the bench is what was observed.
    """
    spec = spec or StrainCouponSpec()
    header = [
        "specimen_id",
        "strip_thickness_nominal_mm",
        "strip_thickness_measured_mm",
        "mandrel_radius_mm",
        "nominal_strain",
        "whitening_on_bend",
        "crack_on_bend",
        "permanent_set_deg_after_release",
        "whitening_after_10_cycles",
        "crack_after_10_cycles",
        "permanent_set_deg_after_10_cycles",
        "verdict",
        "notes",
    ]
    lines = [",".join(header)]
    for row in spec.strain_table():
        lines.append(
            ",".join(
                [
                    "",
                    f"{row['thickness_mm']:.2f}",
                    "",
                    f"{row['radius_mm']:.1f}",
                    f"{row['nominal_strain']:.5f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        )
    lines.append("")
    lines.append("# verdict: pass | marginal | fail")
    lines.append(
        "# allowable_strain = the largest nominal_strain whose verdict is 'pass' "
        "after 10 cycles, across every repeat"
    )
    lines.append(
        "# measure strip thickness with calipers before testing: nominal and "
        "as-printed will differ, and the strain depends on the measured value"
    )
    return "\n".join(lines) + "\n"


def coupon_metadata(
    flexure: FlexureCouponSpec,
    cantilever: CantileverCouponSpec,
    strain: StrainCouponSpec | None = None,
) -> dict[str, Any]:
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
        "strain_mandrels": {
            "mandrel_radii_mm": list((strain or StrainCouponSpec()).mandrel_radii_mm),
            "post_height_mm": (strain or StrainCouponSpec()).post_height_mm,
            "strain_range": _strain_range(strain or StrainCouponSpec()),
        },
        "strain_strips": {
            "thicknesses_mm": list((strain or StrainCouponSpec()).strip_thicknesses_mm),
            "repeats": (strain or StrainCouponSpec()).repeats,
            "strip_count": (strain or StrainCouponSpec()).strip_count(),
            "length_mm": (strain or StrainCouponSpec()).strip_length_mm,
            "height_mm": (strain or StrainCouponSpec()).strip_height_mm,
        },
    }


def _strain_range(spec: StrainCouponSpec) -> list[float]:
    """Smallest and largest nominal strain the mandrel set produces."""
    values = [row["nominal_strain"] for row in spec.strain_table()]
    return [min(values), max(values)]


@dataclass
class CouponSet:
    """Both coupons plus their metadata, ready to export."""

    flexure_spec: FlexureCouponSpec = field(default_factory=FlexureCouponSpec)
    cantilever_spec: CantileverCouponSpec = field(default_factory=CantileverCouponSpec)
    strain_spec: StrainCouponSpec = field(default_factory=StrainCouponSpec)

    def build(self) -> dict[str, cq.Workplane]:
        """Build every coupon, keyed by output file stem."""
        return {
            "flexure_coupon": build_flexure_coupon(self.flexure_spec),
            "cantilever_coupon": build_cantilever_coupon(self.cantilever_spec),
            "strain_mandrels": build_strain_mandrels(self.strain_spec),
            "strain_strips": build_strain_strips(self.strain_spec),
        }

    def metadata(self) -> dict[str, Any]:
        """Return the combined metadata."""
        return coupon_metadata(self.flexure_spec, self.cantilever_spec, self.strain_spec)
