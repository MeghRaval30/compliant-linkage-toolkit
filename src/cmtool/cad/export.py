"""Exporting CAD, checking printability, and writing the print sheet.

A part leaves this module with three things: the geometry (STEP and STL), a
printability verdict against the printer's design rules, and a human-readable
print sheet carrying the slicer settings and the metadata checklist that has to
be filled in at the printer.

The print sheet exists because the dataset's value depends on metadata nobody
remembers to record afterwards: which spool, which orientation, what the room
temperature was. Generating the checklist with the part makes it hard to skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cadquery as cq

from cmtool.core.provenance import Provenance, git_commit
from cmtool.materials.loader import Material, Printer

#: Mesh tolerances for STL export, in mm and radians.
STL_LINEAR_TOLERANCE_MM = 0.01
STL_ANGULAR_TOLERANCE_RAD = 0.1


@dataclass(frozen=True)
class PrintabilityCheck:
    """Verdict on whether a part can be printed as designed."""

    fits_envelope: bool
    bounding_box_mm: tuple[float, float, float]
    envelope_mm: tuple[float, float]
    min_feature_mm: float | None
    min_printable_mm: float | None
    below_minimum: bool
    below_minimum_allowed: bool
    notes: list[str]

    @property
    def ok(self) -> bool:
        """Whether the part passes every check that applies to it."""
        return self.fits_envelope and (self.below_minimum_allowed or not self.below_minimum)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "ok": self.ok,
            "fits_envelope": self.fits_envelope,
            "bounding_box_mm": list(self.bounding_box_mm),
            "envelope_mm": list(self.envelope_mm),
            "min_feature_mm": self.min_feature_mm,
            "min_printable_mm": self.min_printable_mm,
            "below_minimum": self.below_minimum,
            "below_minimum_allowed": self.below_minimum_allowed,
            "notes": list(self.notes),
        }


def bounding_box_mm(solid: cq.Workplane | cq.Shape) -> tuple[float, float, float]:
    """Return the ``(x, y, z)`` extent of a shape in mm.

    Aggregates over every solid on the workplane, so a multi-part plate reports
    the size of the whole arrangement rather than of its first strip.
    """
    raw = solid.vals() if isinstance(solid, cq.Workplane) else [solid]
    boxes = [item.BoundingBox() for item in raw if isinstance(item, cq.Shape)]
    if not boxes:
        raise ValueError("shape has no geometry to measure")
    xmin = min(b.xmin for b in boxes)
    ymin = min(b.ymin for b in boxes)
    zmin = min(b.zmin for b in boxes)
    xmax = max(b.xmax for b in boxes)
    ymax = max(b.ymax for b in boxes)
    zmax = max(b.zmax for b in boxes)
    return (float(xmax - xmin), float(ymax - ymin), float(zmax - zmin))


def check_printability(
    solid: cq.Workplane,
    printer: Printer,
    *,
    min_feature_mm: float | None = None,
    allow_below_minimum: bool = False,
    provenance: Provenance | None = None,
) -> PrintabilityCheck:
    """Check a part against a printer's envelope and minimum feature size.

    Parameters
    ----------
    min_feature_mm
        The part's thinnest designed feature, if known. Geometry alone cannot
        reliably report this, so it is passed in from the design that produced it.
    allow_below_minimum
        Set for the flexure coupon, whose entire purpose is to print features
        below the current assumed minimum and find out which of them survive.
    """
    notes: list[str] = []
    extent = bounding_box_mm(solid)
    envelope = printer.design_envelope_mm()
    fits = extent[0] <= envelope[0] and extent[1] <= envelope[1]
    if not fits:
        notes.append(
            f"part is {extent[0]:.1f} x {extent[1]:.1f} mm, larger than the "
            f"{envelope[0]:.0f} x {envelope[1]:.0f} mm design envelope"
        )

    minimum: float | None
    try:
        minimum = printer.min_flexure_thickness_mm(provenance)
    except Exception as exc:
        minimum = None
        notes.append(f"minimum printable thickness unavailable: {exc}")

    below = bool(min_feature_mm is not None and minimum is not None and min_feature_mm < minimum)
    if below and allow_below_minimum:
        notes.append(
            f"thinnest feature {min_feature_mm:.2f} mm is below the assumed printable "
            f"minimum {minimum:.2f} mm -- intentional: this part exists to measure that limit"
        )
    elif below:
        notes.append(
            f"thinnest feature {min_feature_mm:.2f} mm is below the printable minimum "
            f"{minimum:.2f} mm"
        )

    return PrintabilityCheck(
        fits_envelope=fits,
        bounding_box_mm=extent,
        envelope_mm=envelope,
        min_feature_mm=min_feature_mm,
        min_printable_mm=minimum,
        below_minimum=below,
        below_minimum_allowed=allow_below_minimum,
        notes=notes,
    )


def export_solid(
    solid: cq.Workplane,
    out_dir: str | Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("step", "stl"),
) -> dict[str, Path]:
    """Write a shape to disk in the requested formats; return the paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        if fmt == "stl":
            cq.exporters.export(
                solid,
                str(path),
                tolerance=STL_LINEAR_TOLERANCE_MM,
                angularTolerance=STL_ANGULAR_TOLERANCE_RAD,
            )
        elif fmt == "step":
            cq.exporters.export(solid, str(path))
        else:
            raise ValueError(f"unsupported export format {fmt!r}")
        written[fmt] = path
    return written


def write_print_sheet(
    path: str | Path,
    *,
    title: str,
    printer: Printer,
    material: Material,
    purpose: str,
    check: PrintabilityCheck,
    details: dict[str, Any] | None = None,
    instructions: list[str] | None = None,
) -> Path:
    """Write the markdown print sheet that goes with a part.

    Carries the slicer settings from the printer config, the part's dimensions,
    what the print is *for*, and the metadata checklist to fill in at the machine.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    supplier = material.supplier

    lines: list[str] = [
        f"# Print sheet: {title}",
        "",
        f"Generated by cmtool at commit `{git_commit()}` on {date.today().isoformat()}.",
        "",
        "## Purpose",
        "",
        purpose,
        "",
        "## Printer",
        "",
        f"- Machine: **{printer.model}** (`{printer.name}`, {printer.role})",
        f"- Nozzle: {printer.nozzle_mm():.1f} mm, layer height {printer.layer_height_mm():.2f} mm",
        "- Orientation: mechanism plane flat on the bed (flexures bend in-plane)",
        "",
        "## Part",
        "",
        f"- Bounding box: {check.bounding_box_mm[0]:.1f} x {check.bounding_box_mm[1]:.1f} "
        f"x {check.bounding_box_mm[2]:.1f} mm",
        f"- Design envelope: {check.envelope_mm[0]:.0f} x {check.envelope_mm[1]:.0f} mm",
        f"- Printability: {'OK' if check.ok else 'FAILED'}",
    ]
    for note in check.notes:
        lines.append(f"  - {note}")

    if details:
        lines += ["", "## Details", ""]
        for key, value in details.items():
            lines.append(f"- {key}: {value}")

    slicer = printer.slicer
    settings = slicer.get("settings", {}) if isinstance(slicer, dict) else {}
    if settings:
        lines += [
            "",
            f"## Slicer settings ({slicer.get('target', 'slicer')})",
            "",
            "Design choices for printing thin flexures consistently, not measurements.",
            "Verify the exact setting names in your slicer version.",
            "",
            "| Setting | Value | Why |",
            "| --- | --- | --- |",
        ]
        for key, entry in settings.items():
            if isinstance(entry, dict):
                value = entry.get("value")
                why = str(entry.get("why") or entry.get("note") or "").replace("\n", " ").strip()
            else:
                value, why = entry, ""
            shown = "_to be set_" if value is None else f"`{value}`"
            lines.append(f"| {key} | {shown} | {why} |")

    if instructions:
        lines += ["", "## How to run this test", ""]
        lines += [f"{i}. {step}" for i, step in enumerate(instructions, start=1)]

    lines += [
        "",
        "## Metadata to record (fill in at the printer)",
        "",
        f"- [ ] Printer used: `{printer.name}` (all Phase A/B parts use the primary printer)",
        f"- [ ] Filament brand: {supplier.get('brand') or '____'}",
        f"- [ ] Product line (PLA / PLA+): {supplier.get('product_line') or '____'}",
        f"- [ ] Colour: {supplier.get('colour') or '____'}",
        f"- [ ] Lot number: {supplier.get('lot_id') or '____'}",
        "- [ ] Nozzle temperature: ____ C",
        "- [ ] Bed temperature: ____ C",
        "- [ ] Infill: ____ %",
        "- [ ] Print date: ____",
        "- [ ] Test date: ____",
        "- [ ] Room temperature: ____ C",
        "- [ ] Room humidity: ____ %",
        "- [ ] Anything unusual about the print (stringing, warping, a failed strip): ____",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
