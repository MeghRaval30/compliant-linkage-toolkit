"""The one-page sheet that goes beside the printed demo pair.

Same four-bar, built two ways, on one side of paper: what each costs to make,
what each does, and -- the part that matters -- which of those numbers are
predictions and which are still blank because nobody has measured them.

The temptation with a sheet like this is to fill every cell. Two are left
explicitly empty and say why:

**Pin-joint friction and backlash.** There is no friction model in this toolkit
and no measurement of the printed pins, so the rigid part's input torque is not
predicted. Writing "~0, ideal pins" would be worse than a blank: it would claim
the comparison the demo exists to make.

**Everything downstream of a placeholder.** Torque and strain both come from the
modulus and the allowable strain, which are placeholders. The sheet says so in
its own header rather than only in a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ComparisonRow:
    """One line of the sheet.

    ``rigid`` and ``compliant`` are strings, not numbers, because several of
    them are deliberately not numbers -- "not predicted", "none by construction"
    -- and a formatter that had to special-case those would end up inventing a
    value for the blank ones.
    """

    label: str
    rigid: str
    compliant: str
    note: str = ""


@dataclass
class ComparisonSheet:
    """Everything on the sheet, ready to render."""

    design: str
    rows: list[ComparisonRow] = field(default_factory=list)
    caveat: str | None = None
    placeholders: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unmeasured: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "design": self.design,
            "rows": [
                {"label": r.label, "rigid": r.rigid, "compliant": r.compliant, "note": r.note}
                for r in self.rows
            ],
            "caveat": self.caveat,
            "placeholders": list(self.placeholders),
            "unmeasured": list(self.unmeasured),
            "provenance": self.provenance,
        }

    def to_markdown(self) -> str:
        """Render the sheet as one page of markdown."""
        lines = [
            f"# {self.design}: the same four-bar, built two ways",
            "",
            "One linkage. Same link lengths, same coupler point, same base footprint and "
            "the same M3 hole pattern, so one fixture position serves both parts and both "
            "draw through a pen hole at the same coordinate.",
            "",
        ]
        if self.caveat:
            lines += [
                f"> **{self.caveat}**",
                ">",
                "> Every row below that depends on material data is a prediction from "
                "placeholder constants. It shows the pipeline runs; it is not what the "
                "printed parts will do. Waiting on: "
                + ", ".join(f"`{p}`" for p in self.placeholders),
                "",
            ]

        lines += [
            "| | pin-jointed (rigid) | compliant (flexures) |",
            "|---|---|---|",
        ]
        for row in self.rows:
            lines.append(f"| **{row.label}** | {row.rigid} | {row.compliant} |")
        lines.append("")

        notes = [r for r in self.rows if r.note]
        if notes:
            lines += ["## Notes", ""]
            lines += [f"- **{r.label}.** {r.note}" for r in notes]
            lines.append("")

        if self.unmeasured:
            lines += [
                "## Deliberately blank",
                "",
                "These are not oversights. Filling them in would claim the comparison the "
                "demo exists to make.",
                "",
            ]
            lines += [f"- {item}" for item in self.unmeasured]
            lines.append("")

        provenance = self.provenance
        lines += [
            "---",
            "",
            f"cmtool {provenance.get('version', '?')} @ "
            f"`{str(provenance.get('code_commit', '?'))[:12]}` &middot; config "
            f"`{str(provenance.get('config_hash') or 'n/a')[:12]}` &middot; "
            f"{provenance.get('created_utc', '')}",
            "",
        ]
        return "\n".join(lines)


def build_comparison(
    scene: Any,
    mechanism: Any,
    *,
    rigid_layout: Any | None = None,
    rigid_estimate: dict[str, float] | None = None,
    compliant_estimate: dict[str, float] | None = None,
    rigid_extent_mm: tuple[float, float, float] | None = None,
    compliant_extent_mm: tuple[float, float, float] | None = None,
    clearance_mm: float | None = None,
) -> ComparisonSheet:
    """Assemble the sheet from a solved scene and the two parts' layouts.

    Parameters
    ----------
    scene
        A :class:`~cmtool.viz.scene.ViewerScene` for the compliant mechanism.
    mechanism
        The :class:`~cmtool.convert.base.CompliantMechanism` behind it.
    rigid_layout
        The :class:`~cmtool.cad.rigid.RigidLayout`, for the part and fastener
        counts. Omitted when only the compliant half was built.
    """
    sheet = ComparisonSheet(
        design=scene.name,
        caveat=scene.caveat,
        placeholders=list(scene.provenance.placeholders_used),
        provenance=scene.provenance.to_dict(),
    )
    joints = len(mechanism.base.joints)
    rows = sheet.rows

    rows.append(
        ComparisonRow(
            "printed bodies",
            str(rigid_layout.n_printed_bodies) if rigid_layout else "-",
            "1",
            "The compliant part is one piece by definition. The rigid part is one piece "
            "only because the pins print already assembled; the bolt variant is four.",
        )
    )
    rows.append(
        ComparisonRow(
            "fasteners",
            str(rigid_layout.n_fasteners) if rigid_layout else "-",
            "0",
        )
    )
    rows.append(
        ComparisonRow(
            "assembly",
            "free four joints by hand"
            if rigid_layout and rigid_layout.joint_style == "print_in_place"
            else "bolt four joints",
            "none",
        )
    )
    rows.append(
        ComparisonRow(
            "moving joints",
            f"{joints} sliding pin joints",
            f"{joints} flexures",
        )
    )
    rows.append(
        ComparisonRow(
            "backlash",
            f"{clearance_mm:.2f} mm radial pin clearance" if clearance_mm else "pin clearance",
            "none",
            "The compliant joint has no clearance to take up: it is solid material. The "
            "pin clearance is a design choice that has not been printed yet, so whether "
            "it frees off at all is still open.",
        )
    )
    rows.append(
        ComparisonRow(
            "friction",
            "sliding, at every pin",
            "none; energy stored elastically",
            "Which is why the compliant part needs torque to *hold* a position and the "
            "rigid one does not.",
        )
    )

    torque = _torque_row(scene)
    rows.append(torque)
    rows.append(_path_row(scene, clearance_mm))
    rows.append(_strain_row(scene, mechanism))

    rows.append(
        ComparisonRow(
            "range of motion",
            "continuous rotation, if the links clear",
            f"{abs(mechanism.input_range_deg[1] - mechanism.input_range_deg[0]):.1f} deg "
            "of input, and no more",
            "The hard limit on the compliant side, and the honest cost of the technology: "
            "a flexure cannot rotate continuously, so there is no full-revolution option "
            "anywhere in this toolkit.",
        )
    )

    if rigid_extent_mm and compliant_extent_mm:
        rows.append(
            ComparisonRow(
                "envelope (mm)",
                " x ".join(f"{v:.0f}" for v in rigid_extent_mm),
                " x ".join(f"{v:.0f}" for v in compliant_extent_mm),
            )
        )
    if rigid_estimate and compliant_estimate:
        rows.append(
            ComparisonRow(
                "ESTIMATED mass (g)",
                f"{rigid_estimate['mass_g']:.1f}",
                f"{compliant_estimate['mass_g']:.1f}",
            )
        )
        rows.append(
            ComparisonRow(
                "ESTIMATED print (min)",
                f"{rigid_estimate['estimated_minutes']:.0f}",
                f"{compliant_estimate['estimated_minutes']:.0f}",
                "Solid volume at an assumed deposition rate. A planning figure, not a "
                "measurement; the slicer's number supersedes it.",
            )
        )

    sheet.unmeasured = [
        "**The rigid part's input torque.** There is no friction model here and no "
        "measurement of the printed pins, so it is not predicted. Whatever it takes to "
        "drive the pin-jointed part is friction, and friction is the thing the compliant "
        "design removes -- claiming a number for it would be claiming the result.",
        "**Both parts' measured coupler paths.** Nothing has been printed. The pen holes "
        "exist so the two curves can be drawn on one sheet of paper and compared directly; "
        "until then the measured series is absent from every figure, not zero.",
        "**Whether the print-in-place pins free off at all.** The clearance is a design "
        "choice carried over from common practice. The first print is the experiment.",
    ]
    return sheet


def _torque_row(scene: Any) -> ComparisonRow:
    """Predicted input torque, or a statement that it is not predicted."""
    values = []
    for key in ("prbm", "fea"):
        series = scene.models.get(key)
        if series is None or not series.torque_nmm:
            continue
        peak = float(np.max(np.abs(np.asarray(series.torque_nmm, dtype=float))))
        values.append(f"{peak:.1f} ({key.upper()})")
    return ComparisonRow(
        "peak input torque (N&middot;mm)",
        "not predicted",
        " / ".join(values) if values else "not solved",
        "The path is fixed by geometry under a prescribed input, so only torque can "
        "discriminate between the two stiffness models -- which is why they disagree here "
        "by about half and agree exactly on the path.",
    )


def _path_row(scene: Any, clearance_mm: float | None) -> ComparisonRow:
    """How far each part's path can sit from the ideal rigid one."""
    fea = next(
        (
            c
            for c in scene.comparisons
            if {c["a"], c["b"]} == {"rigid", "fea"} and c.get("mean_mm") is not None
        ),
        None,
    )
    compliant = (
        f"{fea['mean_mm']:.3f} mm mean, {fea['max_mm']:.3f} mm max (FEA)" if fea else "not solved"
    )
    rigid = (
        f"up to &plusmn;{clearance_mm:.2f} mm from pin clearance alone"
        if clearance_mm
        else "set by pin clearance"
    )
    return ComparisonRow(
        "coupler path vs the ideal rigid path",
        rigid,
        compliant,
        "The rigid figure is a bound from the clearance, not a prediction: a pin can sit "
        "anywhere in its hole. The compliant figure is a solved result. They are the same "
        "order, which is the point -- neither construction reproduces the ideal path, and "
        "only one of them can be predicted before printing.",
    )


def _strain_row(scene: Any, mechanism: Any) -> ComparisonRow:
    """Worst flexure strain against the allowable, over the whole arc."""
    fea = scene.models.get("fea")
    if fea is None or not fea.frames:
        return ComparisonRow("flexure strain margin", "n/a -- no flexures", "not solved")
    peaks = {
        joint: max(max(frame.flexure_strain[joint]) for frame in fea.frames)
        for joint in scene.joints
    }
    worst = max(peaks, key=lambda j: peaks[j])
    allowable = mechanism.feasibility.allowable_strain
    margin = allowable / peaks[worst] if peaks[worst] else float("inf")
    tag = " against a PLACEHOLDER allowable" if scene.allowable_strain_is_placeholder else ""
    return ComparisonRow(
        "flexure strain margin",
        "n/a -- no flexures",
        f"{margin:.2f}x at joint {worst} ({peaks[worst] * 100:.3f}% of {allowable * 100:.2f}%)"
        f"{tag}",
        "The number that decides whether the compliant part survives. It has no rigid "
        "counterpart: a pin joint does not care how far it turns, which is exactly the "
        "trade being made.",
    )


def write_comparison(sheet: ComparisonSheet, path: str | Path) -> Path:
    """Write the sheet as markdown, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet.to_markdown(), encoding="utf-8")
    return out
