"""Command-line interface.

Batch work goes through this CLI so that runs are scriptable and logged::

    cmtool info
    cmtool coupons --out out/coupons
    cmtool design --seed 1 --out out/designs
    cmtool convert out/designs/fb_01_0077.json
    cmtool simulate examples/fourbar.json --steps 91 --csv out/path.csv

Subcommands belonging to later milestones are present but refuse to run, naming
the milestone, so the CLI surface is visible from the start without pretending
to do work it cannot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from cmtool import __version__
from cmtool.api import available_solvers, simulate
from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance, git_commit

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Rigid-to-compliant planar linkage toolkit.",
)
console = Console()


def _not_yet(milestone: str, what: str) -> None:
    """Fail clearly for a subcommand that belongs to a later milestone."""
    console.print(f"[yellow]{what} is not implemented yet[/] (milestone {milestone}).")
    raise typer.Exit(code=2)


@app.command()
def info() -> None:
    """Show version, registered plug-ins and the current code commit."""
    from cmtool.api import available_flexures, available_strategies
    from cmtool.kinematics import KINEMATICS

    table = Table(title="cmtool", show_header=False, box=None)
    table.add_row("version", __version__)
    table.add_row("code commit", git_commit())
    table.add_row("solvers", ", ".join(available_solvers()))
    table.add_row("kinematics", ", ".join(KINEMATICS.names()))
    table.add_row("flexures", ", ".join(available_flexures()))
    table.add_row("strategies", ", ".join(available_strategies()))
    console.print(table)


@app.command(name="simulate")
def simulate_cmd(
    linkage_json: Annotated[Path, typer.Argument(help="Linkage JSON file")],
    solver: Annotated[str, typer.Option(help="Registered solver name")] = "rigid",
    steps: Annotated[int, typer.Option(help="Samples across the input arc")] = 61,
    start_deg: Annotated[float | None, typer.Option(help="Arc start (absolute deg)")] = None,
    end_deg: Annotated[float | None, typer.Option(help="Arc end (absolute deg)")] = None,
    output: Annotated[str | None, typer.Option(help="Output point name")] = None,
    csv_out: Annotated[Path | None, typer.Option("--csv", help="Write the path to CSV")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write a summary to JSON")] = None,
) -> None:
    """Simulate a linkage over its input arc and report the coupler path."""
    linkage = Linkage.from_json(linkage_json)

    arc: tuple[float, float] | None = None
    if start_deg is not None and end_deg is not None:
        arc = (start_deg, end_deg)
    elif (start_deg is None) != (end_deg is None):
        console.print("[red]give both --start-deg and --end-deg, or neither[/]")
        raise typer.Exit(code=2)

    result = simulate(linkage, solver=solver, input_range_deg=arc, n_steps=steps)
    path = result.path(output)

    table = Table(title=f"{linkage.name} [{solver}]", show_header=False, box=None)
    table.add_row("states", str(result.n_states))
    table.add_row(
        "input arc (deg)",
        f"{result.input_angles_deg[0]:.3f} -> {result.input_angles_deg[-1]:.3f}",
    )
    for joint, excursion in result.joint_excursion_deg().items():
        table.add_row(f"excursion {joint} (deg)", f"{excursion:.3f}")
    grashof = result.diagnostics.get("grashof")
    if grashof:
        table.add_row("grashof", f"{grashof['classification']} ({grashof['condition']})")
    mu_min = result.diagnostics.get("transmission_angle_min_deg")
    mu_max = result.diagnostics.get("transmission_angle_max_deg")
    if mu_min is not None:
        table.add_row("transmission angle (deg)", f"{mu_min:.2f} .. {mu_max:.2f}")
    if result.diagnostics.get("branch_flip"):
        table.add_row("[red]branch flip[/]", "yes")
    console.print(table)

    caveat = result.provenance.caveat()
    if caveat:
        console.print(f"[yellow]{caveat}[/]")

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["input_angle_deg", "input_sweep_deg", "x_mm", "y_mm"])
            for angle, sweep, (x, y) in zip(
                result.input_angles_deg, result.input_sweep_deg, path, strict=True
            ):
                writer.writerow([f"{angle:.6f}", f"{sweep:.6f}", f"{x:.6f}", f"{y:.6f}"])
        console.print(f"wrote {csv_out}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        summary = result.summary()
        summary["provenance"] = result.provenance.to_dict()
        json_out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        console.print(f"wrote {json_out}")


@app.command(name="convert")
def convert_cmd(
    linkage_json: Annotated[Path, typer.Argument(help="Linkage JSON file")],
    material: Annotated[str, typer.Option(help="Material config name")] = "PLA",
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
    flexure: Annotated[str, typer.Option(help="Registered flexure type")] = "small_length_pivot",
    thickness_mm: Annotated[
        float | None, typer.Option(help="Flexure thickness; default is the printer minimum")
    ] = None,
    fit_arc: Annotated[
        bool, typer.Option(help="Shrink the input arc to meet the excursion target")
    ] = True,
    target_excursion_deg: Annotated[
        float, typer.Option(help="Ceiling for the worst joint excursion")
    ] = 22.0,
    unstressed_at: Annotated[
        str, typer.Option(help="Which configuration is printed: mid_arc or start")
    ] = "mid_arc",
    placement: Annotated[str, typer.Option(help="pivot_matched or unmatched")] = "pivot_matched",
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the report to JSON")
    ] = None,
) -> None:
    """Convert a rigid linkage into a compliant one and report per-joint feasibility."""
    from cmtool.api import convert as convert_api
    from cmtool.convert.arc import ArcFitError, fit_input_arc

    linkage = Linkage.from_json(linkage_json)

    arc = linkage.input_range_deg
    if fit_arc:
        try:
            fit = fit_input_arc(linkage, max_joint_excursion_deg=target_excursion_deg)
        except ArcFitError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        arc = fit.input_range_deg
        console.print(
            f"fitted input arc {arc[0]:.2f} -> {arc[1]:.2f} deg "
            f"({fit.input_excursion_deg:.2f} deg of input; worst joint "
            f"{fit.binding_joint} at {fit.max_excursion_deg:.2f} deg)"
        )

    mech = convert_api(
        linkage,
        flexures=flexure,
        material=material,
        printer=printer,
        input_range_deg=arc,
        thickness_mm=thickness_mm,
        unstressed_at=unstressed_at,
        placement=placement,
    )

    table = Table(title=f"{linkage.name} -> compliant [{mech.flexure_type}]")
    table.add_column("joint")
    for column in (
        "excursion",
        "max bend",
        "L chosen",
        "L strain min",
        "L geom max",
        "peak strain",
        "util",
    ):
        table.add_column(column, justify="right")
    table.add_column("PRBM model")
    table.add_column("ok")

    for name, sized in mech.sizing.items():
        table.add_row(
            name,
            f"{sized.excursion_deg:.2f}",
            f"{sized.max_bend_deg:.2f}",
            f"{sized.geometry.length_mm:.2f}",
            f"{sized.min_length_strain_mm:.2f}",
            f"{sized.max_length_geometric_mm:.2f}",
            f"{sized.strain.peak_strain:.5f}",
            f"{sized.utilisation:.2f}",
            sized.validity.model,
            "[green]yes[/]" if sized.feasible else "[red]NO[/]",
        )
    console.print(table)

    report = mech.feasibility
    verdict = "[green]FEASIBLE[/]" if report.feasible else "[red]NOT FEASIBLE[/]"
    console.print(
        f"{verdict}  limiting joint: [bold]{report.binding_joint}[/] "
        f"(utilisation {report.max_utilisation:.2f})"
    )
    for reason in report.reasons():
        console.print(f"  [red]{reason}[/]")
    for note in report.prbm_notes():
        console.print(f"  [dim]{note}[/]")

    caveat = mech.provenance.caveat()
    if caveat:
        console.print(f"[yellow]{caveat}[/]")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = mech.summary()
        payload["provenance"] = mech.provenance.to_dict()
        json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        console.print(f"wrote {json_out}")


@app.command()
def coupons(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("out/coupons"),
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
    material: Annotated[str, typer.Option(help="Material config name")] = "PLA",
) -> None:
    """Export the flexure and cantilever test coupons, with print sheets.

    These are printed before any mechanism: they measure the minimum printable
    flexure thickness and the material modulus that every later design needs.
    """
    from cmtool.cad import CouponSet, check_printability, export_solid, write_print_sheet
    from cmtool.materials import Material, Printer

    printer_cfg = Printer.load(printer)
    material_cfg = Material.load(material)
    coupon_set = CouponSet()
    solids = coupon_set.build()
    meta = coupon_set.metadata()

    purposes = {
        "strain_mandrels": (
            "Measure the allowable bending strain. Wrap a test strip around each post and "
            "look for whitening, cracking and permanent set -- first on a single bend, then "
            "after ten cycles. The strain a post imposes is (t/2)/(R + t/2), so the posts "
            "cover roughly 0.008 to 0.057 strain across the three strip thicknesses. The "
            "largest strain that still passes after ten cycles sets `allowable_strain`, "
            "which is the number that decides how far every flexure in the project may bend."
        ),
        "strain_strips": (
            "The strips for the mandrel test. They are printed as upright thin walls, not "
            "flat: a flexure bends in-plane, so a strip must bend about the same axis or the "
            "test measures interlayer adhesion instead of the property we need. Print with a "
            "brim -- these are tall, thin and free-standing."
        ),
        "flexure_coupon": (
            "Find the minimum flexure thickness this printer can produce consistently. "
            "Print it, measure every strip with calipers, then bend each one by hand. The "
            "thinnest strip that prints completely, measures close to nominal and survives "
            "handling sets `min_flexure_thickness_mm` in the printer config. Until that "
            "number is measured, every feasibility verdict the toolkit produces is flagged "
            "as not a physical prediction."
        ),
        "cantilever_coupon": (
            "Measure the effective Young's modulus of the printed material by cantilever "
            "deflection. Two thicknesses are included deliberately: a 1 mm strip is almost "
            "entirely perimeter while a 2 mm strip contains infill, so comparing them shows "
            "whether apparent modulus depends on wall structure. If it does, a modulus "
            "measured on a thick strip cannot be applied to a 0.5 mm flexure unchanged."
        ),
    }
    instructions = {
        "strain_mandrels": [
            "Print the mandrel block and the strips together, from the same spool.",
            "Measure each strip's thickness with calipers before bending anything: the "
            "strain depends on the as-printed thickness, not the nominal one.",
            "Start at the largest post (lowest strain) and work down.",
            "Wrap a strip about 90 degrees around the post, hold for five seconds, release.",
            "Record whitening, any crack, and the permanent set angle after release.",
            "Repeat the same strip and post ten times, then record the same three things.",
            "Mark each strip and post combination pass, marginal or fail in the CSV.",
            "The largest nominal strain that still passes after ten cycles, across every "
            "repeat, is the allowable strain. Put it in configs/materials/pla.yaml with "
            "status: measured and the date.",
            "Fill in strain_test_template.csv as you go; it already has the nominal strains.",
        ],
        "strain_strips": [
            "Print with a brim. These are free-standing thin walls and will topple without one.",
            "Keep them with the mandrel block: they are one experiment.",
        ],
        "flexure_coupon": [
            "Print flat on the bed with the settings in the table above. No supports.",
            "Photograph the plate before touching it, and note any strip that failed to print.",
            "Measure each strip's thickness with calipers at three points along its length.",
            "Record nominal against measured thickness, and the spread across those points.",
            "Bend each paddle by hand through roughly 20 degrees and back, ten times.",
            "Record which strips survive, which whiten, and which break.",
            "Set `min_flexure_thickness_mm` in configs/printer/bambu_a1.yaml from the result, "
            "with `status: measured` and the date.",
        ],
        "cantilever_coupon": [
            "Print flat on the bed. Keep all six strips from a single print job.",
            "Measure each strip's thickness and width with calipers and record them separately.",
            "Clamp one end with a known free length (about 80 mm) and record that length.",
            "Hang a known mass at the tip and measure tip deflection, using at least three masses.",
            "Keep deflections small, under about 10 percent of the free length, so linear beam "
            "theory applies.",
            "Compute E = F L^3 / (3 delta I) with I = b h^3 / 12, using the MEASURED b and h.",
            "Record the loading rate and how long each load was held: PLA creeps, so the "
            "modulus depends on both.",
            "Put the result in configs/materials/pla.yaml with `status: measured`, plus the "
            "date, the rate and the orientation.",
        ],
    }

    thin_parts = {
        "flexure_coupon": min(coupon_set.flexure_spec.thicknesses_mm),
        "strain_strips": min(coupon_set.strain_spec.strip_thicknesses_mm),
    }
    for stem, solid in solids.items():
        min_feature = thin_parts.get(stem)
        check = check_printability(
            solid,
            printer_cfg,
            min_feature_mm=min_feature,
            allow_below_minimum=stem in thin_parts,
        )
        paths = export_solid(solid, out, stem)
        sheet = write_print_sheet(
            Path(out) / f"{stem}_print_sheet.md",
            title=stem.replace("_", " "),
            printer=printer_cfg,
            material=material_cfg,
            purpose=purposes[stem],
            check=check,
            details=meta[stem],
            instructions=instructions[stem],
        )
        status = "[green]OK[/]" if check.ok else "[red]FAILED[/]"
        size = check.bounding_box_mm
        console.print(
            f"{stem}: {status}  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm  "
            f"-> {paths['step'].name}, {paths['stl'].name}, {sheet.name}"
        )
        for note in check.notes:
            console.print(f"    [yellow]{note}[/]")

    meta_path = Path(out) / "coupons.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    console.print(f"wrote {meta_path}")

    from cmtool.cad.coupons import strain_data_template

    template = Path(out) / "strain_test_template.csv"
    template.write_text(strain_data_template(coupon_set.strain_spec), encoding="utf-8")
    console.print(f"wrote {template}")


@app.command()
def design(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("out/designs"),
    seed: Annotated[int, typer.Option(help="Random seed")] = 1,
    candidates: Annotated[int, typer.Option(help="Candidates to sample")] = 250,
    keep: Annotated[int, typer.Option(help="Designs to keep")] = 3,
    target_excursion_deg: Annotated[
        float, typer.Option(help="Ceiling for the worst joint excursion")
    ] = 22.0,
    thickness_mm: Annotated[float | None, typer.Option(help="Flexure thickness")] = None,
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
) -> None:
    """Search for four-bar designs that are feasible as compliant mechanisms."""
    from cmtool.convert.search import search
    from cmtool.materials import Printer

    envelope = Printer.load(printer).design_envelope_mm()
    report = search(
        n_candidates=candidates,
        seed=seed,
        keep=keep,
        max_joint_excursion_deg=target_excursion_deg,
        envelope_mm=envelope,
        thickness_mm=thickness_mm,
        printer=printer,
    )

    console.print(
        f"sampling links from {report.link_floor_mm:.1f} mm upward: the closed-form bound "
        f"for {target_excursion_deg:.0f} deg of joint excursion"
    )

    tally = Table(title="rejections")
    tally.add_column("reason")
    tally.add_column("count", justify="right")
    for reason, count in sorted(report.reasons.items(), key=lambda kv: -kv[1]):
        tally.add_row(reason, str(count))
    console.print(tally)

    if not report.kept:
        console.print("[red]no feasible designs found[/]")
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    (out / "search_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    for candidate in report.kept:
        mech = candidate.compliant
        assert mech is not None and candidate.arc_deg is not None
        linkage = candidate.linkage
        linkage.input_range_deg = candidate.arc_deg
        linkage.to_json(out / f"{linkage.name}.json")
        (out / f"{linkage.name}_report.json").write_text(
            json.dumps(candidate.summary(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        worst = max(candidate.excursions_deg, key=lambda k: candidate.excursions_deg[k])
        console.print(
            f"[green]{linkage.name}[/]: arc "
            f"{candidate.arc_deg[0]:.1f}..{candidate.arc_deg[1]:.1f} deg, worst joint "
            f"{worst} at {candidate.excursions_deg[worst]:.1f} deg, limiting joint "
            f"{mech.feasibility.binding_joint} (util {mech.feasibility.max_utilisation:.2f})"
        )


@app.command()
def generate() -> None:
    """Generate random feasible linkage samples for the dataset (Phase B)."""
    _not_yet("B", "dataset sample generation")


def _export_rigid(
    linkage_json: Path,
    *,
    out: Path,
    material: str,
    printer: str,
    joint_style: str,
    clearance_mm: float,
    pen_hole_mm: float,
    link_width_mm: float,
    base_depth_mm: float,
    base_margin_mm: float,
    bolt_inset_mm: float,
) -> None:
    """Export the pin-jointed control part, its print sheet and its layout."""
    from cmtool.cad import check_printability, export_solid, write_print_sheet
    from cmtool.cad.rigid import RigidCadSpec, build_rigid_mechanism, estimate_print
    from cmtool.materials import Material, Printer

    linkage = Linkage.from_json(linkage_json)
    printer_cfg = Printer.load(printer)
    material_cfg = Material.load(material)

    spec = RigidCadSpec(
        joint_style=joint_style,
        clearance_mm=clearance_mm,
        pen_hole_diameter_mm=pen_hole_mm,
        link_width_mm=link_width_mm,
        base_depth_mm=base_depth_mm,
        base_margin_mm=base_margin_mm,
        bolt_inset_mm=bolt_inset_mm,
    )
    try:
        solid, layout = build_rigid_mechanism(linkage, spec=spec)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    collisions = [w for w in layout.warnings if w.startswith("COLLISION")]
    if collisions:
        for note in collisions:
            console.print(f"[red]{note}[/]")
        raise typer.Exit(code=1)

    # The thinnest *solid* feature, not the clearance: a gap is not a wall, and
    # feeding the clearance in here reports the part as unprintable for having a
    # well-made joint.
    thinnest = min(
        spec.cap_thickness_mm,
        spec.link_width_mm / 2.0 - (spec.pin_diameter_mm / 2.0 + spec.clearance_mm),
    )
    check = check_printability(solid, printer_cfg, min_feature_mm=thinnest)
    stem = f"{linkage.name}_rigid"
    paths = export_solid(solid, out, stem)

    provenance = Provenance(notes={"part": stem})
    density = material_cfg.density_kg_per_m3(provenance)
    estimate = estimate_print(solid, density_kg_per_m3=density)

    details: dict[str, object] = {
        "joint style": joint_style,
        "pin clearance (mm, radial)": clearance_mm,
        "thinnest solid wall (mm)": round(thinnest, 2),
        "pin diameter (mm)": spec.pin_diameter_mm,
        "link width x thickness (mm)": f"{spec.link_width_mm} x {spec.link_thickness_mm}",
        "stacked level heights (mm)": {k: round(v, 2) for k, v in layout.level_z_mm.items()},
        "moving joints": layout.n_moving_joints,
        "printed bodies": layout.n_printed_bodies,
        "fasteners needed": layout.n_fasteners,
        "pen hole": (
            f"{pen_hole_mm:.1f} mm dia at "
            f"({layout.pen_hole_mm[0]:.1f}, {layout.pen_hole_mm[1]:.1f}), "
            f"top face z = {layout.pen_hole_z_mm + spec.link_thickness_mm:.1f} mm"
            if layout.pen_hole_mm
            else "none"
        ),
        "bolt holes (mm)": [[round(x, 2), round(y, 2)] for x, y in layout.bolt_holes_mm],
        "ESTIMATED mass (g)": round(estimate["mass_g"], 1),
        "ESTIMATED print time (min)": round(estimate["estimated_minutes"]),
        "estimate basis": (
            f"solid volume {estimate['volume_mm3']:.0f} mm^3 at "
            f"{estimate['assumed_rate_mm3_per_s']:.0f} mm^3/s -- a planning figure, "
            "NOT a measurement; use the slicer's number once you have it"
        ),
    }
    instructions = [
        "Print flat on the bed exactly as exported. No supports, no raft, no brim "
        "unless the base plate lifts.",
        "OrcaSlicer: 0.2 mm layers, 0.4 mm nozzle, Arachne wall generator, 3 walls, "
        "20% infill. Arachne matters here for the same reason as everywhere else in "
        "this project.",
        "Do NOT enable ironing or elephant-foot compensation on the first print: both "
        "change the effective clearance at the joints.",
        "Let the part cool to room temperature before flexing anything.",
        "Free the joints by twisting each link gently back and forth. If a joint will "
        "not free off, reprint with --clearance-mm 0.45; if it rattles, 0.25.",
        "Bolt the base to the fixture with M3 before driving the input link.",
        "For the drawing: put paper under the coupler, drop a fineliner through the pen "
        "hole so its tip rests on the paper, and walk the input link slowly from one end "
        "of its arc to the other.",
        "Draw the compliant part's path on the SAME sheet, with the base in the same "
        "place, so the two curves can be compared directly.",
    ]
    sheet = write_print_sheet(
        Path(out) / f"{stem}_print_sheet.md",
        title=f"rigid pin-jointed four-bar {linkage.name}",
        printer=printer_cfg,
        material=material_cfg,
        purpose=(
            "The control half of the demo pair: the same four-bar built the way it "
            "would have been built before compliant mechanisms. Printed alongside the "
            "compliant part, it is what makes the no-assembly, no-friction argument "
            "visible rather than asserted."
        ),
        check=check,
        details=details,
        instructions=instructions,
    )

    report = {
        "linkage": linkage.to_dict(),
        "layout": layout.to_dict(),
        "printability": check.to_dict(),
        "estimate": estimate,
        "provenance": provenance.to_dict(),
    }
    report_path = Path(out) / f"{stem}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    size = check.bounding_box_mm
    status = "[green]OK[/]" if check.ok else "[red]DOES NOT FIT[/]"
    console.print(
        f"{stem}: {status}  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm, "
        f"~{estimate['mass_g']:.0f} g, ~{estimate['estimated_minutes']:.0f} min (estimated) "
        f"-> {paths['stl'].name}, {paths['step'].name}, {sheet.name}, {report_path.name}"
    )
    for note in layout.warnings:
        console.print(f"    [yellow]{note}[/]")
    for note in check.notes:
        console.print(f"    [yellow]{note}[/]")
    caveat = provenance.caveat()
    if caveat:
        console.print(f"[yellow]{caveat}[/]")


@app.command()
def export(
    linkage_json: Annotated[Path, typer.Argument(help="Linkage JSON file")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("out/mechanisms"),
    material: Annotated[str, typer.Option(help="Material config name")] = "PLA",
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
    thickness_mm: Annotated[
        float | None, typer.Option(help="Flexure thickness; default is the printer minimum")
    ] = None,
    fit_arc: Annotated[
        bool, typer.Option(help="Shrink the input arc to meet the excursion target")
    ] = True,
    target_excursion_deg: Annotated[
        float, typer.Option(help="Ceiling for the worst joint excursion")
    ] = 22.0,
    link_width_mm: Annotated[float, typer.Option(help="Rigid link width")] = 8.0,
    lever_length_mm: Annotated[float, typer.Option(help="Input lever length")] = 45.0,
    rigid: Annotated[
        bool,
        typer.Option(
            "--rigid/--compliant",
            help="Export the pin-jointed control part instead of the compliant one",
        ),
    ] = False,
    joint_style: Annotated[
        str, typer.Option(help="Rigid only: print_in_place or bolt")
    ] = "print_in_place",
    clearance_mm: Annotated[float, typer.Option(help="Rigid only: radial pin clearance")] = 0.35,
    pen_hole_mm: Annotated[
        float,
        typer.Option(help="Through-hole at the coupler point so the part draws its path"),
    ] = 5.0,
    base_depth_mm: Annotated[
        float, typer.Option(help="Base plate depth; match it across a demo pair")
    ] = 38.0,
    base_margin_mm: Annotated[
        float, typer.Option(help="Base plate overhang past each ground pivot")
    ] = 14.0,
    bolt_inset_mm: Annotated[
        float, typer.Option(help="Mounting hole inset from the base plate corners")
    ] = 8.0,
) -> None:
    """Export the printable part for a mechanism, compliant or pin-jointed.

    ``--rigid`` gives the control: the same four-bar with pin joints, the same
    link lengths, coupler point, base footprint and mounting holes. Both halves
    carry a pen hole at the coupler point, so each draws its own coupler path on
    the same sheet of paper.
    """
    if rigid:
        _export_rigid(
            linkage_json,
            out=out,
            material=material,
            printer=printer,
            joint_style=joint_style,
            clearance_mm=clearance_mm,
            pen_hole_mm=pen_hole_mm,
            link_width_mm=max(link_width_mm, 10.0),
            base_depth_mm=base_depth_mm,
            base_margin_mm=base_margin_mm,
            bolt_inset_mm=bolt_inset_mm,
        )
        return

    from cmtool.api import convert as convert_api
    from cmtool.cad import check_printability, export_solid, write_print_sheet
    from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism
    from cmtool.convert.arc import ArcFitError, fit_input_arc
    from cmtool.materials import Material, Printer

    linkage = Linkage.from_json(linkage_json)
    arc = linkage.input_range_deg
    if fit_arc:
        try:
            fit = fit_input_arc(linkage, max_joint_excursion_deg=target_excursion_deg)
        except ArcFitError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        arc = fit.input_range_deg

    mech = convert_api(
        linkage,
        material=material,
        printer=printer,
        input_range_deg=arc,
        thickness_mm=thickness_mm,
    )
    if not mech.feasibility.feasible:
        console.print(
            f"[red]not feasible[/]: joint {mech.feasibility.binding_joint} -- "
            f"{mech.sizing[mech.feasibility.binding_joint].limit_reason}"
        )
        raise typer.Exit(code=1)

    printer_cfg = Printer.load(printer)
    material_cfg = Material.load(material)
    spec = MechanismCadSpec(
        link_width_mm=link_width_mm,
        lever_length_mm=lever_length_mm,
        pen_hole_diameter_mm=pen_hole_mm,
        base_depth_mm=base_depth_mm,
        base_margin_mm=base_margin_mm,
        bolt_inset_mm=bolt_inset_mm,
    )
    solid, layout = build_mechanism(mech, spec=spec, printer=printer_cfg)

    thinnest = min(s.geometry.thickness_mm for s in mech.sizing.values())
    check = check_printability(solid, printer_cfg, min_feature_mm=thinnest)
    stem = linkage.name
    paths = export_solid(solid, out, stem)

    details: dict[str, object] = {
        "input arc (deg)": (f"{mech.input_range_deg[0]:.2f} to {mech.input_range_deg[1]:.2f}"),
        "printed unstressed at (deg)": f"{mech.reference_input_deg:.2f}",
        "flexure thickness (mm)": thinnest,
        "flexure lengths (mm)": {n: round(s.geometry.length_mm, 2) for n, s in mech.sizing.items()},
        "joint excursions (deg)": {n: round(s.excursion_deg, 2) for n, s in mech.sizing.items()},
        "PRBM model per joint": mech.feasibility.prbm_models,
        "all small-length": mech.feasibility.all_small_length,
        "fiducial pad spacing (mm)": round(layout.fiducial_spacing_mm, 2),
        "force hole radius from input pivot (mm)": round(layout.force_radius_mm, 2),
        "marker pad plane z (mm)": layout.pad_plane_z_mm,
        "pen hole at coupler point (mm)": (
            f"{pen_hole_mm:.1f} dia at "
            f"({layout.coupler_pad_mm[0]:.1f}, {layout.coupler_pad_mm[1]:.1f})"
            if pen_hole_mm > 0.0 and layout.coupler_pad_mm
            else "none"
        ),
        "bolt holes (mm)": [[round(x, 2), round(y, 2)] for x, y in layout.bolt_holes_mm],
    }
    instructions = [
        "Print flat on the bed: the mechanism plane goes parallel to the bed so the "
        "flexures bend in-plane.",
        "No supports. Support material on a flexure ruins its surface.",
        "Check every flexure under a bright light before flexing anything, and photograph "
        "the part before first use.",
        "Bolt the base to the fixture with M3 before applying any load to the lever.",
        "Stick the ArUco markers on the raised pads. All pads are coplanar, which the "
        "camera homography depends on.",
        "Move the lever gently by hand through the stated arc; do not force it past either end.",
        "For the torque measurement, hook a spring scale through the hole in the lever and "
        "pull perpendicular to the lever. Record force against angle, loading and unloading, "
        "in the torque template written beside this sheet.",
    ]
    sheet = write_print_sheet(
        Path(out) / f"{stem}_print_sheet.md",
        title=f"compliant mechanism {stem}",
        printer=printer_cfg,
        material=material_cfg,
        purpose=(
            "The first printed compliant mechanism. Its measured coupler path is what the "
            "rigid, PRBM and FEA predictions get compared against."
        ),
        check=check,
        details=details,
        instructions=instructions,
    )

    from cmtool.metrics import write_scale_template, write_weight_template

    # Mode A, dead weight over a pulley, is the primary rig; mode B is the backup.
    template = write_weight_template(
        Path(out) / f"{stem}_torque_weights.csv",
        hole_radius_mm=layout.force_radius_mm,
        printer=printer,
    )
    backup_template = write_scale_template(
        Path(out) / f"{stem}_torque_scale.csv",
        lever_radius_mm=layout.force_radius_mm,
        input_range_deg=mech.input_range_deg,
        printer=printer,
    )

    report = {
        "conversion": mech.summary(),
        "layout": layout.to_dict(),
        "printability": check.to_dict(),
        "provenance": mech.provenance.to_dict(),
    }
    report_path = Path(out) / f"{stem}_mechanism.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    size = check.bounding_box_mm
    status = "[green]OK[/]" if check.ok else "[red]DOES NOT FIT[/]"
    console.print(
        f"{stem}: {status}  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm  -> "
        f"{paths['step'].name}, {paths['stl'].name}, {sheet.name}, {report_path.name}, "
        f"{template.name}, {backup_template.name}"
    )
    for note in check.notes:
        console.print(f"    [yellow]{note}[/]")
    for warning in layout.warnings:
        console.print(f"    [yellow]{warning}[/]")
    for note in mech.feasibility.prbm_notes():
        console.print(f"    [dim]{note}[/]")
    caveat = mech.provenance.caveat()
    if caveat:
        console.print(f"[yellow]{caveat}[/]")


def _convert_for_view(
    linkage: Linkage,
    *,
    material: str,
    printer: str,
    thickness_mm: float | None,
    fit_arc: bool,
    target_excursion_deg: float,
) -> Any:
    """Convert a linkage the same way ``export`` does, so the picture matches the part."""
    from cmtool.api import convert as convert_api
    from cmtool.convert.arc import ArcFitError, fit_input_arc

    arc = linkage.input_range_deg
    if fit_arc:
        try:
            fit = fit_input_arc(linkage, max_joint_excursion_deg=target_excursion_deg)
        except ArcFitError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        arc = fit.input_range_deg
    return convert_api(
        linkage,
        material=material,
        printer=printer,
        input_range_deg=arc,
        thickness_mm=thickness_mm,
    )


@app.command(name="view")
def view_cmd(
    linkage_json: Annotated[Path, typer.Argument(help="Linkage JSON file")],
    html_out: Annotated[Path, typer.Option("--html", help="Where to write the viewer")] = Path(
        "out/view.html"
    ),
    steps: Annotated[int, typer.Option(help="Precomputed states across the input arc")] = 41,
    measured_csv: Annotated[
        Path | None, typer.Option("--measured", help="Tracked path CSV to overlay")
    ] = None,
    solvers: Annotated[
        str, typer.Option(help="Comma-separated models to include: rigid, prbm, fea")
    ] = "rigid,prbm,fea",
    material: Annotated[str, typer.Option(help="Material config name")] = "PLA",
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
    thickness_mm: Annotated[
        float | None, typer.Option(help="Flexure thickness; default is the printer minimum")
    ] = None,
    fit_arc: Annotated[
        bool, typer.Option(help="Shrink the input arc to meet the excursion target")
    ] = True,
    target_excursion_deg: Annotated[
        float, typer.Option(help="Ceiling for the worst joint excursion")
    ] = 22.0,
    min_flexure_mm: Annotated[
        float,
        typer.Option(help="Minimum drawn flexure width; 0 draws them at true scale"),
    ] = 1.6,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the scene as JSON")
    ] = None,
) -> None:
    """Write a self-contained HTML viewer of a design over its input arc.

    The output needs nothing to open: no server, no install, no network. Every
    state is precomputed here and baked into the file, so the slider steps
    through real solver output rather than anything the browser worked out.
    """
    from cmtool.metrics.paths import read_path_csv
    from cmtool.viz.html import write_html
    from cmtool.viz.scene import build_scene

    linkage = Linkage.from_json(linkage_json)
    mech = _convert_for_view(
        linkage,
        material=material,
        printer=printer,
        thickness_mm=thickness_mm,
        fit_arc=fit_arc,
        target_excursion_deg=target_excursion_deg,
    )
    include = tuple(s.strip() for s in solvers.split(",") if s.strip())
    measured = read_path_csv(measured_csv) if measured_csv is not None else None

    if "fea" in include:
        console.print(f"solving {steps} states with the beam FEA; this takes a few seconds")
    scene = build_scene(mech, n_steps=steps, measured=measured, include=include)

    path = write_html(scene, html_out, min_flexure_mm=min_flexure_mm)
    size_kb = path.stat().st_size / 1024.0

    table = Table(title=f"{scene.name} viewer", show_header=False, box=None)
    table.add_row("states", str(scene.n_frames))
    table.add_row("models", ", ".join(scene.models))
    table.add_row(
        "input arc (deg)",
        f"{scene.input_angles_deg[0]:.2f} -> {scene.input_angles_deg[-1]:.2f}",
    )
    for comparison in scene.comparisons:
        mean = comparison["mean_mm"]
        table.add_row(
            f"{comparison['a']} vs {comparison['b']} (mm)",
            ("mean -" if mean is None else f"mean {mean:.3f}")
            + f", Frechet {comparison['frechet_mm']:.3f}",
        )
    table.add_row("file", f"{path} ({size_kb:.0f} kB)")
    console.print(table)

    for note in scene.notes:
        console.print(f"[dim]{note}[/]")
    caveat = scene.caveat
    if caveat:
        console.print(f"[yellow]{caveat}[/]")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(scene.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        console.print(f"wrote {json_out}")


@app.command(name="figures")
def figures_cmd(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("docs/figures"),
    designs: Annotated[
        str | None,
        typer.Option(help="Comma-separated linkage JSON files; default is the three pilots"),
    ] = None,
    steps: Annotated[int, typer.Option(help="States per design across the input arc")] = 41,
    measured_csv: Annotated[
        Path | None, typer.Option("--measured", help="Tracked coupler path CSV")
    ] = None,
    torque_csv: Annotated[
        Path | None, typer.Option("--torque", help="Filled-in torque measurement template")
    ] = None,
    uncertainty_json: Annotated[
        Path | None,
        typer.Option("--uncertainty", help="Report from 'cmtool uncertainty --json'"),
    ] = None,
    only: Annotated[
        str | None, typer.Option(help="Comma-separated figure names; default is all of them")
    ] = None,
    theme: Annotated[str, typer.Option(help="light or dark")] = "light",
    column: Annotated[
        str,
        typer.Option(help="Page width to lay out for: single (one journal column) or double"),
    ] = "single",
    formats: Annotated[str, typer.Option(help="Comma-separated: png, svg, pdf")] = "png",
    material: Annotated[str, typer.Option(help="Material config name")] = "PLA",
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
) -> None:
    """Regenerate every README and paper figure in one command.

    Figures whose measurements do not exist yet are drawn without that series and
    labelled, or skipped with a reason -- never filled in with a stand-in. Pass
    --measured, --torque and --uncertainty as those measurements arrive and the
    same command produces the finished set.

    Every figure is laid out at its final printed width -- one journal column by
    default -- and carries a dash pattern and a marker per series as well as a
    colour, so it survives a greyscale print and a colour-blind reader.
    """
    from cmtool.viz.figures import DEFAULT_DESIGNS, build_inputs, generate_all

    sources = (
        tuple(d.strip() for d in designs.split(",") if d.strip()) if designs else DEFAULT_DESIGNS
    )
    suffixes = tuple(f.strip().lstrip(".") for f in formats.split(",") if f.strip())
    wanted = tuple(n.strip() for n in only.split(",") if n.strip()) if only else None

    console.print(
        f"solving {len(sources)} design(s) x {steps} states with the beam FEA; "
        "this takes a few seconds each"
    )
    inputs = build_inputs(
        sources,
        n_steps=steps,
        measured_csv=measured_csv,
        torque_csv=torque_csv,
        uncertainty_json=uncertainty_json,
        material=material,
        printer=printer,
    )
    results = generate_all(out, inputs, theme=theme, formats=suffixes, only=wanted, column=column)

    table = Table(title=f"figures -> {out}")
    table.add_column("figure")
    table.add_column("status")
    table.add_column("note")
    for result in results:
        if result.produced:
            status = "[green]written[/]"
            note = "missing: " + ", ".join(result.missing) if result.missing else ""
        else:
            status = "[yellow]skipped[/]"
            note = result.reason or ""
        table.add_row(result.name, status, note)
    console.print(table)
    console.print(f"manifest: {Path(out) / 'figures.json'}")

    caveat = inputs.provenance.caveat()
    if caveat:
        console.print(f"[yellow]{caveat}[/]")


@app.command(name="prbm-study")
def prbm_study_cmd(
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the report")] = None,
    ratios: Annotated[
        str, typer.Option(help="Comma-separated tip load ratios P*L/M")
    ] = "0,0.05,0.1,0.2,0.5,1,2,5",
) -> None:
    """Fit PRBM constants to the beam FEA and recommend a long-segment variant.

    Howell gives different constants for different end loadings. Which one applies
    to a flexure joint is an empirical question, and this answers it with our own
    solver rather than by assumption.
    """
    from cmtool.solvers.prbm_study import recommend_variant, sweep

    values = tuple(float(v) for v in ratios.split(",") if v.strip())
    fits = sweep(values)

    table = Table(title="PRBM constants fitted to the beam FEA")
    table.add_column("load ratio P*L/M", justify="right")
    table.add_column("gamma", justify="right")
    table.add_column("pivot from root", justify="right")
    table.add_column("K / (EI/L)", justify="right")
    table.add_column("fit rms / L", justify="right")
    for fit in fits:
        table.add_row(
            f"{fit.load_ratio:.2f}",
            f"{fit.gamma:.4f}",
            f"{fit.pivot_fraction:.4f}",
            f"{fit.stiffness_multiple:.4f}",
            f"{fit.fit_rms_over_length:.2e}",
        )
    console.print(table)

    report = recommend_variant()
    console.print(f"[green]recommended variant: {report['recommended']}[/]")
    console.print(report["reason"])

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sweep": [f.to_dict() for f in fits], **report}
        json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        console.print(f"wrote {json_out}")


@app.command()
def torque(
    linkage_json: Annotated[Path, typer.Argument(help="Linkage JSON file")],
    measured_csv: Annotated[
        Path | None, typer.Option("--measured", help="Filled-in torque template")
    ] = None,
    template_out: Annotated[
        Path | None, typer.Option("--template", help="Write a blank data template here")
    ] = None,
    printer: Annotated[str, typer.Option(help="Printer config name")] = "kobra2_neo",
    thickness_mm: Annotated[float | None, typer.Option(help="Flexure thickness")] = None,
    steps: Annotated[int, typer.Option(help="Samples across the input arc")] = 21,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the report")] = None,
) -> None:
    """Compare a measured input-torque curve against the PRBM and beam FEA.

    Torque is the measurement that tests the stiffness model. The coupler path
    does not: with a prescribed input and no external load it is fixed by geometry,
    so it is the same whatever the flexures are made of.
    """
    from cmtool.api import convert as convert_api
    from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism
    from cmtool.convert.arc import ArcFitError, fit_input_arc
    from cmtool.metrics import (
        compare,
        read_measurements,
        write_scale_template,
        write_weight_template,
    )

    linkage = Linkage.from_json(linkage_json)
    arc: tuple[float, float] | None
    try:
        arc = fit_input_arc(linkage, max_joint_excursion_deg=22.0).input_range_deg
    except ArcFitError:
        arc = linkage.input_range_deg

    mech = convert_api(linkage, printer=printer, input_range_deg=arc, thickness_mm=thickness_mm)
    _, layout = build_mechanism(mech, spec=MechanismCadSpec(), printer=None)

    if template_out is not None:
        weights = write_weight_template(
            template_out,
            hole_radius_mm=layout.force_radius_mm,
            printer=printer,
        )
        scale = write_scale_template(
            template_out.with_name(template_out.stem + "_scale.csv"),
            lever_radius_mm=layout.force_radius_mm,
            input_range_deg=mech.input_range_deg,
            printer=printer,
        )
        console.print(
            f"wrote {weights} (mode A, dead weight over a pulley; lever hole at "
            f"{layout.force_radius_mm:.2f} mm radius)"
        )
        console.print(f"wrote {scale} (mode B, spring scale; backup rig)")

    prbm = simulate(mech, solver="prbm", n_steps=steps)
    fea = simulate(mech, solver="beam_fea", n_steps=steps)

    predictions: dict[str, tuple[Any, Any]] = {}
    for name, solved in (("prbm", prbm), ("beam_fea", fea)):
        if solved.input_torque_nmm is None:
            console.print(f"[red]{name} produced no torque curve[/]")
            raise typer.Exit(code=1)
        predictions[name] = (solved.input_angles_deg, solved.input_torque_nmm)
    summary = Table(title=f"predicted input torque for {linkage.name}")
    summary.add_column("model")
    summary.add_column("peak |T| (N*mm)", justify="right")
    summary.add_column("peak force at lever (N)", justify="right")
    for name, (_, values) in predictions.items():
        peak = float(max(abs(v) for v in values))
        summary.add_row(
            name,
            f"{peak:.2f}",
            f"{peak / max(layout.force_radius_mm, 1e-9):.3f}",
        )
    console.print(summary)
    peaks = {name: float(max(abs(v) for v in values)) for name, (_, values) in predictions.items()}
    console.print(
        "[dim]the two models disagree on torque by "
        f"{abs(peaks['prbm'] - peaks['beam_fea']):.1f} N*mm at peak, which is what the "
        "measurement resolves[/]"
    )

    if measured_csv is None:
        console.print("[yellow]no --measured file given; predictions only[/]")
        return

    readings = read_measurements(measured_csv)
    if not readings:
        console.print(f"[red]no filled-in rows found in {measured_csv}[/]")
        raise typer.Exit(code=1)

    comparison = compare(readings, predictions)
    table = Table(title=f"measured vs predicted ({comparison.n_readings} readings)")
    table.add_column("model")
    table.add_column("mean |err| (N*mm)", justify="right")
    table.add_column("max |err| (N*mm)", justify="right")
    table.add_column("mean rel err", justify="right")
    table.add_column("bias (N*mm)", justify="right")
    for name, errors in comparison.model_errors.items():
        table.add_row(
            name,
            f"{errors['mean_abs_error_nmm']:.2f}",
            f"{errors['max_abs_error_nmm']:.2f}",
            f"{errors['mean_relative_error'] * 100:.1f}%",
            f"{errors['bias_nmm']:+.2f}",
        )
    console.print(table)
    if comparison.hysteresis_nmm is not None and comparison.hysteresis_fraction is not None:
        console.print(
            f"hysteresis (loading vs unloading): {comparison.hysteresis_nmm:.2f} N*mm "
            f"({comparison.hysteresis_fraction * 100:.1f}% of peak) - a material property, "
            "not a model error"
        )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(comparison.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        console.print(f"wrote {json_out}")


def _layout_from_report(report_json: Path) -> tuple[Any, tuple[float, float]]:
    """Build a marker layout from an exported mechanism report."""
    from cmtool.vision import MarkerLayout

    data = json.loads(Path(report_json).read_text(encoding="utf-8"))
    layout_data = data["layout"]
    attachment = layout_data.get("attachment_points", {})
    pivot = (0.0, 0.0)
    for points in attachment.values():
        for name, point in points.items():
            if name == data["conversion"].get("input_joint", "A"):
                pivot = (float(point[0]), float(point[1]))
                break

    layout = MarkerLayout.for_mechanism(
        fiducial_pads_mm=[tuple(p) for p in layout_data["fiducial_pads_mm"]],
        lever_pad_mm=tuple(layout_data["lever_pad_mm"])
        if layout_data.get("lever_pad_mm")
        else None,
        coupler_pad_mm=tuple(layout_data["coupler_pad_mm"])
        if layout_data.get("coupler_pad_mm")
        else None,
    )
    return layout, pivot


@app.command()
def markers(
    mechanism_json: Annotated[Path, typer.Argument(help="Exported *_mechanism.json")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("out/markers"),
    pixels_per_mm: Annotated[float, typer.Option(help="Print resolution")] = 12.0,
) -> None:
    """Write printable ArUco sheets for a mechanism's marker pads."""
    from cmtool.vision import write_marker_sheets

    layout, _ = _layout_from_report(mechanism_json)

    collisions = layout.overlaps()
    for first, second, gap in collisions:
        console.print(
            f"[red]pads {first} and {second} are {gap:.1f} mm apart[/] - markers printed "
            "on top of each other will simply not be detected"
        )

    written = write_marker_sheets(layout, out, pixels_per_mm=pixels_per_mm)
    for name, path in written.items():
        pad = layout.pad(name)
        console.print(
            f"{name}: ids {list(pad.marker_ids)}, {pad.marker_size_mm:.0f} mm markers "
            f"-> {path.name}"
        )
    (Path(out) / "layout.json").write_text(
        json.dumps(layout.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    console.print(
        "[yellow]print at 100 percent scale and measure one marker with calipers before "
        "sticking anything down: the millimetre layout assumes the printed size[/]"
    )


@app.command()
def calibrate(
    images_dir: Annotated[Path, typer.Argument(help="Directory of checkerboard images")],
    out: Annotated[Path, typer.Option(help="Where to write the calibration")] = Path(
        "out/calibration.json"
    ),
    pattern: Annotated[str, typer.Option(help="Inner corners, e.g. 9x6")] = "9x6",
    square_mm: Annotated[float, typer.Option(help="Checkerboard square size")] = 10.0,
    glob: Annotated[str, typer.Option(help="Image filename pattern")] = "*.jpg",
) -> None:
    """Calibrate the camera from checkerboard images."""
    import cv2

    from cmtool.vision import CalibrationError
    from cmtool.vision import calibrate as run_calibration

    cols, rows = (int(v) for v in pattern.lower().split("x"))
    files = sorted(Path(images_dir).glob(glob))
    if not files:
        console.print(f"[red]no images matching {glob} in {images_dir}[/]")
        raise typer.Exit(code=1)

    images = [cv2.imread(str(f)) for f in files]
    try:
        result = run_calibration(
            [i for i in images if i is not None],
            pattern=(cols, rows),
            square_size_mm=square_mm,
        )
    except CalibrationError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"calibrated from {result.n_images} of {len(files)} images, "
        f"reprojection rms {result.reprojection_rms_px:.3f} px"
    )
    console.print(f"wrote {result.save(out)}")


@app.command()
def track(
    mechanism_json: Annotated[Path, typer.Argument(help="Exported *_mechanism.json")],
    source: Annotated[Path, typer.Argument(help="Video file or directory of frames")],
    out: Annotated[Path, typer.Option(help="Where to write the measured path")] = Path(
        "out/measured_path.csv"
    ),
    calibration_json: Annotated[
        Path | None, typer.Option("--calibration", help="Camera calibration")
    ] = None,
    stride: Annotated[int, typer.Option(help="Use every Nth video frame")] = 1,
    glob: Annotated[str, typer.Option(help="Frame filename pattern")] = "*.png",
    json_out: Annotated[Path | None, typer.Option("--json", help="Write a summary")] = None,
) -> None:
    """Track a printed mechanism from video or frames, and write its measured path."""
    from cmtool.vision import CameraCalibration, frames_from_directory, frames_from_video
    from cmtool.vision import track as run_tracking

    layout, pivot = _layout_from_report(mechanism_json)
    calibration = CameraCalibration.load(calibration_json) if calibration_json is not None else None
    if calibration is None:
        console.print(
            "[yellow]no --calibration given: lens distortion is not projective, so a "
            "homography cannot absorb it. Uncorrected, it biases positions across the "
            "frame in a way that mimics a real path deviation.[/]"
        )

    frames = (
        frames_from_directory(source, pattern=glob)
        if source.is_dir()
        else frames_from_video(source, stride=stride)
    )
    result = run_tracking(frames, layout, pivot_mm=pivot, calibration=calibration)

    summary = result.summary()
    table = Table(title=f"tracked {source.name}", show_header=False, box=None)
    table.add_row("frames", str(summary["n_frames"]))
    table.add_row("tracked", f"{summary['n_tracked']} ({summary['tracked_fraction'] * 100:.0f}%)")
    if summary["homography_rms_px_mean"] is not None:
        table.add_row(
            "homography rms (px)",
            f"{summary['homography_rms_px_mean']:.3f} mean, "
            f"{summary['homography_rms_px_max']:.3f} worst",
        )
    console.print(table)
    for reason, count in summary["failures"].items():
        console.print(f"  [yellow]{count} frames: {reason}[/]")

    if result.n_tracked == 0:
        console.print("[red]nothing tracked; check lighting, focus and marker placement[/]")
        raise typer.Exit(code=1)

    console.print(f"wrote {result.write_csv(out)}")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        console.print(f"wrote {json_out}")


@app.command()
def uncertainty(
    static_csv: Annotated[
        Path | None, typer.Option("--static", help="Tracked CSV of a stationary scene")
    ] = None,
    circle_csv: Annotated[
        Path | None, typer.Option("--circle", help="Tracked CSV of a rigid bar on a pin")
    ] = None,
    radius_mm: Annotated[
        float | None, typer.Option(help="Caliper-measured radius of that bar")
    ] = None,
    signal_mm: Annotated[float, typer.Option(help="Signal to resolve, in mm")] = 0.46,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the report")] = None,
) -> None:
    """Report tracking uncertainty against the signal it has to resolve.

    This is the Phase A go/no-go. Give it a static take, a known-motion take, or
    both; the verdict takes the worst of what it is given.
    """
    import numpy as np

    from cmtool.vision import GoNoGoReport, jitter, known_motion

    def load(path: Path) -> Any:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        return np.array(
            [[float(r["coupler_x_mm"]), float(r["coupler_y_mm"])] for r in rows], dtype=float
        )

    estimates = []
    if static_csv is not None:
        estimates.append(jitter(load(static_csv)))
    if circle_csv is not None:
        estimates.append(known_motion(load(circle_csv), expected_radius_mm=radius_mm))
    if not estimates:
        console.print("[red]give --static, --circle, or both[/]")
        raise typer.Exit(code=2)

    report = GoNoGoReport(estimates=estimates, signal_mm=signal_mm)
    table = Table(title="tracking uncertainty")
    table.add_column("method")
    table.add_column("sigma (mm)", justify="right")
    table.add_column("worst (mm)", justify="right")
    table.add_column("samples", justify="right")
    for estimate in report.estimates:
        table.add_row(
            estimate.method,
            f"{estimate.sigma_mm:.4f}",
            f"{estimate.max_deviation_mm:.4f}",
            str(estimate.n_samples),
        )
    console.print(table)

    colour = {"go": "green", "marginal": "yellow", "no-go": "red"}[report.verdict]
    console.print(f"[{colour}]VERDICT: {report.verdict.upper()}[/]")
    console.print(report.explain())

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        console.print(f"wrote {json_out}")


if __name__ == "__main__":
    app()
