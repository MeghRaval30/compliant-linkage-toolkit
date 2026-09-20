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
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cmtool import __version__
from cmtool.api import available_solvers, simulate
from cmtool.core.graph import Linkage
from cmtool.core.provenance import git_commit

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
    printer: Annotated[str, typer.Option(help="Printer config name")] = "bambu_a1",
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
        "L prbm max",
        "peak strain",
        "util",
    ):
        table.add_column(column, justify="right")
    table.add_column("ok")

    for name, sized in mech.sizing.items():
        table.add_row(
            name,
            f"{sized.excursion_deg:.2f}",
            f"{sized.max_bend_deg:.2f}",
            f"{sized.geometry.length_mm:.2f}",
            f"{sized.min_length_strain_mm:.2f}",
            f"{sized.max_length_prbm_mm:.2f}",
            f"{sized.strain.peak_strain:.5f}",
            f"{sized.utilisation:.2f}",
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
    printer: Annotated[str, typer.Option(help="Printer config name")] = "bambu_a1",
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

    for stem, solid in solids.items():
        min_feature = (
            min(coupon_set.flexure_spec.thicknesses_mm) if stem == "flexure_coupon" else None
        )
        check = check_printability(
            solid,
            printer_cfg,
            min_feature_mm=min_feature,
            allow_below_minimum=(stem == "flexure_coupon"),
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
    printer: Annotated[str, typer.Option(help="Printer config name")] = "bambu_a1",
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


@app.command()
def export() -> None:
    """Export CAD for a full compliant mechanism (milestone A2, in progress)."""
    _not_yet("A2", "monolithic mechanism CAD export")


@app.command()
def track() -> None:
    """Track a printed mechanism from video (milestone A5)."""
    _not_yet("A5", "camera tracking")


if __name__ == "__main__":
    app()
