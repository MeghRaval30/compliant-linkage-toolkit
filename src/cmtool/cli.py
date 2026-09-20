"""Command-line interface.

Batch work goes through this CLI so that runs are scriptable and logged::

    cmtool info
    cmtool simulate examples/fourbar.json --solver rigid --steps 91 --csv out/path.csv

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
    table = Table(title="cmtool", show_header=False, box=None)
    table.add_row("version", __version__)
    table.add_row("code commit", git_commit())
    table.add_row("solvers", ", ".join(available_solvers()))

    from cmtool.kinematics import KINEMATICS

    table.add_row("kinematics", ", ".join(KINEMATICS.names()))
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


@app.command()
def convert() -> None:
    """Convert a rigid linkage into a compliant one (milestone A2)."""
    _not_yet("A2", "conversion")


@app.command()
def generate() -> None:
    """Generate random feasible linkage samples (Phase B)."""
    _not_yet("B", "sample generation")


@app.command()
def export() -> None:
    """Export CAD (STEP/STL) for a compliant design (milestone A2)."""
    _not_yet("A2", "CAD export")


@app.command()
def track() -> None:
    """Track a printed mechanism from video (milestone A5)."""
    _not_yet("A5", "camera tracking")


if __name__ == "__main__":
    app()
