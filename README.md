# cmtool — rigid-to-compliant planar linkage toolkit

Converts rigid planar linkages into compliant (flexure-based) mechanisms, simulates them,
exports printable CAD, and measures the printed parts from video — so that simulated
designs can be compared against physical reality on the same footing.

It is the software half of an open benchmark dataset pairing rigid linkages with their
compliant counterparts, including paths measured on real FDM-printed parts.

**Research question:** how well do simulated compliant-mechanism designs (pseudo-rigid-body
model and FEA) transfer to FDM-printed parts, and can a learned model close that
simulation-to-reality gap?

> **Status: Phase A pilot, milestone A2.** Rigid four-bar kinematics, flexure sizing with
> feasibility checking, design search and CAD export all work and are validated. PRBM (A3),
> beam FEA (A4) and camera tracking (A5) are next.
>
> **No physical measurements exist yet.** Every material constant and the minimum printable
> flexure thickness are explicitly flagged placeholders, so every feasibility verdict the
> toolkit currently produces is marked *not a physical prediction*. The test coupons below
> are what replaces them.

New here? [docs/walkthrough.md](docs/walkthrough.md) explains each part in plain language —
what it solves, the equation, why, and where it stops being true.

## Install

```bash
uv sync --extra dev --extra cad
```

## Print these first

Two coupons measure the numbers everything else depends on. They are committed under
[`examples/coupons/`](examples/coupons/) with print sheets, or regenerate them with:

```bash
cmtool coupons --out out/coupons
```

- **`flexure_coupon`** (150 × 40 × 6 mm) — strips at 0.4, 0.5, 0.6, 0.8 and 1.0 mm, sharing
  one rail, identified by pip count. Finds the minimum flexure thickness the printer can
  actually produce.
- **`cantilever_coupon`** (102 × 100 mm) — six strips at two thicknesses for a deflection
  modulus test. Two thicknesses deliberately: if a 1 mm strip (nearly all perimeter) and a
  2 mm strip (with infill) give different moduli, then a modulus measured on a thick strip
  does not transfer to a 0.5 mm flexure.

Each print sheet carries the Bambu Studio settings and the metadata checklist to fill in at
the machine. The setting that matters most is the **Arachne** wall generator — the classic
generator will thin, distort or silently drop a sub-2-perimeter wall, which is every flexure
in this project.

## Use

```python
from cmtool import Linkage, simulate, convert, fit_input_arc

mech = Linkage.from_json("examples/designs/fb_01_0077.json")

arc = fit_input_arc(mech, max_joint_excursion_deg=22.0)   # flexures cannot rotate far
cm = convert(mech, material="PLA", printer="bambu_a1", input_range_deg=arc.input_range_deg)

cm.feasibility.feasible          # can this be built out of flexures?
cm.feasibility.binding_joint     # which joint is limiting the design
cm.sizing["B"].limit_reason      # and why, in words
```

From the command line:

```bash
cmtool design --seed 1 --out out/designs        # search for buildable four-bars
cmtool convert examples/designs/fb_01_0077.json # per-joint feasibility table
cmtool simulate examples/fourbar.json --csv out/rigid_path.csv
```

## The design rule that governs everything

Two constraints act on a small-length flexural pivot at once, and they pull opposite ways.
Strain wants a long flexure, `L ≥ t·θ/(2·ε_allow)`. Validity of the pseudo-rigid-body model
wants a short one, `L ≤ r·l`, where `l` is the shorter adjacent link. Eliminating `L`:

```
θ_max = 2·r·l·ε_allow / (t·SF)
```

The largest usable joint rotation is set by **the length of the neighbouring link**, not by
anything about the flexure alone. Short links cannot carry large rotations at any thickness,
so a classic crank-rocker with a stubby crank is exactly the wrong shape for a compliant
mechanism.

This is not theoretical: the pilot four-bar from milestone A1 has an 18 mm input link and
turns out to be unbuildable — it needs a 7.25 mm flexure to survive a 9.2° bend, but validity
caps it at 1.8 mm. `cmtool convert` says so, and names the joint. The three designs in
[`examples/designs/`](examples/designs/) were found by searching for link lengths that work.

Two decisions each buy a factor of two, for free:

- **Print unstressed at mid-arc**, so each flexure swings symmetrically about zero. Halves
  peak strain. Measured on a real design: worst-joint utilisation 9.59 → 4.89.
- **Pivot-matched placement**, because a flexure pivots about its *centre*. Without it every
  effective link length shifts by half a flexure and the coupler path moves before any
  physics is involved — a conversion artefact that would otherwise be mistaken for a
  simulation-to-reality gap.

## Two rules the code enforces

**1. No invented physical data.** Material constants, strain limits and printer tolerances
live in `configs/`, each carrying a `status`. A `placeholder` is not a measurement: using one
records its name in the result's provenance, warns, and marks the result `is_physical = False`.
`CMTOOL_STRICT_DATA=1` turns that into a hard error.

**2. Limited input arcs only.** A flexure cannot rotate continuously, so there is no
"full revolution" default anywhere. Every simulation states its arc explicitly, and every
joint's excursion over that arc is recorded.

## Architecture

A linkage is a **graph**: bodies are nodes, joints are edges, with ground, input and output
tags. Solvers claim a topology by inspecting the graph (mobility, loop count), never by
reading a type string. Flexure types, materials, solvers and design strategies are all
plug-in registries.

Only `cmtool/kinematics/fourbar.py` and its tests assume a four-bar. Adding a five-bar,
six-bar or a new flexure type is an addition, not a rewrite — see
[docs/extending.md](docs/extending.md).

```
src/cmtool/
  core/        linkage graph, units, registries, provenance, config, quantities
  kinematics/  position solvers, dispatched by topology
  solvers/     rigid | prbm | beam_fea | solid_fea -> one SimulationResult type
  flexures/    geometry + PRBM stiffness + each type's OWN strain model
  convert/     arc fitting, flexure sizing, feasibility, design search
  cad/         coupons, printability checks, STEP/STL export, print sheets
  materials/   materials and printers loaded from configs with provenance
  schema/      pydantic models; JSON Schema is generated from them
  vision/ metrics/ dataset/ viz/   (later milestones)
```

Physics conventions and validity limits: [docs/physics.md](docs/physics.md).

## Hardware

| | |
|---|---|
| Primary printer | **Bambu Lab A1** — all Phase A and B parts, to keep printer variation out of the measurement |
| Secondary | Anycubic Kobra 2 Neo, reserved for a possible Phase C printer comparison |
| Both | 0.4 mm nozzle, 0.2 mm layer height, 180 × 180 mm design envelope so parts fit either bed |
| Material | eSun PLA (product line, colour and lot still to be recorded from the spool label) |
| Orientation | Flat on the bed, mechanism plane parallel to it, so flexures bend in-plane |

## Development

```bash
uv run pytest                 # tests, including solver validation
uv run pytest -m validation   # only the analytical / closed-form checks
uv run pytest -m "not slow"   # skip the slower search tests
uv run ruff check src tests
uv run mypy
uv run pre-commit install
```

Validation tests check against analytical results rather than stored outputs: exact
hand-computed configurations, closed-form special cases (a parallelogram's coupler must
never rotate), independent numerical formulations, and beam-theory identities verified two
different ways.

## Licence

Code: MIT (`LICENSE`). Data and dataset releases: CC BY 4.0 (`LICENSE-DATA`).
