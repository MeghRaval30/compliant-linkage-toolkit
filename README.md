# cmtool — rigid-to-compliant planar linkage toolkit

Converts rigid planar linkages into compliant (flexure-based) mechanisms, simulates them,
exports printable CAD, and measures the printed parts from video — so that simulated
designs can be compared against physical reality on the same footing.

It is the software half of an open benchmark dataset pairing rigid linkages with their
compliant counterparts, including paths measured on real FDM-printed parts.

**Research question:** how well do simulated compliant-mechanism designs (pseudo-rigid-body
model and FEA) transfer to FDM-printed parts, and can a learned model close that
simulation-to-reality gap?

> **Status: Phase A pilot, milestone A5.** Rigid kinematics, flexure sizing and
> feasibility, design search, the PRBM solver, the nonlinear beam FEA, printable CAD for
> coupons and mechanisms, and the camera measurement pipeline all work and are validated.
> What remains is A6: print, film, and produce the demo materials.
>
> **No physical measurements exist yet.** Every material constant and the minimum printable
> flexure thickness are explicitly flagged placeholders, so every feasibility verdict the
> toolkit currently produces is marked *not a physical prediction*. The test coupons below
> are what replaces them.

New here? [docs/walkthrough.md](docs/walkthrough.md) explains each part in plain language —
what it solves, the equation, why, and where it stops being true.
Picking the work back up? [docs/HANDOVER.md](docs/HANDOVER.md) is the current state of
play: what is built, what is still unmeasured, and what happens next.

## Install

```bash
uv sync --extra dev --extra cad
```

## Print these first

Four coupons measure the numbers everything else depends on. They are committed under
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
- **`strain_mandrels` + `strain_strips`** — strips wrapped around posts of decreasing radius
  to measure the allowable strain, `(t/2)/(R + t/2)`, spanning about 0.008 to 0.057. Check
  whitening, cracking and permanent set on one bend and again after ten cycles;
  `strain_test_template.csv` has a row per combination with the strain pre-filled. The
  strips are printed as upright walls and the posts are vertical, so the strip bends
  **in-plane** like a real flexure — bent the other way, the test would measure interlayer
  adhesion instead.

Each print sheet carries the Bambu Studio settings and the metadata checklist to fill in at
the machine. The setting that matters most is the **Arachne** wall generator — the classic
generator will thin, distort or silently drop a sub-2-perimeter wall, which is every flexure
in this project.

## Use

```python
from cmtool import Linkage, simulate, convert, fit_input_arc

mech = Linkage.from_json("examples/designs/fb_02_0052.json")

arc = fit_input_arc(mech, max_joint_excursion_deg=22.0)   # flexures cannot rotate far
cm = convert(mech, material="PLA", printer="bambu_a1", input_range_deg=arc.input_range_deg)

cm.feasibility.feasible          # can this be built out of flexures?
cm.feasibility.binding_joint     # which joint is limiting the design
cm.sizing["B"].limit_reason      # and why, in words
cm.feasibility.prbm_models       # which PRBM model each joint needs (metadata, not a filter)

prbm = simulate(cm, solver="prbm", n_steps=61)      # milliseconds
fea = simulate(cm, solver="beam_fea", n_steps=61)   # seconds, no PRBM assumptions

prbm.input_torque_nmm            # torque to hold each input angle
fea.flexure_strain["B"]          # strain in joint B's flexure along the arc
```

From the command line:

```bash
cmtool coupons --out out/coupons                 # test coupons + print sheets
cmtool design --seed 2 --out out/designs         # search for buildable four-bars
cmtool convert examples/designs/fb_02_0052.json  # per-joint feasibility table
cmtool export examples/designs/fb_02_0052.json   # printable mechanism + print sheet
cmtool simulate examples/fourbar.json --csv out/rigid_path.csv
cmtool prbm-study                                # which PRBM variant fits, from the FEA
cmtool torque examples/designs/fb_02_0052.json   # predicted torque, and measured comparison
cmtool markers examples/mechanisms/fb_02_0052_mechanism.json   # printable ArUco sheets
cmtool calibrate photos/checkerboard --out out/calibration.json
cmtool track examples/mechanisms/fb_02_0052_mechanism.json clip.mp4 --calibration out/calibration.json
cmtool uncertainty --circle out/measured_path.csv --radius-mm 40   # the go/no-go
```

## The design rule that governs everything

Two constraints act on a flexure at once, and they pull opposite ways. Strain wants it
long, `L ≥ t·θ/(2·ε_allow)`. Geometry wants it short — a flexure is a necked-down part of a
link, so rigid material must remain at each end — `L ≤ f·l`, with `f = 0.8` and `l` the
shorter adjacent link. Eliminating `L`:

```
θ_max = 2·f·l·ε_allow / (t·SF)
```

The largest usable joint rotation is set by **the length of the neighbouring link**, not by
anything about the flexure alone. Short links cannot carry large rotations at any thickness,
so a classic crank-rocker with a stubby crank is exactly the wrong shape for a compliant
mechanism.

**Pseudo-rigid-body validity is not in that bound, deliberately.** The length ratio
`L/l` selects which PRBM model represents a flexure — the centre-pivot model below 0.1,
Howell's long-segment model above — and is recorded on every sample. It never rejects a
design. Phase C's fidelity map is a map of where the simple model fails, so filtering those
designs out would hide the result it exists to show.

That distinction is not academic. While the 0.1 ratio was being enforced as a constraint,
the A1 pilot four-bar was reported unbuildable. It is not: under the real rule it builds
fine, three of its four joints simply need the long-segment model. The old rule was also 8×
tighter than physics requires. The designs in [`examples/designs/`](examples/designs/) and
their printable parts in [`examples/mechanisms/`](examples/mechanisms/) come from the
corrected search.

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
  solvers/     rigid | prbm | beam_fea -> one SimulationResult type, plus
               exact references and the PRBM-variant study
  flexures/    geometry, PRBM models (small-length and Howell), own strain model
  convert/     arc fitting, flexure sizing, feasibility, design search
  cad/         coupons, mechanism CAD, printability checks, STEP/STL, print sheets
  materials/   materials and printers loaded from configs with provenance
  metrics/     torque measurement templates and model comparison
  vision/      ArUco layout, calibration, plane map, tracking, uncertainty
  schema/      pydantic models; JSON Schema is generated from them
  vision/ metrics/ dataset/ viz/   (later milestones)
```

Physics conventions and validity limits: [docs/physics.md](docs/physics.md).

## Hardware

| | |
|---|---|
| Primary printer | **Anycubic Kobra 2 Neo** — all Phase A and B parts, to keep printer variation out of the measurement. Took over from the A1 on 2026-09-20 when the A1 became unavailable; nothing had been printed yet, so the switch cost no data |
| Secondary | Bambu Lab A1, reserved for a possible Phase C printer comparison |
| Both | 0.4 mm nozzle, 0.2 mm layer height, 180 × 180 mm design envelope so parts fit either bed |
| Material | eSun PLA (product line, colour and lot still to be recorded from the spool label) |
| Orientation | Flat on the bed, mechanism plane parallel to it, so flexures bend in-plane |
| Slicer | OrcaSlicer for the Kobra, Bambu Studio for the A1; both print sheets carry the settings, and **Arachne** is the one that matters |

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
