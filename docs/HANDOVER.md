# Handover: state of the project as of 2026-09-21

Written so a fresh session (or a new team member) can pick this up without reading the
whole history. Repo: <https://github.com/MeghRaval30/compliant-linkage-toolkit>, branch
`main`, CI green on Ubuntu + Windows × Python 3.11/3.12.

**486 tests pass; lint (ruff), formatting and types (mypy) are clean.** Ten commits,
`A0+A1` through `A6 (software half)`.

---

## 1. What the project is

An open-source Python toolkit plus a benchmark dataset pairing **rigid planar linkages**
with their **compliant (flexure-based) counterparts**, including paths measured on real
FDM-printed parts.

**Research question:** how well do simulated compliant-mechanism designs (PRBM and FEA)
transfer to FDM-printed parts, and can a learned model close that gap?

Deadlines: **Phase A must finish by early November 2026** (driven by summer-internship
applications, not the research calendar). euspen abstract Dec 2026–Jan 2027, conference
paper Mar 2027, JOSS deferred until after the conference draft.

Scope discipline: planar four-bars only for now, but the architecture is general — solvers
claim a topology by inspecting the graph, and flexures, materials, solvers and design
strategies are plug-in registries.

---

## 2. Milestone status

| Milestone | State | What it delivers |
|---|---|---|
| A0 repo setup | done | package layout, uv, CI, licences, JSON schemas |
| A1 rigid kinematics | done | four-bar position solver, Grashof, transmission angle |
| A2 conversion + coupons | done | flexure sizing, feasibility, design search, test coupons |
| A3 PRBM | done | quasi-static solver, two PRBM models, mechanism CAD |
| A4 beam FEA | done | co-rotational beam FEA, PRBM variant study, torque rig |
| A5 camera | done | ArUco tracking, calibration, uncertainty / go-no-go |
| A6 software | done | HTML viewer, figure scripts, path-distance metrics |
| **A6 physical** | **not started** | **print, measure, film, demo video** |

**Everything remaining in Phase A is physical.** No software work blocks it: the figures
regenerate with the measured series filled in as soon as it exists.

---

## 3. The two rules the code enforces

**No invented physical data.** Every config quantity carries a `status`:
`measured` / `vendor` / `literature` / `design_choice` / `confirmed` / `placeholder`.
A placeholder has `value: null` plus a `provisional_value` that exists only so code can run.
Using one records its name in the result's `provenance.placeholders_used`, emits a
`ProvisionalDataWarning`, and sets `is_physical = False`. `CMTOOL_STRICT_DATA=1` makes it a
hard error. **Never write a number with `status: measured` that was not measured.**

**Limited input arcs only.** A flexure cannot rotate continuously, so there is no
full-revolution default anywhere. `simulate()` refuses to run without an explicit arc, and
every joint's excursion over that arc is recorded.

---

## 4. Findings that shaped the design

These are the non-obvious results. Each cost real work to establish; do not silently
re-litigate them.

### The governing design bound

Two constraints act on a flexure at once. Strain wants it long, `L ≥ t·θ·SF/(2·ε_allow)`.
Geometry wants it short — a flexure is a necked-down part of a link, so rigid material must
remain at each end — `L ≤ f·l` with `f = 0.8`. Eliminating `L`:

```
θ_max = 2·f·l·ε_allow / (t·SF)
```

**Usable joint rotation is set by the adjacent link length**, not by the flexure. Short
links cannot carry large rotations at any thickness, so a crank-rocker with a stubby crank
is the wrong shape for a compliant mechanism.

### PRBM validity is metadata, never a filter

The small-length ratio `L/l` selects which pseudo-rigid-body model applies and is recorded
on every sample. It does **not** reject designs. Phase C's fidelity map is a map of where
the simple model fails, so filtering those designs out would hide the result it exists to
show. This was changed mid-project: while the 0.1 ratio was acting as a constraint, the A1
pilot four-bar was reported unbuildable. It is not — it builds fine, three of its four
joints just need the long-segment model.

### Two PRBM variants, and the FEA picked one

Howell gives different constants for different end loadings, **and they do not share a
stiffness formula**:

```
end force    K = γ·K_Θ·EI/L = 0.85 × 2.65  = 2.25 EI/L
end moment   K =   K_Θ·EI/L =        1.5164 = 1.52 EI/L     ← no γ
```

Multiplying the end-moment constants gives ≈1.11, which looks as though it would erase the
step at the model boundary. It does not.

`cmtool prbm-study` fits γ and K to our own beam FEA across tip load ratios `λ = PL/M`.
A flexure joint is **moment-dominated** (`λ ~ L/l ≈ 0.1`), and at that ratio the end-moment
variant is within **0.5%** on stiffness while end-force is off by **48%**. `active_variant`
is therefore `end_moment`, recorded as `measured`.

The end-moment constants are independently corroborated: the exact circular arc gives
γ → 3/4 and K_Θ → 3/2 analytically in the small-angle limit.

### The pivot is at `(1 − γ)L` from the root, not `γL`

The rigid link of length `γL` runs from the pivot to the tip. Fitting the exact arc with
`(1−γ)L` reproduces the tip path to 2×10⁻⁴ L; the other convention is off by 0.15 L, and at
γ = 0.85 the rigid link cannot even reach the tip. **Pivot matching places the flexure so
whichever model's pivot lands on the rigid joint** — centring is correct only for the
small-length model.

### Stiffness does not move the path

With a prescribed input, one degree of freedom and no external load, the coupler path is
fixed by geometry. Doubling every `K` gives the identical curve and twice the torque. So
**the path cannot test the stiffness model** — only torque can. That is why the torque rig
exists.

### Two free factors of two

- **Print unstressed at mid-arc**: each flexure swings symmetrically about zero, halving
  peak strain. Measured: worst-joint utilisation 9.59 → 4.89.
- **Pivot-matched placement**: without it every effective link length shifts by part of a
  flexure length and the path moves before any physics enters — a conversion artefact that
  would otherwise be mistaken for a sim-to-real gap. `placement="unmatched"` is kept so the
  artefact can be measured, and it now has been: on `fb_02_0052`, whose flexures are
  7.5—8.7 mm, it moves the coupler path **0.567 mm mean / 1.166 mm max** — *larger* than the
  0.451 mm PRBM-to-FEA disagreement on the same design. It scales with flexure length, so
  the earlier ≈0.2 mm figure was for a 4 mm flexure. See `docs/figures/conversion_artefact.png`.

### Model boundary discontinuity

The two models differ by 1.52× in stiffness where they meet. Physical stiffness does not
jump, so samples near the threshold are **flagged** (`near_model_boundary`), never blended.
Blending would be inventing a model.

---

## 5. Architecture

A linkage is a **graph**: bodies are nodes, joints are edges, with ground/input/output tags.
Only `cmtool/kinematics/fourbar.py` and its tests assume a four-bar.

```
src/cmtool/
  core/        graph, units, registries, provenance, config, quantities
  kinematics/  position solvers, dispatched by topology
  solvers/     rigid | prbm | beam_fea, plus beam.py, beam_reference.py, prbm_study.py
  flexures/    geometry, PRBM models (small_length / long_segment variants), strain
  convert/     arc fitting, flexure sizing, feasibility, design search, limits
  cad/         coupons, mechanism CAD, printability, STEP/STL, print sheets
  materials/   materials and printers from configs, with provenance
  metrics/     torque templates (two rigs), model comparison, path distance
  vision/      ArUco layout, calibration, plane map, tracking, uncertainty, synthetic
  viz/         scene model, one-file HTML viewer, palette, the eight figure scripts
  schema/      pydantic models; JSON Schema generated from them and CI-checked
  dataset/     still empty — Phase B
```

### CLI

```
cmtool info | simulate | convert | coupons | design | export
            | prbm-study | torque | markers | calibrate | track | uncertainty
            | view | figures
cmtool generate    # still a stub: Phase B
```

### Validation philosophy

Solvers are checked against **analytical or independently-derived references**, never
against stored outputs of this code. Exact hand-computed configurations, closed-form
special cases, independent numerical formulations, and property-based tests.

Notable: the beam FEA matches an exact circular arc to <10⁻⁴ L with second-order
convergence and machine-precision tip slope, and matches a Bisshopp–Drucker end-load
solution (integrated by quadrature here, not recalled as an elliptic-integral formula) to
<10⁻⁴ L.

---

## 6. Hardware and current settings

| | |
|---|---|
| Primary printer | **Anycubic Kobra 2 Neo** (took over from the A1 on 2026-09-20 when the A1 became unavailable; nothing had been printed, so no data was lost) |
| Secondary | Bambu Lab A1, held for a Phase C printer comparison |
| Both | 0.4 mm nozzle, 0.2 mm layer, 180 × 180 mm design envelope |
| Slicer | OrcaSlicer (Kobra) / Bambu Studio (A1). **Arachne wall generator is the setting that matters** |
| Material | eSun **PLA** |
| Orientation | flat on the bed, mechanism plane parallel to it, flexures bend in-plane |
| Flexure thickness | **held at 0.6 mm** until the coupon says otherwise; switch to 0.4 mm if it prints reliably |
| Fixture | reusable, M3 bolts |

**One machine prints all of Phase A and B.** Switching mid-phase reintroduces the confound
the choice exists to avoid. Decide before the first `print_purpose: data` part.

Every print record carries the `printer` and a `print_purpose` of `trial` or `data`.

---

## 7. What is still a placeholder — the critical path

Nothing the toolkit currently outputs is a physical prediction, because these are unmeasured:

| Quantity | Config | How to measure |
|---|---|---|
| `min_flexure_thickness_mm` | `configs/printer/kobra2_neo.yaml` | flexure coupon |
| `youngs_modulus_MPa` | `configs/materials/pla.yaml` | cantilever coupon |
| `allowable_strain` | `configs/materials/pla.yaml` | strain coupon (mandrels) |
| `density_kg_per_m3` | `configs/materials/pla.yaml` | weigh a printed strip |
| `flow_ratio` | `configs/printer/kobra2_neo.yaml` | slicer flow calibration |
| marker pad mass | `configs/fixture/default.yaml` | jeweller's scale |
| camera / metrology details | `configs/fixture/default.yaml` | fill in from the actual kit |

Filament **product line (PLA vs PLA+), colour and lot** are still blank in
`configs/materials/pla.yaml` — they were to come from the spool label.

**Flow calibration matters more than it looks:** flow error goes straight into flexure
thickness, which enters stiffness as `t³`. A 5% flow error is a 16% stiffness error.

---

## 8. Ready to print — `examples/`

| Folder | Contents |
|---|---|
| `examples/coupons/` | flexure coupon (150×40×6, strips at 0.4–1.0 mm), cantilever coupon (6 strips, 1 and 2 mm), strain mandrels + upright strips, print sheets, `strain_test_template.csv` |
| `examples/designs/` | three pilot four-bars, all joints 19–22°, each limited by a different joint |
| `examples/mechanisms/` | printable STLs (148×142, 93×165, 125×169 mm), print sheets, layouts, both torque templates each |
| `examples/markers/` | printable ArUco sheets + `layout.json` |

Regenerate any of it with `cmtool coupons`, `cmtool design`, `cmtool export`,
`cmtool markers`.

### Measurement rigs

**Torque, mode A (primary), dead weight over a pulley.** The mechanism lies flat so the
pull is horizontal and a kitchen scale (which weighs vertically) cannot read it. Thread from
the 3 mm lever hole → horizontally to a pulley at the table edge → hanging weights. The
thread direction changes as the lever swings, so the applied torque is
`T = W·cross(H − A, û)` and the analysis resolves the moment arm per reading. Record the
pulley position in the mechanism's mm frame. Mode B (spring scale) is the backup.

Predicted peak torque: ~32 N·mm (FEA) vs ~48 N·mm (PRBM) at a 36 mm arm — roughly 90 g vs
135 g of hanging weight. The models differ by far more than kitchen weights resolve.

Loading/unloading gap is reported as **hysteresis plus rig friction**, not hysteresis alone.

**Path, camera.** `cmtool markers` → print at 100% and check one marker with calipers →
`cmtool calibrate` → `cmtool track` → `cmtool uncertainty`.

---

## 9. The Phase A go/no-go

**Is tracking uncertainty clearly smaller than the sim-to-real gap we need to resolve?**

The reference signal is the **0.46 mm mean PRBM-vs-beam-FEA coupler path disagreement**
measured on the pilot designs (`DEFAULT_SIGNAL_MM` in `cmtool/vision/uncertainty.py`).
Thresholds: **5:1 go, 3:1 marginal**.

Two deliberate choices that make the verdict harsher rather than kinder:

- It takes the **worst** of the methods used, not the most complete. A circle fit absorbs
  uniform offset and scale that static jitter never sees; jitter ignores everything
  systematic. Neither contains the other.
- The circle residual is a **shape** check. A wrong scale fits a circle perfectly, so
  `known_motion` wants a caliper-measured radius and combines the radius error with the
  shape residual in quadrature.

**Synthetic numbers are not the rig's accuracy.** The harness recovers shape to ~14 µm but
carries a few tenths of a percent of its own scale bias from the render-then-detect round
trip, present even at zero tilt. It validates the software. **The real number needs real
footage.**

---

## 10. A6 — what is left, in order

1. **Print the flexure and cantilever coupons.** Measure; set
   `min_flexure_thickness_mm` and `youngs_modulus_MPa` with `status: measured` and dates.
2. **Print the strain coupon.** Bend strips over the mandrels, check whitening/cracking/
   permanent set on one bend and after 10 cycles; set `allowable_strain`.
3. **Decide 0.4 vs 0.6 mm flexures.** At 0.4 mm the pilots become fully small-length
   (ratios 0.046–0.080); at 0.6 mm they sit just past the threshold and use the
   long-segment model. If 0.4 prints, regenerate the designs — one command.
4. **Calibrate flow** before any `print_purpose: data` part.
5. **Print the three mechanisms.** 3 copies of at least one design.
6. **Set up the camera and the pulley rig**, print the markers, calibrate.
7. **Run the known-motion ground truth** (rigid bar on a pin, caliper-measured radius) and
   **report the go/no-go explicitly** against 0.46 mm.
8. **Film the mechanisms**, track, measure torque loading and unloading.
9. **A6 deliverables — these are hard deadlines**: README with images, the
   rigid/PRBM/FEA/measured overlay figure, and a 30–60 s demo video. If the go/no-go fails,
   ship them anyway as an honest status report.

### Software that would help, but does not block

- The full PyVista 3D viewer is not written. The HTML viewer and the figures both consume
  a `ViewerScene`, so it is a third renderer of the same data rather than new geometry work.
- `cmtool generate` (Phase B dataset sampling) is still a stub.
- The end-**force** PRBM constants are corroborated only in trend, not to the 2% the
  end-moment ones reach — the FEA sweep does not reach the pure-force limit.
- Flexures are prismatic with sharp corners, matching the model exactly.
  `fillet_radius_mm` exists and defaults to **off** deliberately: let the first prints show
  where they crack before geometry and model are allowed to disagree.

---

## 10b. What the software side of A6 delivers

Both are driven from the CLI and both were built so that nothing has to be redrawn by hand
once the parts exist.

**`cmtool view <design> --html out/view.html`** writes one self-contained file: no server,
no install, no network, no Python on the far end. A slider steps through precomputed beam
FEA states -- nothing is solved in the browser -- with the flexures coloured by strain, the
rigid and PRBM outlines over them as ghosts, and all the coupler paths underneath. About
180 kB for 41 states. A test scans the generated document for any external reference and
fails the build over one.

**`cmtool figures --out docs/figures`** regenerates every README and paper figure in one
command, writing `figures.json` recording which inputs produced each one:

| figure | needs |
|---|---|
| `path_overlay` | rigid / PRBM / FEA / measured paths, plus the separation panel |
| `torque_curves` | PRBM vs FEA torque, measured readings when the CSV has them |
| `strain` | one panel per joint against the allowable |
| `feasibility` | both hard limits per joint, binding joint marked |
| `design_rule` | theta_max against adjacent link length, pilots marked |
| `conversion_artefact` | pivot-matched vs unmatched placement |
| `prbm_variant` | gamma and K fitted across load ratios |
| `uncertainty` | **skipped until real footage exists** |

Pass `--measured`, `--torque` and `--uncertainty` as those measurements arrive.

Every figure is laid out at its **final printed width**, one journal column (3.4 in) by
default, and identity is carried by hue *and* dash pattern *and* marker, with a hatch on
filled marks. So they read in one ink and to a colour-blind reader. `--column double` gives
the full-width variant. The series hues were chosen against three gates at once -- all-pairs
colour vision, greyscale separation, and contrast against the surface -- which is what ruled
out the first palette: orange and aqua separate well in colour and land ten grey levels
apart, which is no pair at all in print. Tests enforce the greyscale separation, the
uniqueness of the dash and marker per series, and that every figure still fits a column.

### Two things this turned up

**The placement artefact is larger than the physics gap, on every pilot.** Written up in
full in [`docs/results/conversion_artefact.md`](results/conversion_artefact.md):

| design | artefact mean (mm) | PRBM-vs-FEA mean (mm) | ratio |
|---|---|---|---|
| `fb_02_0052` | 0.563 | 0.448 | 1.3x |
| `fb_02_0090` | 1.718 | 0.150 | **11.5x** |
| `fb_02_0203` | 1.901 | 0.202 | **9.4x** |

The shift is exactly proportional to the **pivot displacement** `f_pivot * L`, not to
flexure length -- the constant holds to 0.7-2.4% across a 6x length range and across the
PRBM model boundary, where `f_pivot` drops from 0.5 to 0.265 and the raw artefact therefore
*falls* as the flexure gets longer. Pivot matching is not a refinement: without it the
conversion artefact is the dominant term in the measurement, and it is indistinguishable
from a sim-to-real gap by inspection.

**The beam FEA no longer needs a CAD kernel.** `_attachment_points` lived in
`cmtool/cad/mechanism.py`, which imports cadquery at module scope, so every beam FEA run
pulled in an OCC kernel it never used. It now lives in `cmtool/convert/placement.py` and is
shared by the CAD, the solver and the viewer.

### One correction to §9's framing

The go/no-go signal, 0.46 mm, is the **PRBM-vs-FEA** disagreement. Worth stating alongside
it that the conversion artefact above is of the same order, so a take that resolves one
resolves the other -- and that the 0.46 mm figure itself depends on the placeholder flexure
thickness, since stiffness goes as `t^3`. It moves if the coupon says 0.4 mm prints.

---

## 11. Working agreements

- Plan before writing substantial code; report at each milestone.
- Never invent physical data. Ask when something is physically ambiguous.
- Validate every solver against analytical or published results, as automated tests.
- Keep everything reproducible: pinned env, seeded randomness, YAML configs, and the
  config hash / code commit / seed logged with every result.
- Library-quality code: typed, documented, `pytest`, CI.
- Bulk data (STL, video, Parquet) is **not** committed; it goes to Zenodo. Small example
  STLs under `examples/` and the README figures under `docs/figures/` are whitelisted
  exceptions -- the figures are about 870 kB for the set and a README without them is
  useless.
- **Drawing follows the same two rules.** A missing measured series is absent and labelled,
  never zero; a caveat that is not true is as much a misstatement as an invented number, so
  the dimensionless PRBM-variant figure carries none. See `docs/physics.md` section 11.
