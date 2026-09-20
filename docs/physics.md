# Physics conventions, models and their validity limits

This file records the modelling decisions the code depends on, and — more importantly —
where each model stops being valid. Every number the toolkit produces should be traceable
to something written down here.

> **Status.** Rigid kinematics (A1), flexure sizing and feasibility (A2), the PRBM
> quasi-static solver (A3), the nonlinear beam FEA (A4) and camera tracking (A5) are all
> implemented and validated. The viewer and figure scripts draw from them (A6). What is
> not yet done is physical: nothing has been printed or measured.

---

## 1. Units and frames

Internally everything is **millimetres and radians**. Degrees appear only at the public API
boundary, where the name says so (`input_range_deg`, `joint_excursion_deg`). All conversions
live in `cmtool/core/units.py`.

**Absolute angles** are measured from the world +x axis. The input coordinate `theta2` is
the direction of the vector from the input joint to the input link's far joint.

**The reference configuration** is the as-defined geometry in `joints_mm` — which, for a
compliant mechanism, is the **as-printed, unstressed** state. Everything that matters to a
flexure is reported relative to it:

- `input_sweep_deg` — input rotation from the reference state
- `joint_rotation_deg(joint)` — relative rotation of the two bodies at a joint
- `joint_excursion_deg()` — **peak-to-peak** of the above over the whole arc

Peak-to-peak is the headline excursion number because a flexure's total bend range, not its
displacement from an arbitrary zero, is what the strain limit and fatigue life respond to.

## 2. Limited input arcs

**A flexure cannot rotate continuously.** A crank that turns fully has no compliant
equivalent, so there is no "full revolution" default anywhere in the toolkit:
`simulate()` refuses to run without an explicit arc.

Grashof classification still matters, and `classify_grashof()` reports
`input_fully_rotates`. That flag is **informational, not a rejection criterion**: a
crank-rocker driven over a 40° arc is a perfectly good compliant candidate. What actually
disqualifies a design is excessive joint excursion (strain), not its Grashof type. The
bundled `examples/fourbar.json` is deliberately a crank-rocker for exactly this reason.

The reachable arc is available in closed form. With the ground link direction `theta1`, the
output dyad closes only while `|r3 - r4| <= |BD| <= r3 + r4`, and since

```
|BD|² = r1² + r2² − 2 r1 r2 cos(theta2 − theta1)
```

the condition becomes a pair of bounds on `cos(theta2 − theta1)`, giving
`phi_min <= |theta2 − theta1| <= phi_max`. See `reachable_input_arc()`; the test suite
checks it against a brute-force assembly scan at 0.25° resolution.

## 3. Branches and toggles

The two intersections of the coupler and output circles are the two **assembly branches**.
Branch `+1` is the intersection left of the directed line `B -> D`. A sweep stays on the
branch of the reference configuration.

A **branch flip** during a sweep is not a pose change — it is a different mechanism. It is
detected and reported (`diagnostics["branch_flip"]`), and Phase B's generator rejects on it.
`dyad_clearance()` gives the distance from a toggle in mm; it reaches zero exactly where the
circles become tangent and the branches merge.

## 4. Flexure strain

### The leaf-flexure model

For a prismatic leaf flexure of thickness `t` and length `L` bent through angle `theta`,
uniform curvature gives radius `rho = L / theta`, so the peak surface strain is

```
eps = t / (2 rho) = t * theta / (2 L)
```

This is the model named `leaf_uniform_bending` in the sample schema. It applies to the
small-length flexural pivot (A2).

### What it excludes, and why that is recorded

1. **Axial stress.** Link loads put the flexure in tension or compression. This is computed
   and reported *separately* rather than folded into the bending figure, so the two
   contributions stay visible. `StrainSpec.includes_axial` records which is which.
2. **Stress concentration.** A prismatic leaf has none of consequence. A **circular notch
   hinge does**, and using this formula for one would under-predict peak strain
   substantially. Strain is therefore a per-flexure-type method (`FlexureType.strain()`),
   not a global function. When the notch hinge is added in Phase B it must supply its own
   model with a geometric stress-concentration factor from config, and set
   `includes_stress_concentration`.
3. **Non-uniform curvature.** Real flexures under combined load do not bend in a perfect
   circular arc. The beam FEA (A4) does not make this assumption, and the PRBM-vs-FEA
   disagreement metric is partly a measure of this error.

### The allowable strain

`allowable_strain` is a **config value per material and is currently a placeholder**. It is
the single most important number for feasibility filtering, since it sets the maximum joint
excursion. It will come from the cantilever strip tests plus a deliberate flexure-to-failure
test (A3, week 3).

## 5. Two PRBM models, and why validity is metadata

### The models

A prismatic flexure is one piece of geometry with **two** pseudo-rigid-body models,
selected by the length ratio `L_flexure / L_link`:

| | pivot from the root | stiffness | γ, K_Θ |
|---|---|---|---|
| `small_length` | `0.5 L` (the centre) | `K = E I / L` | — |
| `long_segment`, end force | `(1 − γ) L` = 0.15 L | `K = γ·K_Θ·E I / L` = 2.25 EI/L | 0.85, 2.65 |
| `long_segment`, end moment | `(1 − γ) L` = 0.265 L | `K = K_Θ·E I / L` = 1.52 EI/L | 0.7346, 1.5164 |

with `I = w t³ / 12`. Two things here are easy to get wrong.

**The stiffness formula differs by loading case.** End force multiplies by γ; end moment
does not. Multiplying the two end-moment constants together gives ≈1.11, which looks as
though it would erase the step at the model boundary. It does not — the correct
end-moment stiffness is 1.52 EI/L.

**The pivot is at `(1 − γ) L` from the root, not `γ L`.** The rigid link of length `γL`
runs from the pivot to the tip, so the pivot is that far *back*. Fitting the exact
circular arc with the pivot at `(1−γ)L` reproduces the tip path to 2×10⁻⁴ L; the other
convention is off by 0.15 L, about 900× worse, and for γ = 0.85 the rigid link is then too
short to reach the tip at all.

### Which variant, and how we know

**The end-moment variant.** Not assumed — determined by our own beam FEA
(`cmtool prbm-study`), which loads a cantilever flexure with a tip moment `M` and
transverse force `P` at ratio `λ = PL/M` and fits γ and K to the resulting tip path:

| λ = PL/M | γ | K/(EI/L) |
|---|---|---|
| 0 (pure moment) | 0.749 | 1.500 |
| **0.1 (a flexure joint)** | **0.754** | **1.523** |
| 1.0 | 0.783 | 1.676 |
| 5.0 (force-heavy) | 0.819 | 1.949 |

A flexure joint is **moment-dominated**. Under no external load the only forces in a
compliant four-bar are those bending the other flexures, so the moment a flexure carries
is of order `K·Δφ ~ (EI/L)Δφ` while the transverse force is of order that divided by a
link length. Their ratio is `λ ~ L/l` — the flexure's own length ratio, about 0.1.

At λ = 0.1 the end-moment variant is within **0.5%** on stiffness; end-force is off by
**48%**.

The end-moment constants are independently corroborated twice over: the exact circular arc
gives γ → 3/4 and K_Θ → 3/2 analytically in the small-angle limit, and the FEA fit gives
0.749 / 1.500. Both agree with the published 0.7346 / 1.5164 to about 2%. The end-force
constants are corroborated in *trend* but not to that precision, because the sweep does not
reach the pure-force limit; that stays open.

### Validity does not filter

**The length ratio selects the model and is recorded on every sample. It never rejects a
design.** Phase C's fidelity map is a map of where the simple model fails, so filtering
those designs out would hide the result it exists to show. Where the simple model does not
apply, beam FEA is the reference.

This was not always so, and changing it moved a headline result. While the 0.1 ratio was
being used as a feasibility constraint, the A1 pilot four-bar was reported unbuildable. It
is not: it builds fine, three of its four joints just need the long-segment model. The old
rule was also about **8× tighter** than the real one (§5.3), which was quietly shrinking the
whole design space.

### The two hard limits

| Limit | Meaning | Consequence of breaking it |
|---|---|---|
| **Strain** | `L ≥ t·θ·SF / (2·ε_allow)` | the flexure cracks |
| **Geometry** | `L ≤ f · l`, `f = 0.8` | no rigid material left to attach to |

Feasibility is these two, and nothing else. A joint's **utilisation** is their ratio; above
1.0 it cannot be built.

### The discontinuity at the switch

The two models still do not agree where they meet. With the end-moment variant the step is
`1.52×` rather than the `2.25×` the end-force variant would give — a real improvement, and
one that came out of choosing the variant on evidence, but not an elimination. Physical
stiffness does not jump at all, so neither model is trustworthy right at the threshold.

These samples are **flagged** (`near_model_boundary`) rather than blended. Blending would
be inventing a model, and inventing a model is the same sin as inventing a measurement.
Resolving that region is one of the things A4 is for.

### What is recorded per sample

`length_ratio`, `small_length_limit`, the `model` chosen (including its variant),
`is_small_length`, `near_model_boundary`, `within_model_angle`, `model_verified`,
`slenderness`, and the `pivot_fraction` the chosen model puts the characteristic pivot at.

That last one matters for geometry as well as kinematics: **pivot matching places the
flexure so the *model's* pivot lands on the rigid joint**, which means centring it only for
the small-length model. A long segment pivots at `(1−γ)L ≈ 0.265 L` from its root, so the
strip sits asymmetrically about the joint. Centring one would put its pivot about 0.24 L
away from where the kinematics assume it is.

Phase A *prefers* small-length designs for pilot prints — the search ranks them higher —
but keeps the rest.

## 6. Pivot-matched placement

This is a conversion rule, not a physical model, but getting it wrong would look like
physics.

For a small-length flexural pivot the PRBM characteristic pivot is at the **centre of the
flexure**, not at the original rigid joint coordinate. If flexures are dropped in by
extending links to the old joint locations, every one of the four effective link lengths
changes by roughly `L_flexure / 2`, and the compliant coupler path is displaced *before any
physics enters*.

So the default `naive` strategy is **pivot-matched**: flexure geometry is placed so that the
characteristic pivots land on the original rigid joint coordinates, and the residual is
recorded in `FlexureSpec.pivot_offset_mm`.

A `naive_unmatched` variant is kept deliberately, so the paper can *show* the size of this
effect rather than silently suffer from it. `FlexureSpec.placement` records which was used,
keeping conversion artefacts separable from the simulation-to-reality gap.

## 7. Gravity, marker mass, and the 2D assumption

The mechanism is tested **lying horizontal**, raised off the table. Two consequences:

**In-plane, gravity contributes exactly zero.** Gravity acts perpendicular to the mechanism
plane, so neither self-weight nor marker-pad mass appears in the in-plane equilibrium the 2D
solvers compute. Adding pad mass to the 2D FEA would be adding zero. (Raising the part off
the table is a good decision for a second reason: it removes table friction, which is larger
and far less predictable than anything gravity does here.)

**Out-of-plane, gravity causes sag and twist,** which the 2D solver cannot see. This is
where the pad mass has to be accounted for, and the check is an out-of-plane one:

*Why the part is stiff in that direction.* A flexure's cross-section is `t` in-plane by `w`
out-of-plane. Second moments are `I_in = w t³ / 12` for the intended motion and
`I_out = t w³ / 12` for sag, so

```
I_out / I_in = (w / t)²
```

With the current design rules (`w = 6 mm`, `t_min = 0.6 mm`) that is a factor of 100. The
part is deliberately far stiffer out of plane than in it — that is what makes the planar
assumption reasonable, and it is why `part_thickness_mm` is a design rule rather than an
afterthought.

*The check to be implemented (A2/A4).* An analytic out-of-plane estimate for the coupler
point, conservatively treating the weakest path as a cantilever:

```
delta_z ≈ m' g L⁴ / (8 E I_out)   +   M_pad g L³ / (3 E I_out)
           (distributed self-weight)   (concentrated pad mass)
```

*Why it matters to the measurement, not the mechanism.* Sag of `delta_z` is not primarily a
mechanism error — it is a **measurement** error. The homography assumes markers lie in the
calibration plane, so a marker that rises out of it by `delta_z` at lateral distance `r`
from the optical axis, with camera standoff `h`, produces an apparent in-plane displacement
of roughly

```
error_apparent ≈ r * delta_z / h
```

The criterion, checked at the A5 go/no-go: `error_apparent` must be well under the
measurement uncertainty. If it is not, the fix is a stiffer part or a longer standoff, not a
3D solver.

**This cannot be evaluated yet.** It needs the measured pad mass and the measured modulus,
both currently placeholders. The estimate will be computed through the placeholder machinery
(§9) so it is reported with the caveat attached until those two measurements exist —
expected week 2–3.

**One in-plane exception.** Pad mass does enter in-plane through *inertia*. Hand-actuated
quasi-static testing makes this negligible. Servo-driven cycling in Phase C does not:
inertial loads scale with the square of the drive frequency, so pad mass must be revisited
there.

## 8. Material behaviour

Printed PLA is **viscoelastic and anisotropic**. One Young's modulus will not serve every
test in this project:

- A modulus measured in a slow cantilever test will not predict a 1 Hz cycling test.
- Creep will drift the path over minutes of filming.

So: modulus is measured at the same rate as the path tests; the filming protocol logs a
settle time; and the config stores `measured_at_rate`, `measured_orientation` and
`measured_date` alongside the value. Phase C's cycling study should expect to need a
different modulus — that is a result, not a bug.

Parts are printed **flat on the bed with the mechanism plane parallel to it**, so flexures
bend in-plane and inter-layer bonds are not loaded in tension. Print orientation is the
single largest strength variable, which is why Phase C varies it deliberately.

## 9. The placeholder discipline

Every physical quantity in `configs/` carries a `status`:

| status | meaning |
|---|---|
| `measured` | our own measurement, with date, rate and orientation recorded |
| `vendor` | datasheet or stated specification, cited |
| `design_choice` | a decision we made, with a justification |
| `confirmed` | a fact about the setup confirmed by the team |
| `placeholder` | **not a measurement** |

A placeholder has `value: null` and a `provisional_value` that exists only so code can
execute. Using one:

1. records the quantity's name in the result's `provenance.placeholders_used`,
2. emits a `ProvisionalDataWarning`,
3. sets `is_physical = False` on the result and on any sample derived from it,
4. attaches a caveat line to plots and reports.

Setting `CMTOOL_STRICT_DATA=1` turns step 1 into a hard error instead. Use it when producing
anything that claims to be a physical result.

The rigid kinematic solver uses no material data at all, so its results are always
`is_physical = True`. The rigid path is pure geometry.

---

## 10. Comparing two paths

The project's headline number is a distance between two curves, so it is worth being
explicit about which distance.

**Pointwise, at matched input angles.** Mean, max and RMS of `|a(theta) - b(theta)|`. This
is the honest comparison when both curves are parameterised the same way, and it is what
the sample schema stores as `prbm_vs_fea_mean_mm` and `prbm_vs_fea_max_mm`. It requires
matched sampling; `resample_to_angles()` exists for when the sampling differs, which it
always does between a solver sweep and a measured take.

**Discrete Frechet.** The shortest leash needed to walk both curves forward together. It
compares curves as *shapes*, ignoring how each was sampled, which is what makes it the
right measure against a measured path whose frames land wherever the camera caught them.
`prbm_vs_fea_frechet_mm`.

The two are kept apart rather than averaged, because a pointwise comparison of two curves
sampled at different rates is a common way to manufacture a discrepancy that is not there.

**One caveat with teeth.** The implementation is the *discrete* Frechet distance: the
walkers stop at vertices, never partway along an edge. It therefore overstates the
continuous distance by at most the coarser curve's vertex spacing — two parallel lines
2 mm apart, sampled every 1 mm against every 0.1 mm, measure 2.06 mm. Curves sampled at
**matched** angles are unaffected, since walking them in step is an admissible coupling, so
a matched comparison never reports more than its own pointwise maximum. When there are no
angles to match on — a measured take whose lever could not be read — the residual inflation
is part of the reported figure and is stated as such.

## 11. What the viewer and the figures may not do

Drawing is where invented data is easiest to introduce and hardest to notice, so the same
two rules apply to `cmtool.viz` as to everything else:

- **A missing series is absent, never zero.** No measured curve is drawn until one exists.
  The absence is stated in words on the figure and recorded in `figures.json`, and the Phase
  A go/no-go figure is *skipped entirely* rather than drawn from the synthetic harness — that
  harness validates the tracking software, not the rig.
- **A caveat that is not true is also a misstatement.** `cmtool prbm-study` fits
  dimensionless constants, so no config quantity reaches it and its figure carries no
  placeholder caveat. Stamping one on would overstate the uncertainty as surely as omitting
  one understates it.
- **Exaggeration is labelled.** A 0.6 mm flexure beside an 8 mm link is invisible at page
  scale, so the viewer draws flexures at a minimum on-screen width. The legend says so,
  gives the true thickness, and a toggle turns it off.
- **The strain colour ramp is normalised against `allowable_strain`.** While that is a
  placeholder the ramp is a relative scale with no physical meaning, and every renderer says
  so on the page.
