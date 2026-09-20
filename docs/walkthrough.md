# Walkthrough: what each part of the toolkit actually does

Written so that anyone on the team can explain the code without reading it — in a
supervisor meeting, a lab demo, or an interview. Each section says **what it solves**,
**the equation**, **why we do it that way**, and **where it stops being true**.

The last section is a 60-second version.

---

## 0. The chain

```
rigid linkage  →  conversion  →  PRBM  →  beam FEA  →  printed part  →  camera
  (geometry)     (flexure sizing)  (fast)   (accurate)    (reality)     (measurement)
```

Each stage predicts the path traced by one point on the mechanism. The research question
is how far apart the predictions and the measurement end up, and whether a learned
correction can close the difference.

| Stage | Status | What it gives you |
|---|---|---|
| Rigid kinematics | **working** | The ideal path, pure geometry |
| Conversion + feasibility | **working** | Flexure sizes, and whether the design is buildable |
| Test coupons + mechanism CAD | **working** | Printable parts and print sheets |
| PRBM | **working** | Input torque and strain, fast |
| 2D beam FEA | **working** | Path, torque and strain without the PRBM's assumptions |
| Camera tracking | **working** | The measured path, and its uncertainty |
| 3D solid FEA | Phase C | Checks whether the 2D assumption held |

---

## 1. Rigid kinematics — where does the mechanism go?

**What it solves.** Given four link lengths and an input angle, find where every joint is.

**How.** Not with Freudenstein's equation, which is the textbook route, but geometrically.
Put joint `A` at the origin and `D` on the x-axis. The input link fixes `B` on a circle
about `A`. Then `C` must be exactly the coupler length from `B` *and* exactly the output
length from `D` — so `C` is where two circles cross:

```
|C − B| = coupler length
|C − D| = output length
```

**Why this way.** Three reasons, and it is worth being able to give them:

1. Two circles cross at two points, and those two points *are* the two ways the linkage
   can be assembled. The branch falls out of the geometry instead of being an extra case.
2. Toggle positions become obvious: the branches merge exactly where the circles are
   tangent. `dyad_clearance()` returns that distance in millimetres, so "near a toggle"
   is a number rather than a judgement.
3. It generalises. Most planar linkages decompose into pairs of links like this, so the
   same routine serves a five-bar or six-bar later without change.

**Where it stops.** It is pure geometry: no forces, no stiffness, no material. The rigid
path is what an ideal pin-jointed mechanism would trace. Real compliant mechanisms deviate
from it, and measuring that deviation is the point of the project.

**How we know it is right.** Not by comparing against stored output. A 3-4-5 construction
whose joint lands at exactly `(2.88, 3.84)` with a transmission angle of exactly 90°; a
parallelogram, whose coupler must never rotate; and agreement with two independent
formulations (a numerical vector-loop solve and Freudenstein's equation).

---

## 2. Conversion — can this design be built out of flexures?

This is where most of the engineering judgement sits, and it is the part most worth being
able to explain.

**The problem.** A pin joint rotates freely. A flexure is a thin strip that bends, and
bending it too far breaks it. So not every rigid linkage has a compliant equivalent.

**Two constraints, pulling opposite ways.**

*Strain wants a long flexure.* Bend a strip of thickness `t` and length `L` through angle
`θ` and it curves to radius `ρ = L/θ`. The surface strain of a strip bent to radius `ρ` is

```
ε = t / (2ρ) = t·θ / (2L)
```

So surviving a given bend needs `L ≥ t·θ / (2·ε_allow)`.

*Geometry wants a short flexure.* A flexure is a necked-down section of a link, so rigid
material has to remain at each end: `L ≤ f · l`, with `f = 0.8` and `l` the shorter adjacent
link.

**Eliminate `L` and you get the rule that governs everything:**

```
θ_max = 2 · f · l · ε_allow / (t · SF)
```

`f` is the geometric fraction (0.8), `l` the shorter adjacent link, `SF` a safety factor.

Note what is *not* in that bound: pseudo-rigid-body validity. The small-length ratio (0.1)
used to stand where `f` is now, which made model validity a feasibility constraint and put
the bound 8× tighter than physics requires. Validity is metadata now — it picks which PRBM
model represents the flexure, and nothing more.

**Why this matters, and it is counter-intuitive:** the largest usable joint rotation is set
by **the length of the neighbouring link**, not by anything about the flexure by itself.
Short links cannot carry large rotations at any thickness. A classic crank-rocker with a
stubby crank is exactly the wrong shape for a compliant mechanism.

Worth knowing how this played out, because it is a good lesson in what a filter costs.
While small-length validity was being enforced as a constraint, the A1 pilot four-bar (18 mm
input link) was reported unbuildable — it needs a 7.25 mm flexure for a 9.2° bend against a
1.8 mm cap. Under the correct rule the cap is 14.4 mm and the design is fine; three of its
four joints simply use the long-segment model. A modelling convenience had been quietly
rejecting real designs, including exactly the ones whose model error the project exists to
measure.

**Two decisions that buy a factor of two each, for free:**

*Print unstressed at mid-arc.* The part is unstressed in the configuration it was printed
in. Print it at the **middle** of its travel and each flexure swings symmetrically about
zero, so the largest bend is half of what it would be if printed at one end. Peak strain
halves; the required flexure length halves. Measured on a real design: worst-joint
utilisation drops from 9.59 to 4.89.

*Pivot-matched placement.* A small-length flexure behaves like a pin at its **centre**, not
at its ends. Drop flexures in without accounting for that and every effective link length
shifts by about half a flexure length — the coupler path moves before any physics is
involved. So flexures are centred on the original joint by default. The `unmatched` variant
is kept deliberately so the size of that artefact can be measured instead of guessed at;
keeping the two separable is what stops a conversion bug being mistaken for a
simulation-to-reality gap.

**What the feasibility report tells you.** For each joint: the bend it must take, the
flexure length strain requires, the length that actually fits, and their ratio — the
**utilisation**. Above 1.0 the joint cannot be built. The joint with the highest
utilisation is the one **limiting the design**, which is the actionable number: it says
which joint to fix. Alongside it, and kept firmly separate, sits the PRBM model each joint
uses and any caveats about it — model notes are not failures.

---

## 3. Test coupons — measuring what we currently guess

Every feasibility verdict above depends on two numbers we do not yet have: the thinnest
flexure the printer can make, and the material's modulus and strain limit. Both are
flagged placeholders, so every result the toolkit currently produces is marked
**not a physical prediction**.

The **flexure coupon** is a plate of strips at 0.4, 0.5, 0.6, 0.8 and 1.0 mm sharing one
rail, identified by pip count rather than text (no font dependency, legible at 0.4 mm
nozzle resolution). Print it, measure every strip with calipers, bend each by hand. The
thinnest strip that prints completely, measures close to nominal and survives handling sets
`min_flexure_thickness_mm`.

The **cantilever coupon** is six strips for a deflection test, `E = F·L³/(3·δ·I)`. Two
thicknesses deliberately: a 1 mm strip is nearly all perimeter, a 2 mm strip contains
infill. If their apparent moduli differ, then a modulus measured on a thick strip does not
transfer to a 0.5 mm flexure unchanged — which is something to find out before trusting any
simulation.

The **strain coupon** measures `ε_allow`: strips wrapped around vertical posts of
decreasing radius, where the strain is `(t/2)/(R + t/2)` — the exact form, not the familiar
`t/(2R)`, which overstates it by 6% at the tightest radius and this is the one number where
that matters. The posts span roughly 0.008 to 0.057 strain. Check whitening, cracking and
permanent set on a single bend and again after ten cycles; the largest strain that still
passes sets the limit.

One geometric detail that decides whether the test means anything: a flexure bends
**in-plane**, so the strips are printed as upright thin walls and wrapped around *vertical*
posts. Printed flat and bent over a horizontal bar, a strip is loaded across its layer
boundaries instead, and what you measure is interlayer adhesion — a different and usually
much lower failure strain.

**The slicer setting that matters most** is the wall generator: **Arachne**. It varies
extrusion width to fill thin features exactly. The classic generator works in whole-nozzle
perimeters and will thin, distort or silently drop a sub-2-perimeter wall — which is every
flexure in this project.

---

## 4. PRBM — the fast approximation

**What it solves.** The rigid model says where the mechanism goes. It does not say what
force is needed to drive it, or how much strain each flexure sees.

**How.** Replace each flexure with a pin at its characteristic pivot plus a torsional
spring, then use energy. Writing `Δφⱼ(θ)` for the rotation at joint `j` measured from the
configuration the part was *printed* in (which is unstressed):

```
U(θ) = Σⱼ ½ Kⱼ Δφⱼ(θ)²        T(θ) = dU/dθ
```

One degree of freedom, so that derivative is the whole story.

**Which K.** Two models, picked by the length ratio: `K = EI/L` for a short flexure pinned
at its centre, and for a long segment Howell's model pinned at `(1−γ)L` from the root. That
second one has two variants with different constants *and different formulas* — end force
uses `γ K_Θ EI/L`, end moment uses `K_Θ EI/L`. Our beam FEA says flexure joints are
moment-dominated, so the end-moment variant is the one in use.

**The thing people get wrong.** With a *prescribed* input angle, one degree of freedom and
no external load, **stiffness does not affect the path at all**. Double every `K` and you
get the identical coupler curve and exactly twice the torque. So the PRBM path differs from
the rigid path only because the pivots sit somewhere else — and with pivot-matched
placement, they don't. What the PRBM buys in Phase A is the torque curve, the strain, and
being the reference the FEA and measurements are compared against.

The path *will* diverge once effects the PRBM cannot represent enter: a real flexure
stretches and shears as well as bending, so the links are not quite the length it assumes.
That is what the `prbm_vs_fea` metric measures, and why A4 exists.

**Why bother.** Milliseconds, not seconds — which is what makes thousands of dataset
samples possible. And it is the standard model in the field, so results are comparable with
published work.

**How we know it works.** A parallelogram. Every joint in one rotates by exactly the input
sweep, so with equal springs `T(θ) = (ΣK)·Δθ` in closed form. The solver matches that to
about 2 × 10⁻¹¹ N·mm against a 148 N·mm peak — which exercises the kinematics, the energy
sum and the numerical differentiation all at once, against an answer derived rather than
recorded.

---

## 5. 2D nonlinear beam FEA — the accurate one

**What it solves.** The same thing, without assuming circular bending or small deflections.

**How.** Chop the whole mechanism into beam elements — flexures slender, links stout — and
clamp the ground. Each element carries its own rotating local frame, so a large rotation of
the element is separated from the small strain inside it. That is what "co-rotational"
means, and it is what lets ordinary linear beam theory describe a flexure that has swung
40°: large displacement, small strain.

The resulting equations are nonlinear, so Newton–Raphson solves them: guess, compute the
out-of-balance force, correct, repeat. The input rotation is applied in increments, each
warm-started from the last converged state, because a good initial guess is what keeps
Newton in its basin.

**Why bother when the PRBM exists.** Three things the PRBM cannot represent:

- **distributed compliance** — the flexure bends along its length rather than hinging at a
  point, and the shape of that bend changes with load;
- **axial and shear compliance** — a real flexure stretches, so the effective link lengths
  are not constant. This is the main reason the two predict different *paths* at all, since
  with a prescribed input the PRBM path is pure geometry;
- **pivot drift** — the effective pivot moves as the load ratio changes through the cycle.

On the pilot mechanisms the two disagree by about 0.5 mm mean on the coupler path and about
a third on peak torque. Both numbers go in the dataset as `prbm_vs_fea` metrics, and both
are large enough to measure.

**How we know it is right.** Two benchmarks, neither of them a stored number:

- a cantilever under a **pure end moment**, which bends into an exact circular arc. The FEA
  matches the whole deflected shape to under 10⁻⁴ L, converges at second order in element
  size, and gets the tip *slope* exact to machine precision — the co-rotational element
  represents constant-curvature rotation exactly.
- a cantilever under a **transverse end load**, integrated from `EI θ'' + P cos θ = 0`.
  This is the classical Bisshopp–Drucker solution; it is evaluated here by quadrature
  rather than recalled as an elliptic-integral formula, so the reference does not depend on
  anyone's memory. Tip position matches to under 10⁻⁴ L out to a 3.0 load parameter.

Both are comfortably inside the 1% the project asked for.

---

## 5b. Input torque — the measurement that tests the stiffness

Worth separating from the path, because it is easy to assume the path does this job.

With a prescribed input, one degree of freedom and no external load, the coupler path is
fixed by geometry. Double every flexure's stiffness and the path does not move. So **the
path cannot test the stiffness model at all** — it tests kinematics and pivot placement,
which matter, but nothing about `K`.

Input torque is directly proportional to stiffness. It is where the measured modulus, the
PRBM's `K` and the FEA's distributed compliance all meet reality. So the printed lever
carries a through-hole at a known radius.

**The rig, corrected.** A scale weighs *vertically* and the mechanism lies *flat*, so the
pull on the lever is horizontal and a kitchen scale cannot read it. The primary rig is
therefore a dead weight: thread from the lever hole, horizontally to a pulley at the table
edge, weights hanging from it. The mechanism settles at an equilibrium angle read from the
same camera.

The subtlety is that the thread direction is not fixed — as the lever swings, the hole moves
and the thread swings with it, so the moment arm changes through the sweep. The applied
torque is `T = W·cross(H − A, û)`, with the cross product *being* the effective moment arm.
Assuming it stays at `r` would be wrong by however much the geometry rotates. A spring
scale at a known angle is kept as a backup mode.

For the pilot mechanisms the predicted peak torque is about 32 N·mm (FEA) against 48 N·mm
(PRBM) — at a 36 mm arm, roughly 90 g versus 135 g of hanging weight. The models differ by
far more than a set of kitchen weights resolves, so this discriminates between them.

Record loading and unloading separately. PLA is viscoelastic so the branches will not
coincide, and the gap between them is hysteresis — a material property, not a measurement
error.

## 6. Measurement — the number the paper rests on *(A5)*

Camera on a tripod, ArUco marker boards on the base, input lever and coupler point. Camera
calibration removes lens distortion; a homography maps the image to millimetres using
fiducials at known spacing.

**The go/no-go test at the end of Phase A:** is the measurement uncertainty clearly smaller
than the sim-to-real gap being measured? If not, no amount of solver work helps. It is
tested two ways: static marker jitter, and a **known-motion ground truth** — a rigid bar on
a pin swept through a known arc, where the true path is a circle we can compute exactly.
The second is the honest number, because it also catches lens distortion residual,
out-of-plane motion and fixture flex.

**Why gravity is not in the 2D model.** The mechanism is tested lying horizontal, so
gravity acts perpendicular to the plane and contributes exactly zero to the in-plane
equations. What it does cause is out-of-plane sag, and that is a *measurement* error, not a
mechanism error: a marker lifted out of the calibration plane appears displaced sideways.
The part is designed ~100× stiffer out of plane than in it (`I_out/I_in = (w/t)²`), which
is why the 6 mm part thickness is a design rule and not an afterthought.

---

## 7. The rules the code enforces on itself

**No invented physical data.** Every config quantity carries a status. A placeholder has
`value: null` and a `provisional_value` that exists only so code can run. Using one records
its name in the result's provenance, warns, and marks the result `is_physical = False`.
`CMTOOL_STRICT_DATA=1` turns that into an error. This is why the coupons are printed before
any mechanism.

**Limited input arcs only.** A flexure cannot rotate continuously, so there is no
full-revolution default anywhere. Every simulation states its arc, and every joint's
excursion over that arc is recorded.

---

## 8. The 60-second version

> Compliant mechanisms replace pin joints with thin flexing strips, so they can be printed
> in one piece with no assembly and no friction. The catch is that a flexure bends a limited
> amount before it breaks, so not every linkage has a compliant equivalent, and simulations
> of the ones that do don't match reality as well as you'd like.
>
> We're building an open toolkit that converts rigid linkages to compliant ones, simulates
> them three ways, and measures the printed parts on camera — plus the first open dataset
> pairing all four.
>
> The most useful thing we found so far is a design rule that falls out of two constraints
> in tension. Strain wants a long flexure; the pseudo-rigid-body model's validity wants a
> short one. Eliminating the length between them gives a bound on joint rotation that
> depends on the *adjacent link length* — so short links can't carry large rotations at any
> flexure thickness. Our first pilot design turned out to be unbuildable for exactly that
> reason, and the fix was redesigning the linkage, not the flexure.
