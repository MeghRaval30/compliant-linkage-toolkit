# The flexure-placement artefact is larger than the model error it would be mistaken for

**Status: a computational result, reproducible today — with one placeholder dependency,
stated up front.** Both curves compared are rigid kinematic simulations, so no material
constant enters the *comparison*: the scaling law below, and the ratio of artefact to
modelling error, hold whatever the material turns out to be.

What does depend on unmeasured data is the **flexure lengths**. Those come from the sizing
rule, which divides by `allowable_strain` — currently a placeholder. So the absolute
millimetres in the first table move when the strain coupon is measured. The figure carries
the placeholder caveat for exactly this reason. The scaling result does not: it is measured
by forcing the length, so it is independent of how the length was chosen.

Regenerate with `cmtool figures --only conversion_artefact`; the figure is
[`docs/figures/conversion_artefact.png`](../figures/conversion_artefact.png).

---

## The claim

When a rigid linkage is converted to a compliant one, each pin joint is replaced by a
flexure. A flexure does not pivot at its end — it pivots at its **characteristic pivot**,
a point inside it whose position depends on which pseudo-rigid-body model applies. Drop the
flexures in so that their *ends* land on the original joint coordinates, and every
characteristic pivot is displaced. Every effective link length changes. The coupler path
moves.

That displacement is a **conversion artefact**: pure bookkeeping, no physics. On two of our
three pilot four-bars it is **an order of magnitude larger** than the PRBM-versus-FEA
disagreement — the actual modelling error the project exists to measure.

Anyone doing rigid-body replacement who does not account for pivot position is measuring
their own placement convention and calling it a simulation-to-reality gap.

## The measurement

Two rigid simulations of the same linkage over the same arc, at 41 matched input angles.
The only difference is where the flexures were placed:

- **pivot-matched** — flexure geometry positioned so the model's characteristic pivot lands
  on the original rigid joint (`placement="pivot_matched"`, the default);
- **unmatched** — flexures dropped in by extending links to the old joint coordinates
  (`placement="unmatched"`, kept in the toolkit precisely so this can be measured).

No material, no stiffness, no solver difference. Whatever separates the two curves came from
the placement rule alone.

## Result across the three pilots

Flexures sized by the default rule at 0.6 mm thickness, arcs fitted to a 22° worst-joint
excursion. All four joints of all three designs use the long-segment end-moment model, whose
characteristic pivot sits at `(1 − γ)L = 0.265 L` from the root.

| design | flexure length (mm) | artefact mean / max (mm) | PRBM-vs-FEA mean / max (mm) | artefact ÷ model error |
|---|---|---|---|---|
| `fb_02_0052` | 7.48 – 8.74 | **0.563** / 1.166 | 0.448 / 0.964 | **1.3×** |
| `fb_02_0090` | 6.60 – 8.87 | **1.718** / 2.141 | 0.150 / 0.312 | **11.5×** |
| `fb_02_0203` | 7.09 – 8.71 | **1.901** / 2.457 | 0.202 / 0.398 | **9.4×** |

The right-hand column is the point. On `fb_02_0090` and `fb_02_0203` an unmatched conversion
would put roughly ten times more displacement into the coupler path than the difference
between the two physics models being compared — and it would look exactly like a
simulation-to-reality gap, because it is a smooth, systematic, angle-dependent offset of the
predicted path.

It is worth noting *why* `fb_02_0052` is the mild case: that design's coupler point happens
to be relatively insensitive to link-length perturbation. Its `k` below is a quarter of the
others'. Nothing about it makes the artefact small in principle, and picking the pilot that
looked worst would have made the effect look four times larger than it typically is.

## How it scales

Forcing the flexure length across a 6× range on each design, the mean path shift is
**exactly proportional to the displacement of the characteristic pivot**:

```
mean path shift  =  k · f_pivot · L
```

where `L` is the flexure length, `f_pivot` is the model's pivot fraction (0.5 for the
small-length model, 0.265 for the long-segment end-moment model), and `k` is a property of
the linkage geometry — how much the coupler point moves per millimetre of link-length error.

| design | k (mean) | k (max) | spread over L = 2 … 12 mm |
|---|---|---|---|
| `fb_02_0052` | 0.252 | 0.525 | 0.7 % |
| `fb_02_0090` | 0.951 | 1.191 | 1.1 % |
| `fb_02_0203` | 1.009 | 1.295 | 2.4 % |

Two things are worth drawing out.

**The constant is the same on both sides of the PRBM model boundary.** Between `L = 6 mm`
and `L = 8 mm` on `fb_02_0052` the flexures cross from the small-length model to the
long-segment one, and the pivot fraction drops from 0.5 to 0.265. The raw artefact drops
with it — from 0.763 mm to 0.538 mm, *falling* as the flexure gets longer. Divided by the
pivot displacement, `k` does not move: 0.2542 against 0.2535. So the governing quantity is
**how far the pivot is displaced**, not how long the flexure is. Anyone who fits a scaling
law to flexure length alone will find it breaks at the model boundary for no apparent reason.

**`k ≈ 1` is normal.** On two of three designs the coupler path moves about as far as the
pivot does. There is no built-in attenuation to rely on.

## Why this matters for anyone doing rigid-body replacement

1. **Place flexures by their pivot, not their ends.** The correction is free — it is a
   geometric offset applied at conversion time — and on a typical design it removes an error
   comparable to or larger than the modelling error you are trying to study.

2. **Which pivot depends on which model applies.** Centring the flexure is correct only for
   the small-length flexural pivot. A long segment pivots at `(1 − γ)L` from its root, so
   centring one puts its pivot about `0.235 L` from where the kinematics assume it is. Since
   the model in force depends on the length ratio `L/l`, the correct offset changes as a
   design is iterated — it is not a constant you can apply once.

3. **A conversion artefact and a sim-to-real gap are indistinguishable from the data.**
   Both are smooth, systematic, angle-dependent displacements of the predicted path. Nothing
   in a measured coupler path separates them. The only defence is to eliminate the artefact
   by construction and keep the ability to measure what it would have been — which is why
   `placement="unmatched"` is kept in the toolkit rather than deleted.

4. **Check it against your own signal before trusting a comparison.** The number that
   matters is not the artefact in millimetres, it is the artefact divided by the effect being
   measured. Ours ranges from 1.3× to 11.5× across three designs of the same family.

## Relation to the Phase A go/no-go

The measurement go/no-go is stated against a 0.46 mm signal, the mean PRBM-versus-FEA
disagreement on `fb_02_0052`. The artefact on that design is 0.563 mm — the same order. A
camera take good enough to resolve one resolves the other, so this does not change the
uncertainty target. What it changes is the interpretation: had the conversion been
unmatched, the thing being resolved at 0.46 mm would not have been a modelling error at all.

## Caveats

- **Rigid kinematics only.** Both curves come from the rigid solver. The artefact is a
  kinematic displacement, and adding compliance to both sides would not remove it, but the
  numbers here are not a compliant-mechanism prediction.
- **Three designs, one family.** All are planar four-bars from the same search, with link
  lengths of 64–84 mm and similar arcs. `k` is a geometric sensitivity and will differ for
  other topologies and for coupler points placed differently on the coupler.
- **`L` is set by the strain rule.** The flexure lengths in the first table are what the
  sizing rule chose for a 0.6 mm thickness. If the flexure coupon says 0.4 mm prints
  reliably, the lengths fall, the designs become fully small-length, the pivot fraction rises
  to 0.5, and the artefact has to be recomputed — the two changes push in opposite
  directions.
- **The unmatched variant is one particular wrong answer.** It is the most common one, and
  the one the toolkit would have produced by accident, but a different careless convention
  would give a different number.

## Reproducing the numbers

```bash
cmtool figures --only conversion_artefact --out docs/figures
```

The per-design and scaling tables came from two rigid sweeps per configuration, using
`cmtool.api.convert` with `placement` set both ways and `cmtool.metrics.paths.compare_paths`
at matched input angles. The figure's manifest entry in
[`docs/figures/figures.json`](../figures/figures.json) records the design, the numbers and
the code commit that produced them.
