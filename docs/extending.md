# Extending cmtool

The toolkit is scoped to planar four-bars first, but the architecture is general from the
start. Every extension point is a plug-in registry (`cmtool/core/registry.py`). Adding a
capability should be an *addition*, never an edit to a dispatch table.

The rule the codebase holds itself to: **only `cmtool/kinematics/fourbar.py`, its tests, and
the `Linkage.four_bar` convenience constructor may assume a four-bar.** If you find yourself
writing `if linkage.type == "four_bar"` anywhere else, the abstraction has leaked.

---

## The data model you are extending

A linkage is a graph: bodies are nodes, joints are edges, with ground / input / output tags.
Solvers claim a topology by *asking the graph*, not by reading a type string:

```python
linkage.mobility()            # Gruebler: 3(n-1) - sum(constraints).  Four-bar: 1.  Five-bar: 2.
linkage.independent_loops()   # fundamental cycle basis. Four-bar: 1 loop. Six-bar: 2.
linkage.joints_of(body)
linkage.ground, linkage.input_body
```

The data model already represents topologies that have no solver yet — `tests/test_graph.py`
builds and validates a 2-DOF five-bar to prove it. What is missing for those is a solver,
not a data structure.

---

## Adding a linkage type (five-bar, six-bar, ...)

1. Create `cmtool/kinematics/<yourtype>.py`.
2. Write a class with `name`, `can_solve(linkage) -> bool` and
   `solve(linkage, input_angles_rad, **kwargs) -> list[MechanismState]`.
3. `can_solve` must inspect the graph — body/joint counts, `mobility()`,
   `len(independent_loops())`, and whether roles can be identified — and return `False`
   rather than raising for topologies it does not handle.
4. Register it at the bottom of the module: `KINEMATICS.add("your_type", YourSolver())`.
5. Import it from `cmtool/kinematics/__init__.py` so registration happens on import.

Nothing else changes. `simulate()` picks your solver up automatically via `solver_for()`.

**Reuse the dyad.** `solve_dyad()` intersects two circles and returns the requested branch;
it is topology-agnostic. Most planar chains decompose into dyads, which is why position
analysis is formulated geometrically rather than through Freudenstein's equation.

**Two notes for multi-DOF mechanisms.** A five-bar has mobility 2, so `input_angles_rad`
becomes insufficient — extend the state with a second input coordinate rather than
overloading the first. And each independent loop brings its own assembly branch, so
`MechanismState.branch` will need to become a per-loop tuple. Both are additive changes to
`MechanismState`; plan for them rather than working around them.

---

## Adding a flexure type

A flexure type must provide three things:

| Method | Returns | Notes |
|---|---|---|
| `geometry(...)` | CAD profile | consumed by `cmtool/cad/` |
| `prbm_stiffness(...)` | `K` in N·mm/rad | with its validity envelope |
| `strain(...)` | peak strain + model name | **must be specific to the type** |

The third is the one that bites. The leaf-flexure formula `eps = t*theta/(2L)` is correct for
a small-length flexural pivot and **wrong for a circular notch hinge**, which concentrates
strain at its thinnest section. A notch hinge must supply its own model with a geometric
stress-concentration factor read from config, and set
`StrainSpec.includes_stress_concentration = True`. See
[physics.md §4](physics.md#4-flexure-strain-planned-a3).

Also state the characteristic-pivot location, because the conversion strategy needs it for
pivot-matched placement ([physics.md §6](physics.md#6-pivot-matched-placement)).

Register with `FLEXURES.add("notch", NotchHinge())`.

---

## Adding a material

Materials are config, not code. Add `configs/materials/<name>.yaml` following the structure
in `pla.yaml`. Every physical quantity needs a `status` field.

If you do not have a measurement, use `status: placeholder` with `value: null` and a
`provisional_value`. **Do not** write a number with `status: measured` that you did not
measure — that is the one thing this project cannot recover from. A bare scalar with no
provenance block is treated as a placeholder for the same reason.

---

## Adding a solver

Implement `can_solve(linkage)` and `solve(linkage, input_angles_rad, **kwargs)` returning a
`SimulationResult`, then `SOLVERS.add("your_solver", YourSolver())`.

Fill `input_torque_nmm` and `flexure_strain` if your solver computes them; leave them `None`
if it does not. **`None` means "not computed", never "zero"** — downstream code relies on
being able to tell those apart.

If your solver reads material data, thread the result's `Provenance` through every
`Quantity.get(provenance)` call so placeholder usage propagates into the result and into any
sample or figure derived from it.

---

## Adding a design strategy

Design strategies turn a rigid linkage into a compliant one: naive replacement, optimiser,
surrogate, LLM agent. They share one interface so the Phase C baselines are comparable on
equal terms — same train/validation/test splits, same metrics.

Register with `STRATEGIES.add("optimizer", OptimizerStrategy())`; select via
`convert(mech, strategy="optimizer", ...)`.

---

## Testing an extension

Validation tests check solvers against **analytical or independent references**, never
against stored outputs of the code under test. The four-bar suite is the pattern to copy
(`tests/validation/test_fourbar.py`), in increasing order of independence:

1. **Residuals** — the solution must satisfy the constraints exactly (link lengths
   reproduced). Needs no external reference.
2. **Exact hand-computed cases** — a 3-4-5 construction whose joint positions and
   transmission angle are worked out on paper and asserted to `1e-12`.
3. **Closed-form special cases** — a parallelogram must keep its coupler unrotated, its
   output parallel to its input, and give every joint the same excursion. Sign errors cannot
   survive these.
4. **Independent formulations** — the same answer from a numerical vector-loop solve
   (`scipy.optimize.fsolve`) and from Freudenstein's equation. Different derivations, so
   agreement means something.
5. **Property-based tests** (`hypothesis`) — invariants over random valid geometry: closure
   residuals stay zero, sweeps stay on one branch.

Mark them `@pytest.mark.validation` so `pytest -m validation` runs the physics checks alone.

For a new solver, the analogues are the published large-deflection benchmarks: a cantilever
under an end moment (closed-form circular arc) and under an end load (elliptic-integral
solution). Those are milestone A4's acceptance criteria.
