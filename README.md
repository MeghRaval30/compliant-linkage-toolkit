# cmtool — rigid-to-compliant planar linkage toolkit

Converts rigid planar linkages into compliant (flexure-based) mechanisms, simulates them,
exports printable CAD, and measures the printed parts from video — so that simulated
designs can be compared against physical reality on the same footing.

It is the software half of an open benchmark dataset pairing rigid linkages with their
compliant counterparts, including paths measured on real FDM-printed parts.

**Research question:** how well do simulated compliant-mechanism designs (pseudo-rigid-body
model and FEA) transfer to FDM-printed parts, and can a learned model close that
simulation-to-reality gap?

> **Status: early pilot (Phase A, milestone A1).** Rigid four-bar kinematics work and are
> validated. Conversion, PRBM, FEA, CAD export and camera tracking are not implemented yet.
> No physical measurements exist yet — every material constant in `configs/` is an
> explicitly flagged placeholder, and any result computed from one is labelled as
> not a physical prediction.

## Install

```bash
uv sync --extra dev
```

## Use

```python
from cmtool import Linkage, simulate

mech = Linkage.from_json("examples/fourbar.json")
result = simulate(mech, solver="rigid", input_range_deg=(40, 100), n_steps=61)

result.path()                    # (N, 2) coupler path in mm
result.joint_excursion_deg()     # per-joint rotation over the arc
result.diagnostics["grashof"]    # classification and whether the input can fully rotate
```

From the command line:

```bash
cmtool simulate examples/fourbar.json --steps 91 --csv out/rigid_path.csv
```

## Two rules the code enforces

**1. No invented physical data.** Material constants, strain limits and printer tolerances
live in `configs/`, each carrying a `status` field. A `placeholder` value is not a
measurement: using one records its name in the result's provenance, emits a warning, and
marks the result `is_physical = False`. Setting `CMTOOL_STRICT_DATA=1` turns that into a
hard error. Simulated numbers are never presented as measured ones.

**2. Limited input arcs only.** A flexure cannot rotate continuously, so there is no
"full revolution" default anywhere. Every simulation states its input arc explicitly, and
every joint's excursion over that arc is recorded — that excursion is what the flexure
strain limit acts on.

## Architecture

A linkage is a **graph**: bodies are nodes, joints are edges, with ground, input and output
tags. Solvers claim a topology by inspecting the graph (mobility, loop count), never by
reading a type string. Flexure types, materials, solvers and design strategies are all
plug-in registries.

Only `cmtool/kinematics/fourbar.py` and its tests assume a four-bar. Adding a five-bar,
six-bar or a new flexure type is an addition, not a rewrite — see [docs/extending.md](docs/extending.md).

```
src/cmtool/
  core/        linkage graph, units, registries, provenance, quantities
  kinematics/  position solvers, dispatched by topology
  solvers/     rigid | prbm | beam_fea | solid_fea -> one SimulationResult type
  schema/      pydantic models; JSON Schema is generated from them
  flexures/ materials/ convert/ cad/ vision/ metrics/ dataset/ viz/   (later milestones)
```

Physics conventions, the strain model and its validity limits: [docs/physics.md](docs/physics.md).

## Development

```bash
uv run pytest                 # tests, including solver validation
uv run pytest -m validation   # only the analytical / published-reference checks
uv run ruff check src tests
uv run mypy
uv run pre-commit install
```

Validation tests check solvers against analytical results rather than against stored
outputs: exact hand-computed configurations, closed-form special cases, and independent
numerical formulations.

## Licence

Code: MIT (`LICENSE`). Data and dataset releases: CC BY 4.0 (`LICENSE-DATA`).
