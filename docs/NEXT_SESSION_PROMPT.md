# Prompt for the next Claude Code session

Copy everything between the lines into a new Claude Code chat opened in
`C:\Users\raval\Documents\Compliant_Mechanism_Research`.

---

I'm continuing work on the Rigid-to-Compliant Linkage Toolkit (`cmtool`) in this repo.

**First, read `docs/HANDOVER.md` in full.** It has the complete state: what's built, the
findings that shaped the design, hardware settings, what's still unmeasured, and what's
left. Then skim `README.md`, `docs/physics.md` (physics conventions and validity limits)
and `docs/walkthrough.md` (plain-language explanation of each solver).

Short version of where things stand: milestones A0–A5 are done, committed and pushed;
381 tests pass with lint and types clean; CI is green on Ubuntu + Windows × Python
3.11/3.12. Everything left in Phase A is **physical** — printing and measuring, not
solver work.

Two rules the codebase enforces and you must keep enforcing:

1. **Never invent physical data.** Material constants, strain limits and printer
   tolerances live in `configs/` with a `status` field. Anything unmeasured stays a
   `placeholder` with `value: null`; using one marks the result `is_physical = False`.
   Never write `status: measured` for a number that wasn't measured. Ask me instead of
   guessing.
2. **Limited input arcs only** — a flexure can't rotate continuously, so there's no
   full-revolution default anywhere.

How I'd like you to work: plan before writing substantial code and wait for my approval on
anything large; report at each milestone with what works, what was tested and what's open;
validate every solver against analytical or published results as automated tests, never
against stored output of our own code; keep everything reproducible (pinned env via `uv`,
seeded randomness, YAML configs, config hash + code commit + seed logged with each result).
Run `uv run pytest`, `uv run ruff check src tests` and `uv run mypy` before committing, and
push so CI runs.

Development commands:

```bash
uv sync --extra dev --extra plot --extra data --extra cad --extra vision
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
```

**What I want to do this session:** <replace this line with your actual goal>

Some likely options, in the order `docs/HANDOVER.md` section 10 recommends:

- I've printed the coupons and have measurements — help me put them into the configs
  correctly and re-run everything that depended on them.
- Build the A6 demo materials: the rigid/PRBM/FEA/measured overlay figure, the README with
  images, and the demo video script. `cmtool.viz` is currently empty.
- Start Phase B: `cmtool generate` is still a stub (random sample generation, filters,
  dedup, the resumable batch pipeline, and the metrics module).
- Something else — tell me and I'll work from the handover.

If I say I have measurements, ask me for the numbers **and** their provenance (what was
measured, how, on which printer, at what rate, on what date) before writing anything into
a config.

---

## Notes for whoever pastes this

- Replace the `**What I want to do this session:**` line with the actual goal. Leaving it
  vague will cost a round trip.
- If you have coupon results, have these ready: measured vs nominal thickness for each
  strip, which strips survived 10 bend cycles, the cantilever deflection data (force, free
  length, measured b and h, loading rate, hold time), and the strain-coupon verdicts from
  `examples/coupons/strain_test_template.csv`.
- The repo is at <https://github.com/MeghRaval30/compliant-linkage-toolkit> if a session
  ever needs to start from a fresh clone.
