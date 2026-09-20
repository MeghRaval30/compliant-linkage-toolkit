## What changed

## Physics / data checklist

- [ ] No physical constant was invented. New numbers are `measured`, `vendor` or
      `design_choice` with a source, or flagged `placeholder` with `value: null`.
- [ ] New solver behaviour is covered by a test against an analytical or published
      reference, not against stored output of this code.
- [ ] Results that use placeholder inputs are still reported as `is_physical = False`.
- [ ] `uv run pytest`, `uv run ruff check src tests`, `uv run mypy` all pass.
