# Query-Point Abaqus Route Review

## Summary

- What changed:
- Why it changed:
- Route version or issue:

## Required Checks

- [ ] Route remains conceptually correct.
- [ ] Data contract is explicit and documented.
- [ ] `LE`, `B`, `ip_keys`, and `point_features` stay row-aligned.
- [ ] `B` labels use the same q coordinate system as the branch input.
- [ ] Train/validation split has no unintended case or geometry leakage.
- [ ] Point feature names and order are checked across compacts.
- [ ] Exporter audits can block bad ODB/compact data when required.
- [ ] Launcher defaults do not silently use old route assumptions.
- [ ] Tests cover the realistic failure mode, not just tensor shapes.

## Validation

- [ ] `python -m compileall src/macro_deeponet scripts`
- [ ] `pytest tests/smoke_test.py -q`
- [ ] Additional validation:

## Risk Level

Choose one:

- [ ] Must fix before training
- [ ] Safe for smoke training only
- [ ] Safe for long training

## Notes

- Remaining risks:
- Follow-up issue/tag:
