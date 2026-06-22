---
name: v1.1 hard guards
about: Track hard-guard work for the query-point Abaqus route
title: "v1.1 hard guards: "
labels: ["query-point-abaqus", "hard-guards"]
assignees: ""
---

# v1.1 Hard Guards

## Goal

Prevent real Abaqus compact data from training successfully when the data rows,
labels, or validation split are inconsistent.

## Must Complete

- [ ] Train/validation split is non-overlapping by default.
- [ ] Formal training defaults to case-level or geometry-level split.
- [ ] Frame-level split is explicitly marked debug/ablation.
- [ ] Overlap validation requires an explicit debug flag.
- [ ] Split metadata is written to config and training summaries.
- [ ] Merge compact validates `q48_raw` within tolerance.
- [ ] Merge compact validates `LE128_base` within tolerance.
- [ ] Merge compact validates `ip_keys` equality.
- [ ] Missing merge `ip_keys` fails in formal mode.
- [ ] Point feature names and order must match across compacts.
- [ ] IP geometry audit can block export on `COORD` mismatch.
- [ ] TRUE176/CSS8 `detJ` vs `IVOL` audit can block export.
- [ ] Nonstandard element labels cannot silently use 4x4 spatial ID features.
- [ ] Query-point Abaqus launcher defaults to the new route.
- [ ] Smoke tests cover the failure modes above.
- [ ] GitHub Actions compile and smoke tests pass.

## Explicitly Not In Scope

- [ ] Do not add a new model architecture.
- [ ] Do not run long training.
- [ ] Do not claim arbitrary query-point generalization beyond the current data contract.

## Completion Criteria

- [ ] `pytest tests/smoke_test.py -q` passes locally.
- [ ] `python -m compileall src/macro_deeponet scripts` passes locally.
- [ ] GitHub Actions smoke workflow passes.
- [ ] Exporter fails on intentionally mismatched merge compact data.
- [ ] Trainer fails or explicitly marks overlap validation.
- [ ] Changelog records the commit, validation commands, and remaining risks.
