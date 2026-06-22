# Query-Point Abaqus Route Changelog

This file records the route-level history for the Abaqus real-integration-point
DeepONet/query-point training line.  Keep it updated whenever this route changes.

## v1 - 2026-06-22 - First query-point Abaqus route baseline

Planned marker:

- Branch: `query-point-abaqus-route`
- Tag: `query-point-abaqus-v1`
- Base commit: `9bab6fc Add FE linear residual DeepONet baseline`

Purpose:

- Treat Abaqus-exported real integration points as the source of truth.
- Remove the fixed 128-point model-structure assumption from the new DeepONet route.
- Align every point feature row with the same row in `LE` and `B`.

Included:

- `QueryFELinearResidualDeepONet` for arbitrary `P` query points.
- Generic Sobolev trainer support for `--model-style query-fe-linear-residual`.
- Complete compact exporter for Abaqus ODB data, including `ip_keys`, `ip_xi`,
  `ip_xyz`, `ip_J`, `ip_invJ`, `ip_detJ`, `ip_frame`, and `point_features`.
- Explicit point-feature loader that prefers compact-stored point data over
  shape4 reconstruction.
- `ip_keys`-based ID features when compact keys are available.
- Optional `--le-normalization global-component` for arbitrary query-point
  inference compatibility.
- Optional `--train-point-sample-count` so query-point training can use random
  point subsets while evaluation stays on the full loaded point table.
- Smoke tests for complete compact loading, query-point training, `ip_keys`
  alignment, and dynamic point counts.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- `B_LE128_forward` still requires finite-difference or perturbed-ODB merge data;
  a single Abaqus ODB does not directly provide `dLE/dq48`.
- Production batch export and merge orchestration still needs a wrapper.
- Arbitrary query-point inference needs a polished entry point that builds and
  normalizes point features, then denormalizes predicted `LE`.
- Dataset split should be controlled at case/geometry level for fair MLP vs
  DeepONet comparison.

Change-note rule for future route work:

- Add one dated section here for every meaningful route change.
- Mention the commit hash, user-facing behavior change, validation command, and
  any remaining risk.

## workflow-baseline - 2026-06-22 - Route tracking toolchain

Purpose:

- Add a fixed workflow for continuing the query-point Abaqus route without
  losing track of data-contract risks.

Included:

- GitHub Actions smoke workflow for `compileall` and `pytest`.
- GitHub issue template for `v1.1 hard guards`.
- Pull request review template focused on data trustworthiness.
- ChatGPT Project instructions for long-running route review.
- Route workflow document with version plan and review gates.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- The `v1.1 hard guards` issue checklist is now documented, but the guards still
  need to be implemented in code.

## v1.1 - 2026-06-22 - Hard guards for data trustworthiness

Planned marker:

- Branch: `query-point-abaqus-route`
- Tag: `query-point-abaqus-v1.1`

Purpose:

- Stop bad real-data compacts from training silently when labels, point rows, or
  validation splits are inconsistent.

Included:

- Strict validation split helper with explicit `case`, `geometry`, `frame`, and
  `overlap-debug` modes.
- Generic trainer records validation split metadata in config, checkpoints, and
  training summaries.
- Point feature loader requires feature names and order to match across all
  compacts.
- `ip_keys` ID feature guard: standard TRUE176 `1..16` labels can use 4x4
  spatial IDs; nonstandard labels fall back to rank-only IDs.
- Abaqus complete compact exporter can fail on merge `q48_raw`, `LE128_base`,
  and `ip_keys` mismatch.
- Abaqus complete compact exporter can fail on IP geometry audit mismatch.
- Query-point Abaqus launcher defaults to the new route and does not enable ID
  features by default.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- This route still trains on the Abaqus 128-IP table and selected subsets of
  that table; full arbitrary query-point label generation remains future work.

## v1.1 follow-up - 2026-06-22 - Mark legacy split helper

Purpose:

- Make the remaining old split helper visibly legacy/debug-only so future
  scripts do not mistake its overlapping validation fallback for formal route
  validation.

Included:

- `split_indices()` now documents its historical overlapping fallback and emits
  a runtime warning when called.
- Smoke coverage checks that the legacy warning remains present.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- The old fixed-128-IP trainer still calls `split_indices()` for compatibility;
  formal query-point/Abaqus training should keep using `split_indices_with_meta()`.
