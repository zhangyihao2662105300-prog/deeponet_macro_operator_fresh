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

## v1.1 follow-up - 2026-06-22 - One-case complete compact smoke audit

Purpose:

- Exercise the hard guards on a real Abaqus ODB plus matching Sobolev B compact
  before attempting multi-case query-point training.

Included:

- Exporter `--shape4` and `--shape4-json` now accept a single 4-vector and
  broadcast it across all exported frames.
- Point-feature metadata compresses repeated per-frame `ip_keys` to one `[P,3]`
  table with an explicit repeat marker, avoiding huge config/checkpoint JSON
  when all frames share the same integration-point order.

Audit result:

- Generated one training-ready complete compact from
  `sample_894100.../base/t176_s4_plus_894100_base.odb` with `--strain-field E`
  and merged `css8_shape4_nonzero_sample_894100.npz`.
- Merge/audit guards passed:
  `merge_q48_max_abs_diff=0`,
  `merge_LE128_base_max_abs_diff=0`,
  `merge_ip_keys_match=true`,
  `audit_ref_ip_xyz_vs_abaqus_coord_max_abs=4.76837158203125e-07`,
  `audit_detJ_vs_IVOL_max_abs=7.639755494892597e-10`.
- Formal `case` split correctly failed because only one case/geometry was
  available.
- A 3-epoch `overlap-debug` trainer smoke ran on 8 target IPs and recorded
  `validation_is_overlapping=true`, worst-IP metrics, and random-direction B
  metrics.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- This is not a formal validation run. It uses one case and `overlap-debug`;
  next training audit still needs at least two real training-ready complete
  compacts/cases for case-level or geometry-level split.

## v1.1 follow-up - 2026-06-22 - Strain-field consistency guard

Purpose:

- Prevent future training batches from silently mixing Abaqus `E` and `LE`
  labels while the arrays are still named `LE128_base` / `B_LE128_forward` for
  route compatibility.

Included:

- Complete compact exporter now writes `strain_field`, `B_label_strain_field`,
  `strain_label_key`, and `B_label_key`.
- Merge guard rejects explicit `strain_field` or `B_label_strain_field`
  mismatch between the current ODB export and merged Sobolev B compact, unless
  `--allow-merge-mismatch` is explicitly used for debug.
- Compact loader rejects mixed explicit strain fields across multiple compact
  files and records `strain_meta`.
- Generic trainer records `strain_meta` in config, checkpoints, and training
  summaries.
- Legacy compacts without strain metadata remain loadable as `unknown`, but
  future training-ready compacts should declare this field explicitly.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- Existing old compacts that lack `strain_field` cannot be retroactively proven
  to be `E` or `LE`; they remain marked `unknown` and should not be mixed into
  formal training without audit notes.

## v1.1 follow-up - 2026-06-22 - Reference-frame IP audit

Purpose:

- Keep the IP geometry audit strict while avoiding false failures when Abaqus
  `COORD` in loaded frames reports deformed integration-point coordinates.

Included:

- Complete compact exporter now uses frame 0 `COORD` / `IVOL` as the reference
  geometry audit source when available.
- Selected training/eval frame `COORD` values are stored separately as
  `ip_xyz_abaqus_coord_selected_frames` and summarized by
  `audit_selected_frame_coord_vs_reference_max_abs`.
- `ip_xyz_abaqus_coord` / `ip_IVOL_abaqus` remain the blocking audit fields and
  now explicitly carry reference-frame scope metadata.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q`
- `py -3 -m compileall src/macro_deeponet scripts`

Known gaps:

- This still assumes the TRUE176/CSS8 reference-frame `IVOL` is directly
  comparable to `detJ`; other element families may need quadrature weights.

## v1.1 follow-up - 2026-06-22 - Constant shape4 loader support

Purpose:

- Let the generic query-point trainer consume complete compacts where a constant
  geometry is stored once as `shape4[4]` or `shape4[1,4]`, matching the exporter
  behavior.

Included:

- Compact frame count inference now prioritizes frame-aligned arrays
  (`q48_raw`, `LE128_base`, `B_LE128_forward`) before reading `shape4`.
- `load_one_compact()` broadcasts constant `shape4` metadata to the selected
  frame count, while still requiring `q48_raw`, `LE128_base`, and
  `B_LE128_forward` to remain frame-aligned.
- Smoke coverage verifies that `shape4[4]` is accepted and broadcast safely.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q` -> `40 passed`
- `py -3 -m compileall src/macro_deeponet scripts`

## v1.1 follow-up - 2026-06-22 - Three-case formal small training audit

Purpose:

- Move from one-case `overlap-debug` smoke to a real non-overlapping
  case-level query-point trainer run.

Inputs:

- `complete_case019_training_ready.npz`
- `complete_case025_training_ready.npz`
- `complete_case031_training_ready.npz`
- All three declare `strain_field=LE`, `B_label_strain_field=LE`,
  `merge_q48_max_abs_diff=0`, `merge_LE128_base_max_abs_diff=0`,
  `merge_ip_keys_match=true`, `audit_ref_ip_xyz_vs_abaqus_coord_max_abs`
  about `2.38e-7`, and `audit_detJ_vs_IVOL_max_abs` about `4.29e-10`.

Trainer:

- `model_style=query-fe-linear-residual`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=64`
- `split_mode=case`, `val_fraction=0.34`, `allow_overlap_val=false`
- `epochs=50`

Audit result:

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[25,31]`, `val_cases=[19]`
- `train_frames=20`, `val_frames=10`
- `point_feature_source=data_generic`, `feature_dim=35`
- `supports_dynamic_points=true`
- `strain_meta.strain_field=LE`

Best observed checkpoint:

- Epoch `25`, `score=2.3568`
- `train_LE_rel=0.8248`, `val_LE_rel=1.3879`
- `train_AD_B_rel=0.9668`, `val_AD_B_rel=0.9689`
- `train_rand_dir_B_rel=0.9872`, `val_rand_dir_B_rel=0.9867`

Latest epoch:

- Epoch `50`, `score=5.3967`
- `train_LE_rel=0.8435`, `val_LE_rel=4.0148`
- `train_AD_B_rel=1.4325`, `val_AD_B_rel=1.3820`

Conclusion:

- Formal small training pipeline audit passed: real complete compacts, explicit
  strain metadata, data point features, dynamic query model, and non-overlap
  case split all worked end to end.
- Model performance is not established. With only three load cases and one
  validation case, validation metrics are poor and unstable; this run should not
  be used as evidence that the architecture trains well.

Known gaps:

- This is case-level load extrapolation within one geometry, not geometry
  generalization.
- More real training-ready compacts are needed before judging convergence or
  comparing architecture quality.

## v1.1 follow-up - 2026-06-22 - Query B baseline warm-start diagnostics

Purpose:

- Address the diagnostic gap where fixed-128 per-IP B baselines fit Sobolev B
  labels well, but the query-point model's point-conditioned B baseline learns
  too slowly from a zero-last `point_b_net`.

Included:

- Added optional `query-fe-linear-residual` warm-start arguments:
  `--b-baseline-warmstart-steps`, `--b-baseline-warmstart-lr`,
  `--b-baseline-warmstart-weight-decay`, and
  `--b-baseline-warmstart-source=train-only`.
- Warm-start uses only `train_idx` frames:
  `B_mean_train[ip,6,48] = mean_train B_norm[frame,ip,6,48]`,
  initializes `global_b_norm` from the IP mean, and trains `point_b_net` on the
  train-only residual `B_mean_train - B_global`.
- Validation cases are recorded as excluded in `warmstart_meta`; they are not
  used to form the B prior.
- Config, checkpoints, partial/final summaries, and per-epoch rows record
  warm-start and point-B diagnostics, including
  `b_prior_before_train_rel`, `b_prior_after_train_rel`,
  `b_prior_after_train_cos`, `global_b_prior_rms`,
  `point_b_correction_rms`, and `point_b_correction_norm_ratio`.
- Baseline diagnostics now include both normalized-J metrics and physical/raw-B
  metrics, including eval-column subsets, so warm-start can be compared against
  `AD_B_rel` without mixing metric scales.
- CSV loss history now includes the point-B correction diagnostics.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q` -> `41 passed`
- `py -3 -m compileall src/macro_deeponet scripts`
- Real 3-case overlap-debug, 1-epoch diagnostic:
  `b_prior_after_train_rel = 0.5929` in normalized-J space, while
  `b_prior_after_train_evalcols_B_rel = 0.3579` matches
  `train_AD_B_rel = 0.3579` in physical/raw-B space. This confirms the new
  raw-B baseline metrics are directly comparable with `AD_B_rel`.

Known gaps:

- Warm-start is a model-diagnostic and training aid, not a replacement for
  wider load-direction coverage.
- It is currently restricted to `train-only`; other sources are intentionally
  rejected to avoid validation leakage.
