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

## v1.1 follow-up - 2026-06-22 - Query warm-start main-training controls

Purpose:

- Diagnose why full Sobolev main training can degrade a useful query B prior
  after warm-start.

Included:

- Added `--le-loss-weight` so B-only main-training diagnostics can set LE loss
  weight to zero without changing the training loop by hand.
- Added `--freeze-b-baseline-after-warmstart` to freeze `global_b_norm` and
  `point_b_net` after train-only warm-start, isolating whether the main
  optimizer is directly corrupting the B baseline.
- Added `--global-b-lr-scale` and `--point-b-lr-scale` optimizer parameter
  groups to test smaller baseline learning rates during full training.
- Config and summaries now record `main_train_control`, including optimizer
  parameter counts and group learning rates.

Intended diagnostics:

- Warm-start + frozen B baseline: if the baseline metric stays good but full
  model `AD_B_rel` worsens, the residual AD path is the likely source.
- Warm-start + B-only main train (`--le-loss-weight 0`): if B degrades despite
  no LE loss, inspect AD-B loss/optimizer settings before increasing epochs.
- Warm-start + reduced baseline learning rates: tests whether preserving the
  prior while training residual improves stability.

Real 3-case overlap-debug diagnostics:

- All runs below are diagnostic only: they use overlapping train/validation
  frames and must not be interpreted as case-level generalization.
- Warm-start baseline before main training:
  `b_prior_after_train_evalcols_B_rel = 0.3579`.
- Experiment A, frozen B baseline with full LE + AD loss:
  the baseline stayed fixed at `b_prior_current_train_evalcols_B_rel = 0.3579`,
  but full-model `train_AD_B_rel` worsened from `0.3586` at epoch 1 to
  `15.2793` at epoch 20. This rules out direct corruption of
  `global_b_norm/point_b_net` as the only problem and points to the residual
  AD path as a major failure mode.
- Experiment B, B-only main train with normalized-J loss:
  normalized `train_AD_B_norm_rel` improved from about `0.5913` to `0.4356`,
  while physical/raw `train_AD_B_rel` worsened from `0.9217` to `1.8168`.
  Optimizing normalized-J alone can move the model in a direction that looks
  better in normalized space but worse in physical B space.
- Experiment B2, B-only main train with physical-J loss:
  raw `train_AD_B_rel` stabilized near `0.49-0.50`, better than normalized-J
  B-only but still worse than the warm-start prior. The loss scale is safer,
  but the main optimizer can still erode the prior.
- Experiment C, full LE + physical-J loss with small baseline learning rates
  (`global_b_lr_scale = 0.05`, `point_b_lr_scale = 0.10`):
  the B prior was preserved (`b_prior_current_train_evalcols_B_rel = 0.3571`
  at epoch 20), full-model `train_AD_B_rel = 0.3574`,
  `val_AD_B_rel = 0.3657`, and LE improved to `train_LE_rel = 0.3028`,
  `val_LE_rel = 0.2514`.

Current interpretation:

- The useful query B prior can be preserved during main training, but it needs
  physical/raw-B-aware J loss and conservative baseline parameter updates.
- The remaining route question is not whether B labels or AD-B are wired, but
  how to schedule residual and baseline updates so LE improves without
  injecting destructive q-derivatives.

## v1.1 follow-up - 2026-06-22 - Formal case split warm-start audit

Purpose:

- Move the best overlap-debug training strategy back to a real non-overlapping
  case-level validation split.
- Add current train/validation B-prior metrics to the generic trainer summary
  and CSV history so future runs can distinguish whether the query B prior
  itself generalizes to the held-out case, or whether residual AD terms dominate
  the total model derivative.

Code changes:

- `query-fe-linear-residual` training now logs both
  `b_prior_current_train_*` and `b_prior_current_val_*`.
- The legacy DeepONet CSV writer includes the validation B-prior fields so
  `loss_history.csv` retains the same diagnostics as the JSON summaries.

Formal 3-case audit:

- Output directory:
  `outputs/query_point_v1_1_formal_small_audit/trainer_case_split_3compact_warmstart_physical_smallBLR_ep50`
- Inputs:
  `complete_case019_training_ready.npz`,
  `complete_case025_training_ready.npz`,
  `complete_case031_training_ready.npz`
- Split:
  `split_mode=case`, `allow_overlap_val=false`,
  `train_cases=[25,31]`, `val_cases=[19]`,
  `train_frames=20`, `val_frames=10`
- Model/training:
  `model_style=query-fe-linear-residual`,
  `point_feature_source=data`,
  `branch_feature_mode=xkeep-qraw`,
  `le_normalization=global-component`,
  `train_point_sample_count=128`,
  train-only B warm-start for 500 steps,
  `j_loss_mode=physical`,
  `global_b_lr_scale=0.05`,
  `point_b_lr_scale=0.10`,
  `le_loss_weight=1.0`,
  `epochs=50`

Warm-start and latest metrics:

- Train-only warm-start, before main training:
  `b_prior_before_train_evalcols_B_rel = 0.9781`,
  `b_prior_after_train_evalcols_B_rel = 0.4313`,
  `b_prior_after_train_evalcols_B_cos = 0.9022`
- Latest and best checkpoint are both epoch 50, `score = 2.7052`
- Current B prior:
  `b_prior_current_train_evalcols_B_rel = 0.3756`,
  `b_prior_current_val_evalcols_B_rel = 0.3694`
- Full model B:
  `train_AD_B_rel = 0.3768`, `train_AD_B_cos = 0.9317`,
  `val_AD_B_rel = 0.3716`, `val_AD_B_cos = 0.9339`
- Physical random-direction B:
  `train_phys_rand_dir_B_rel = 0.3773`,
  `val_phys_rand_dir_B_rel = 0.3708`
- LE:
  `train_LE_rel = 0.2253`, `val_LE_rel = 2.3336`
- Worst aggregate errors:
  `train_AD_B_ip_rel_max = 2.4735`,
  `val_AD_B_ip_rel_max = 2.5196`,
  `val_AD_B_col_rel_max = 1.8485`

Epoch trend:

- Epoch 1: `val_AD_B_rel=0.4385`, `val_LE_rel=7.0280`
- Epoch 10: `val_AD_B_rel=0.3726`, `val_LE_rel=5.9601`
- Epoch 25: `val_AD_B_rel=0.3735`, `val_LE_rel=4.0671`
- Epoch 50: `val_AD_B_rel=0.3716`, `val_LE_rel=2.3336`

Current interpretation:

- The formal case split is clean and non-overlapping.
- The query B strategy is now credible under a held-out case: train-only
  warm-start plus physical-J and conservative B-baseline learning rates keeps
  validation physical/raw `AD_B_rel` near `0.37`.
- LE held-out case generalization is still not established. It improves from a
  very poor initial value, but `val_LE_rel = 2.3336` remains too high.
- With only three load cases and one held-out case, the next scientific step is
  adding independent cases/load directions, not extending this 3-case run to a
  long training schedule.

Validation:

- `py -3 -m pytest tests/smoke_test.py -q` -> `42 passed`
- `py -3 -m compileall src/macro_deeponet scripts`
- `git diff --check`

## v1.2 start - 2026-06-22 - Data-coverage audit tooling

Purpose:

- Start `v1.2 data-coverage audit`.
- Keep model/training strategy fixed and first test whether poor held-out `LE`
  prediction is caused by insufficient independent case/load-direction
  coverage.
- Provide a reproducible data ledger before generating or training on
  `10-20` real training-ready complete compacts.

Included:

- Added `scripts/audit_query_point_data_coverage.py`.
- The script reads complete compact NPZ files and writes:
  `compact_manifest.csv`, `compact_manifest.json`,
  `q_direction_cosine_matrix.csv`, `q_direction_abs_cosine_matrix.csv`,
  `le_rms_distribution.csv`, `audit_summary.json`,
  `train_80_20_summary.json`, and `loo_summary.json`.
- Each compact row records case id, frame count, strain metadata, q norm
  statistics, representative q direction, internal q-direction cosine
  statistics, q-direction nearest-existing coverage, q-direction cluster id,
  `LE_rms`, `B_rms`, merge guards, strain guard, and reference-frame IP audit
  fields.
- Strict mode requires the v1.1 hard-guard metadata:
  `training_ready_sobolev`, explicit non-unknown matching strain fields,
  merge q/LE tolerances, `merge_ip_keys_match`,
  `merge_strain_field_match`, reference COORD audit, and TRUE176/CSS8
  `detJ` vs `IVOL` audit.
- The generated 80/20 and leave-one-case-out files are coverage/split plans
  with metric fields set to `null`; they should be filled only after the
  corresponding formal training runs complete.
- Updated workflow version boundaries: `v1.2` is now data-coverage audit, and
  ID/inference polish is shifted to `v1.3`.

Current v1.2 boundary:

- Do not prioritize model architecture changes in v1.2.
- First expand the real complete compact pool and record q-direction coverage,
  LE scale coverage, and clean case-level splits.
- Keep the first v1.2 training configuration fixed:
  train-only query-B warm-start, physical-J loss,
  `global_b_lr_scale=0.05`, `point_b_lr_scale=0.10`,
  `train_point_sample_count=128`, `split_mode=case`,
  `allow_overlap_val=false`.

Validation:

- Added smoke coverage for manifest/matrix/split-plan generation and strict
  hard-guard failure on missing audit metadata.
- Real 3-case probe using the existing case019/case025/case031 complete
  compacts passed strict v1.2 audit and produced two q-direction clusters at
  `abs(cos) >= 0.95`; pairwise abs-cosine ranged from `0.8967` to `0.9938`.
  This supports the current interpretation that the 3-case pool is still too
  thin in independent load directions.

## v1.2 follow-up - 2026-06-22 - Legacy TRUE176 boundary-control candidate audit

Purpose:

- Use old TRUE176 compact data as a candidate boundary-displacement direction
  library without treating it as formal v1.2 query-point/Abaqus training data.
- Select independent `q48_raw` directions that should be reapplied as fresh
  boundary-displacement controls, then regenerated through Abaqus and the
  current exporter as strict v1.2 complete compacts.

Included:

- Added `scripts/select_legacy176_reexport_candidates.py`.
- The script reads legacy `q48_raw`, `LE128_base`/`le`, optional B metadata,
  and `sample_paths`; it does not train a model and does not require old
  compacts to pass strict v1.2.
- The script compares legacy representative q directions against the current
  strict-pass pool, clusters old candidates by absolute cosine, and writes:
  `legacy176_candidate_manifest.csv`,
  `legacy176_candidate_manifest.json`,
  `legacy176_to_current_abs_cosine.csv`,
  `legacy176_candidate_cluster_summary.json`, and
  `legacy176_reexport_plan.md`.
- The script also writes `legacy176_boundary_control_plan.json`, which contains
  the selected unit `q_direction_48` vectors for future
  `q_new = alpha * q_direction_48` displacement-control runs.
- Added `docs/query_point_v1_2_legacy176_reexport_plan.md` to document that
  old TRUE176 compacts are boundary-control direction sources only and must not
  be added directly to the formal v1.2 pool.

Audit result:

- Current strict-pass pool remains case019/case025/case031:
  `strict_v1_2_pass=true`, `compact_count=3`, `case_count=3`,
  `frame_count=30`, `q_direction_cluster_count=2`,
  `pairwise_abs_cos_min=0.8966673549`,
  `pairwise_abs_cos_median=0.9078195202`,
  `pairwise_abs_cos_max=0.9937922950`.
- Legacy candidate audit scanned 6 old compact sources and found 733 candidate
  samples in 17 q-direction clusters.
- 150 candidates met `max_abs_cos_to_current < 0.95`; 100 of those met the
  stronger `< 0.90` threshold.
- The top 12 recommended re-export candidates are assigned tentative new case
  ids case040-case051.  Their `max_abs_cos_to_current` range is
  `0.226673` to `0.862846`, so they are all less redundant than the current
  case019/case031 near-duplicate direction.
- The audit was rerun with `--compute-b-rms`; recommended candidate `B_rms_mean`
  values were recorded in the manifest and re-export plan.

Known gaps:

- The old TRUE176 compacts still lack the current strict v1.2 metadata
  (`point_features`, IP geometry/Jacobian tables, `ip_keys`, explicit strain
  fields, merge-audit fields, and reference-frame IP audit fields).  They remain
  candidate-direction data only.
- The selected candidates' local `sample_path` entries do not currently resolve
  to ready-to-use current v1.2 samples: top-12 `base_odb_exists=false` and
  `perturb_dir_exists=false`.
- Formal pool entry still requires regenerating a fresh base ODB from
  `q_new = alpha * q_direction_48`, generating a matching Sobolev B compact,
  running the current complete-compact exporter, and passing strict v1.2.

## v1.2 follow-up - 2026-06-22 - Fresh boundary-control case plan

Purpose:

- Convert selected legacy `q_direction_48` candidates into explicit fresh
  Abaqus boundary-displacement control plans for case040-case051.
- Keep v1.2 focused on data generation coverage, not model/loss/training
  changes.

Included:

- Added `scripts/build_v1_2_boundary_control_cases.py`.
- The script reads `legacy176_boundary_control_plan.json` and writes:
  `fresh_boundary_control_plan.json`, `fresh_boundary_control_plan.csv`,
  per-case `caseXXX_q48_frames.csv`, and
  `abaqus_boundary_control_command_templates.md`.
- Each selected case gets 10 frames with
  `alpha_factors = 0.1, 0.2, ..., 1.0` and
  `q48_frame = alpha_factor * q_norm_seed_mean * q_direction_48`.
- The script validates that every frame satisfies `||q48_frame|| = alpha` and
  keeps the same direction as the selected `q_direction_48`.
- Added `docs/query_point_v1_2_fresh_boundary_control_cases.md` to document the
  q48 order and fresh Abaqus/export/strict-audit workflow.

Boundary condition note:

- q48 is the current 16 keep-node x 3 DOF control vector.  Existing generated
  Abaqus `.inp` examples often apply displacements on 32 boundary nodes / 96
  DOF through a boundary-contract bridge.  If the Abaqus generator expects 96
  DOF, it must explicitly apply the same `q48 -> q_boundary[32,3]` bridge; q48
  must not be silently reinterpreted as a 96-DOF vector.

Known gaps:

- This commit does not run Abaqus and does not generate ODB/B/complete compact
  files.
- A project-specific q48-to-Abaqus boundary-control generator, shape4 values,
  and matching Sobolev B compact generation are still required before any
  case040-case051 compact can enter the formal v1.2 pool.

## v1.2 follow-up - 2026-06-22 - Abaqus boundary bridge requirements

Purpose:

- Resolve the bridge between generated `case040-case051` q48 frame plans and
  Abaqus boundary-node displacement inputs before running fresh ODB generation.

Included:

- Added `scripts/build_v1_2_abaqus_boundary_inputs.py`.
- The script applies an explicit bridge contract:
  `q_boundary96 = T_boundary_96x48 @ q48`.
- The bridge is read from `boundary_contract_arrays.npz`, which provides
  `T_boundary_96x48`, `keep_nodes`, and `boundary_nodes`.
- The script writes per-case `case###_boundary96_frames.csv`,
  `case###_boundary_nodes.json`, and `case###_abaqus_bc_commands.txt`, plus a
  summary JSON/CSV under the output directory.
- Added `docs/query_point_v1_2_abaqus_boundary_bridge_requirements.md` to record
  the bridge source, node orders, pilot sequence, exporter template, and strict
  audit template.

Bridge audit:

- Local boundary-contract files contain a fixed `T_boundary_96x48` with
  `boundary_nodes = [1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25,26,27,28,29,30,31,35,36,40,41,45,46,47,48,49,50]`.
- The checked contracts share the same T matrix and node orders.
- For a representative contract,
  `q48_frames @ T_boundary_96x48.T` reproduces stored
  `q_boundary_frames` with `max_abs_diff=0`.

Known gaps:

- The generated Abaqus BC command files are templates only.  A fresh Abaqus
  base model/input generator, shape4 selection, step/increment convention, field
  output requests, and matching Sobolev B compact generation are still required.
- Old TRUE176 `LE/B` values remain scale/provenance hints only and are not used
  as v1.2 labels.

## v1.2 follow-up - 2026-06-22 - Pilot fresh Abaqus case041 chain

Purpose:

- Run one representative fresh Abaqus pilot case before attempting any batch
  case040-case051 generation.
- Verify the end-to-end chain:
  `case041 q48/boundary96 plan -> fresh Abaqus ODB -> matching Sobolev B compact
  -> current complete compact exporter -> strict v1.2 coverage audit`.

Included:

- Added `scripts/run_v1_2_pilot_case041_fresh_abaqus.py`.
- The script is intentionally narrow and defaults to case041 only.
- It validates that the generated q48 frames are linear 0.1-1.0 scale factors
  of the final q48 vector and that
  `T_boundary_96x48 @ q48 == boundary96` before running Abaqus.
- It uses the audited shape4/T-boundary bridge and existing fresh Abaqus
  base-plus-48-forward-perturbation generator to build a matching Sobolev B
  compact.
- It repacks the fresh B compact with explicit
  `strain_field=LE` and `B_label_strain_field=LE`, then calls the current
  `export_abaqus_true176_complete_compact.py` with `--merge-compact`,
  `--require-b`, `--require-merge-ip-keys`, and `--require-ip-audit`.

Pilot result:

- Fresh base job and 48 forward perturbation jobs completed successfully for
  case041.
- Generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041\abaqus_run\base\fresh_case041.odb`.
- Generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041\b_compact\case041_sobolev_B_compact.npz`.
- Generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case041\complete\complete_case041_training_ready.npz`.
- Single-case strict audit passed:
  `strict_v1_2_pass=true`, `compact_count=1`, `case_count=1`,
  `frame_count=10`, `q_direction_cluster_count=1`.
- Full-pool audit with case019/case025/case031 plus fresh case041 passed:
  `strict_v1_2_pass=true`, `compact_count=4`, `case_count=4`,
  `frame_count=40`, `q_direction_cluster_count=3`,
  `pairwise_abs_cos_min=0.3343678087`,
  `pairwise_abs_cos_median=0.6789307736`,
  `pairwise_abs_cos_max=0.9937922950`.
- case041 adds a new q-direction cluster.  Its nearest existing direction is
  case031 with `max_abs_cos_to_current=0.4611941923`.

Known gaps:

- This is still a data-generation/contract pilot, not a training result.
- Large generated Abaqus/NPZ outputs remain outside git and must not be
  committed.
- Old TRUE176 `LE/B` values were not used as training labels; only the selected
  boundary-displacement direction and audited bridge were used to produce fresh
  Abaqus data.

## v1.2 follow-up - 2026-06-22 - Extend fresh Abaqus pilots to case043 and case050

Purpose:

- Extend the validated case041 pilot path to two more selected fresh
  boundary-control directions without running the full case040-case051 batch.
- Keep this as a data-generation and coverage-audit step only; no training or
  model/loss/learning-rate changes were made.

Included:

- Generalized `scripts/run_v1_2_pilot_case041_fresh_abaqus.py` so its manifest
  file is `pilot_{case_id}_plan.json` instead of the case041-specific
  `pilot_case041_plan.json`.
- Reused the same audited bridge, fresh Abaqus base-plus-48-forward-perturbation
  chain, B compact repack, current complete compact exporter, and strict v1.2
  data coverage audit for case043 and case050.

Pilot result:

- case043 generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case043\abaqus_run\base\fresh_case043.odb`,
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case043\b_compact\case043_sobolev_B_compact.npz`,
  and
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case043\complete\complete_case043_training_ready.npz`.
- case050 generated:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\abaqus_run\base\fresh_case050.odb`,
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\b_compact\case050_sobolev_B_compact.npz`,
  and
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\complete\complete_case050_training_ready.npz`.
- Both single-case strict audits passed with `strict_v1_2_pass=true`.
- Full-pool audit with case019/case025/case031/case041/case043/case050 passed:
  `strict_v1_2_pass=true`, `compact_count=6`, `case_count=6`,
  `frame_count=60`, `q_direction_cluster_count=5`,
  `pairwise_abs_cos_min=0.1808575415`,
  `pairwise_abs_cos_median=0.5679210886`,
  `pairwise_abs_cos_max=0.9937922950`.
- case043 adds q-direction cluster 3 with
  `max_abs_cos_to_current=0.5679210886`, nearest existing case019,
  `LE_rms_mean=0.0001830914007`, and `B_rms_mean=9.1729672619`.
- case050 adds q-direction cluster 4 with
  `max_abs_cos_to_current=0.8533500881`, nearest existing case019,
  `LE_rms_mean=0.0002458211478`, and `B_rms_mean=9.1749346929`.

Known gaps:

- The pool is improved but still below the v1.2 target of 10-20 strict-pass
  complete compacts and 6-10 q-direction clusters.
- The original case019/case031 near-duplicate remains in the pool, so
  `pairwise_abs_cos_max` is still `0.9937922950`.
- Large generated Abaqus/NPZ outputs remain outside git and must not be
  committed.
- Old TRUE176 `LE/B` values were not used as training labels.

## v1.2 follow-up - 2026-06-22 - Second fresh Abaqus batch reaches coverage target

Purpose:

- Expand the v1.2 fresh strict-pass compact pool beyond the first three fresh
  pilots while still avoiding training/model/loss changes.
- Re-rank the remaining case040-case051 candidates against the current 6-case
  strict-pass pool before selecting the next cases.

Selection:

- Selected case044, case046, case049, and case045.
- All selected candidates had `max_abs_cos_to_current_pool < 0.95`, were
  expected to add q-direction clusters, and had non-tiny amplitude/LE scale
  hints.
- Deferred case040 even though its direction was independent because its
  expected LE scale was extremely low (`LE_rms_hint_mean=2.8848e-06`,
  `q_norm_max=6.6611e-05`).
- Deferred case048 because it was near-duplicate with existing case050
  (`max_abs_cos_to_current_pool=0.9551`).
- Deferred case051 because it was close to case045/case049 in the candidate
  set.  case042 and case047 remain usable follow-up candidates.

Pilot result:

- case044, case046, case049, and case045 each generated a fresh Abaqus base
  ODB, matching Sobolev B compact, and complete training-ready compact.
- All four single-case strict v1.2 audits passed.
- Full-pool audit with case019/case025/case031/case041/case043/case050 plus
  this second batch passed:
  `strict_v1_2_pass=true`, `compact_count=10`, `case_count=10`,
  `frame_count=100`, `q_direction_cluster_count=9`,
  `pairwise_abs_cos_min=0.0279701647`,
  `pairwise_abs_cos_median=0.6286590998`,
  `pairwise_abs_cos_max=0.9937922950`.
- case044 adds q-direction cluster 5 with
  `max_abs_cos_to_current=0.6653788417`, nearest case031,
  `LE_rms_mean=0.00008300474246`, and `B_rms_mean=9.1698764680`.
- case046 adds q-direction cluster 6 with
  `max_abs_cos_to_current=0.7669577541`, nearest case019,
  `LE_rms_mean=0.0001091349404`, and `B_rms_mean=9.1698297192`.
- case049 adds q-direction cluster 7 with
  `max_abs_cos_to_current=0.8364611150`, nearest case031,
  `LE_rms_mean=0.0001956932680`, and `B_rms_mean=9.1729725926`.
- case045 adds q-direction cluster 8 with
  `max_abs_cos_to_current=0.8522134771`, nearest case049,
  `LE_rms_mean=0.0001667079867`, and `B_rms_mean=9.1675470365`.

Current v1.2 status:

- The data pool now meets the minimum coverage target:
  10 strict-pass complete compacts and 9 q-direction clusters.
- The pool still contains the original case019/case031 near-duplicate, so
  `pairwise_abs_cos_max` remains `0.9937922950`.
- This is a data coverage milestone only.  No training was run, no model/loss
  settings changed, and old TRUE176 `LE/B` values were not used as labels.
- Large Abaqus/NPZ outputs remain outside git and must not be committed.

## v1.2 follow-up - 2026-06-22 - Fixed-strategy 10-case training audit

Purpose:

- Start the first v1.2 fixed-strategy training audit after the fresh
  strict-pass compact pool reached the minimum coverage target.
- Keep the v1.2 strategy frozen and test whether the expanded 10-case /
  9-cluster pool improves held-out LE/B behavior relative to the 3-case formal
  split.
- Do not tune the model, loss, learning rates, or epoch count in response to
  this run.

Pool:

- Source coverage audit:
  `D:\IS-FEM\outputs\query_point_v1_2_data_coverage_audit\full_pool_after_second_fresh_batch`
- Frozen training manifest:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest`
- Pool cases:
  `case019`, `case025`, `case031`, `case041`, `case043`,
  `case044`, `case045`, `case046`, `case049`, `case050`.
- Coverage:
  `strict_v1_2_pass=true`, `compact_count=10`, `case_count=10`,
  `frame_count=100`, `q_direction_cluster_count=9`,
  `pairwise_abs_cos_min=0.0279701647`,
  `pairwise_abs_cos_median=0.6286590998`,
  `pairwise_abs_cos_max=0.9937922950`.
- All compacts declare `strain_field=LE` and
  `B_label_strain_field=LE`; no unknown strain metadata and no legacy TRUE176
  label compact were mixed into the pool.

Split:

- Split plan:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\split_plan_main_80_20.json`
- Train cases:
  `[19, 25, 31, 43, 44, 45, 46, 50]`
- Validation cases:
  `[41, 49]`
- Trainer split metadata:
  `validation_split_mode=case`, `validation_is_overlapping=false`,
  `train_frames=80`, `val_frames=20`.
- Validation case notes:
  `case041` is a low-LE fresh direction with nearest-train abs-cosine
  `0.6417599947` to `case044`.
  `case049` is a mid-LE fresh direction with nearest-train abs-cosine
  `0.8522134771` to `case045`.

Fixed strategy:

- Output directory:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20`
- Trainer:
  `macro_deeponet.train_true176_generic_sobolev`
- Fixed settings:
  `model_style=query-fe-linear-residual`,
  `point_feature_source=data`,
  `branch_feature_mode=xkeep-qraw`,
  `le_normalization=global-component`,
  `train_point_sample_count=128`,
  `split_mode=case`,
  `allow_overlap_val=false`,
  `epochs=50`,
  `b_baseline_warmstart_source=train-only`,
  `b_baseline_warmstart_steps=500`,
  `j_loss_mode=physical`,
  `global_b_lr_scale=0.05`,
  `point_b_lr_scale=0.10`,
  `le_loss_weight=1.0`.
- Warm-start was train-only:
  `warmstart_train_cases=[19,25,31,43,44,45,46,50]`,
  `warmstart_val_cases_excluded=[41,49]`,
  `warmstart_validation_is_overlapping=false`.

Result:

- Best and latest checkpoints are both epoch 50.
- Best/latest metrics:
  `best_score=5.6694053020`,
  `train_LE_rel=0.3710049271`,
  `val_LE_rel=5.1376408458`,
  `train_AD_B_rel=0.5343235027`,
  `val_AD_B_rel=0.5317644562`,
  `train_AD_B_cos=0.8530025687`,
  `val_AD_B_cos=0.8755700366`,
  `train_phys_rand_dir_B_rel=0.5360939388`,
  `val_phys_rand_dir_B_rel=0.5349325279`,
  `b_prior_current_train_evalcols_B_rel=0.5338542326`,
  `b_prior_current_val_evalcols_B_rel=0.5306775207`,
  `train_AD_B_ip_rel_max=4.0114861980`,
  `val_AD_B_ip_rel_max=3.5443742903`,
  `val_AD_B_col_rel_max=0.9385777742`.
- Epoch trend:
  `val_LE_rel` decreased from `12.7145` at epoch 1 to `5.1376` at epoch 50,
  while `val_AD_B_rel` stayed near `0.532`.

Interpretation:

- The fixed-strategy 10-case run completed with a clean non-overlapping
  case-level split.
- The B prior and full AD-B metrics remain closely aligned, so the residual AD
  path did not destroy the B field in this split.
- B did not match the 3-case formal split quality:
  `val_AD_B_rel` is about `0.5318` here versus about `0.3716` in the 3-case
  split.
- LE generalization is not established:
  `val_LE_rel=5.1376`, worse than the previous 3-case audit value
  `val_LE_rel=2.3336`.
- Conservative conclusion:
  the v1.2 fixed-strategy training audit ran cleanly, but this main 80/20 split
  does not show LE generalization improvement from the expanded 10-case pool.
  The next diagnostic step should be additional fixed-strategy split or
  leave-one-case-out audits, not ad hoc model/loss/lr/epoch tuning.

Validation boundary:

- No model structure, loss, learning-rate strategy, or epoch schedule was
  changed for this audit.
- Old TRUE176 `LE/B` values were not used as v1.2 labels.
- Large generated training artifacts and checkpoints remain outside git and
  must not be committed.

## v1.2 follow-up - 2026-06-23 - Fixed-strategy split diagnostic

Purpose:

- Diagnose whether the poor first 10-case fixed-strategy result was specific
  to the split_A validation cases (`case041`, `case049`) or reflects a broader
  fixed-strategy LE generalization issue.
- Keep the training strategy frozen: no model, loss, learning-rate, or epoch
  changes.
- Add a lightweight checkpoint attribution helper that evaluates selected
  cases from an existing checkpoint without training.

Included:

- Added `scripts/evaluate_v1_2_checkpoint_by_case.py`.
- The script reconstructs the generic query-point preprocessing from a saved
  checkpoint and compact list, then calls the existing trainer evaluation
  metrics for selected case ids.
- Generated split diagnostic plans under:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_diagnostic_plan`.
- Planned splits:
  split_A `[41,49]` completed, split_B `[44,46]` run this round, split_C
  `[43,50]` planned only, split_D `[25,45]` planned only.

Split_A attribution:

- Output:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_main_80_20_case_attribution`
- Per-case metrics:
  `case041`: `LE_rel=9.1422`, `AD_B_rel=0.5318`,
  `phys_rand_dir_B_rel=0.5330`, `B_prior_rel=0.5306`,
  `AD_B_cos=0.8754`.
  `case049`: `LE_rel=4.0235`, `AD_B_rel=0.5317`,
  `phys_rand_dir_B_rel=0.5330`, `B_prior_rel=0.5307`,
  `AD_B_cos=0.8757`.
- Interpretation: split_A's LE failure was worse on case041 than case049, while
  B metrics were nearly identical across the two validation cases.

Split_B:

- Output:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46`
- Split:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`,
  `validation_is_overlapping=false`.
- Fixed strategy matched split_A exactly:
  `model_style=query-fe-linear-residual`,
  `point_feature_source=data`,
  `branch_feature_mode=xkeep-qraw`,
  `le_normalization=global-component`,
  `train_point_sample_count=128`,
  `split_mode=case`,
  `allow_overlap_val=false`,
  `epochs=50`,
  `b_baseline_warmstart_source=train-only`,
  `b_baseline_warmstart_steps=500`,
  `j_loss_mode=physical`,
  `global_b_lr_scale=0.05`,
  `point_b_lr_scale=0.10`,
  `le_loss_weight=1.0`.
- Best and latest checkpoints are both epoch 50:
  `train_LE_rel=0.3992`,
  `val_LE_rel=8.5155`,
  `train_AD_B_rel=0.5178`,
  `val_AD_B_rel=0.5103`,
  `val_AD_B_cos=0.8834`,
  `val_phys_rand_dir_B_rel=0.5143`,
  `b_prior_current_val_evalcols_B_rel=0.5094`,
  `val_AD_B_ip_rel_max=3.3736`,
  `val_AD_B_col_rel_max=1.0005`.

Split_B attribution:

- Output:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\fixed_strategy_split_B_val44_46_case_attribution`
- Per-case metrics:
  `case044`: `LE_rel=9.6526`, `AD_B_rel=0.5102`,
  `phys_rand_dir_B_rel=0.5132`, `B_prior_rel=0.5094`,
  `AD_B_cos=0.8835`.
  `case046`: `LE_rel=7.7823`, `AD_B_rel=0.5104`,
  `phys_rand_dir_B_rel=0.5189`, `B_prior_rel=0.5094`,
  `AD_B_cos=0.8833`.

Current interpretation:

- split_B did not rescue LE generalization; `val_LE_rel=8.5155` is worse than
  split_A's `val_LE_rel=5.1376`.
- B remains stable and aligned with the B prior in both splits:
  split_A `val_AD_B_rel=0.5318` vs `B_prior_val=0.5307`;
  split_B `val_AD_B_rel=0.5103` vs `B_prior_val=0.5094`.
- The conservative diagnosis is now stronger:
  the v1.2 10-case LE failure is not just a case041/case049 split-specific
  artifact.  Under the frozen strategy and current single-geometry 10-case
  pool, LE case-level generalization appears systematically weak while B
  remains controlled by the query B prior.
- This still does not prove the model architecture is bad.  It means the next
  useful diagnostic question is why the residual/LE prediction branch fails to
  generalize when the B prior remains stable.

Validation boundary:

- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No model, loss, learning-rate, or epoch settings were changed.
- Large generated checkpoints, NPZ/ODB files, and loss histories remain outside
  git and must not be committed.

## v1.2 follow-up - 2026-06-23 - LE failure decomposition

Purpose:

- Decompose why fixed-strategy split_A and split_B have high held-out `LE_rel`
  while `AD_B` remains stable and aligned with the query B prior.
- Keep this as an evaluation-only diagnostic: no training, no model/loss/lr
  changes, and no epoch changes.

Included:

- Added `scripts/diagnose_v1_2_le_failure.py`.
- The script reconstructs the generic query-point preprocessing from a saved
  checkpoint, runs forward LE prediction for selected cases, and writes:
  `le_case_summary.csv`, `le_component_summary.csv`,
  `le_worst_ip_summary.csv`, per-case frame trend CSVs,
  `le_offset_summary.csv`, `le_offset_by_case.csv`, and
  `le_failure_diagnostic_summary.json`.
- It also computes zero-prediction and train-mean baselines, scale/bias
  corrections, train-only affine LE oracle diagnostics, and B-prior offset
  diagnostics.

Outputs:

- Split_A diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\le_failure_diagnostic\split_A_val41_49`
- Split_B diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\le_failure_diagnostic\split_B_val44_46`

Key LE scale result:

- The model predicts held-out LE values that are much larger than the true
  target RMS:
  `case041 target_rms=9.33e-05, pred_rms=8.60e-04, LE_rel=9.14`;
  `case049 target_rms=2.21e-04, pred_rms=9.04e-04, LE_rel=4.02`;
  `case044 target_rms=9.36e-05, pred_rms=9.18e-04, LE_rel=9.65`;
  `case046 target_rms=1.23e-04, pred_rms=9.64e-04, LE_rel=7.78`.
- `zero_pred_LE_rel=1.0` for all checked validation cases, so the model is
  worse than predicting zero on these held-out cases.
- Scale-only correction improves each case only to about zero-baseline quality:
  `scaled_LE_rel` is about `0.98-0.99`.
- Bias-only correction barely helps, so the failure is not a simple constant
  offset.

Component/IP localization:

- Worst average components are dominated by `LE11`, `LE23`, and `LE22`.
- Worst local IP errors are severe:
  split_A `case041` reaches `IP_LE_rel=43.56`;
  split_B `case044` reaches `IP_LE_rel=40.99`.
- Low component targets amplify some local relative errors, but the aggregate
  predicted LE RMS is also too large, so this is not only a reporting artifact.

Frame/alpha trend:

- Low-alpha frames are worst, but high-alpha frames remain poor:
  `case041 alpha=0.1 -> LE_rel=52.05`, `alpha=1.0 -> LE_rel=6.37`;
  `case044 alpha=0.1 -> LE_rel=59.65`, `alpha=1.0 -> LE_rel=6.06`.
- Thus low target RMS amplifies the metric, but does not fully explain the
  failure.

Affine oracle:

- The train-only affine LE oracle is numerically ill-conditioned:
  split_A rank `41/49`, condition about `1.51e20`;
  split_B rank `41/49`, condition about `1.37e20`.
- Its held-out errors are enormous, so the current 8-train-case q design does
  not provide a stable unconstrained affine value-field extrapolation basis.

Offset / anchor result:

- True offset `LE_true - B_true @ q48` is tiny for the validation cases:
  `offset_true_rms` ranges from about `2.87e-06` to `1.14e-05`.
- Predicted offset `LE_pred - B_prior @ q48` is much larger:
  about `7.9e-04` to `9.0e-04`.
- Offset relative errors are large:
  `case041=213.1`, `case049=68.9`, `case044=252.0`, `case046=313.2`.
- This is the clearest signal: B is stable, but the LE value anchor/integration
  constant learned by the network is wrong on held-out cases.

Current interpretation:

- The high `LE_rel` is partly amplified by low validation target RMS, but the
  model is genuinely predicting an overlarge held-out LE field and is worse
  than the zero baseline.
- The issue is not bad B labels, residual AD destroying B, legacy TRUE176 label
  contamination, or split_A bad luck only.
- The most likely current bottleneck is LE value-anchor / residual value-field
  generalization under the frozen strategy and current single-geometry,
  low-LE-heavy fresh validation cases.

Validation boundary:

- No new model was trained for this decomposition.
- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No model, loss, learning-rate, or epoch settings were changed.
- Large generated artifacts remain outside git and must not be committed.

## v1.2 follow-up - 2026-06-23 - B@q anchor oracle diagnostic

Purpose:

- Test whether the stable query B prior provides a better held-out LE value
  anchor than the current learned LE head.
- Check the data-side oracle `B_true @ q48` against `LE_true`.
- Keep this as an evaluation-only diagnostic: no training, no model/loss/lr
  changes, and no epoch changes.

Included:

- Added `scripts/diagnose_v1_2_bq_anchor_oracle.py`.
- The script reconstructs the checkpoint preprocessing, evaluates existing
  checkpoints on selected held-out cases, and writes:
  `bq_anchor_case_summary.csv`, `bq_anchor_frame_summary.csv`,
  `zero_q_anchor_summary.csv`, `residual_offset_summary.csv`, and
  `bq_anchor_diagnostic_summary.json`.

Outputs:

- Split_A diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\bq_anchor_oracle_diagnostic\split_A_val41_49`
- Split_B diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\bq_anchor_oracle_diagnostic\split_B_val44_46`

Key results:

- `B_true @ q48` reconstructs `LE_true` well:
  `Btrue_q_LE_rel=0.0400` for `case041`, `0.0517` for `case049`,
  `0.0380` for `case044`, and `0.0233` for `case046`.
- `B_prior @ q48` is substantially better than the current model LE head, but
  still worse than zero on these low-LE held-out cases:
  `Bprior_q_LE_rel=3.4515` vs `model_LE_rel=9.1422` for `case041`;
  `1.6303` vs `4.0235` for `case049`;
  `1.3224` vs `9.6526` for `case044`;
  `1.5006` vs `7.7823` for `case046`.
- The q=0 model output is large and near the observed error scale:
  `zero_q_pred_rms=7.81e-04` for split_A and `9.02e-04` for split_B.
- The full-model residual/value offset remains large:
  `current_offset_rms=7.87e-04` to `8.99e-04`, compared with
  `true_offset_rms=2.87e-06` to `1.14e-05`.

Current interpretation:

- The fresh strict-pass compact `q/LE/B` contract is strong: the data-side
  `B_true @ q48` oracle nearly reconstructs LE.
- The learned LE value head is not using the stable query B prior as a reliable
  value anchor.  `B_prior @ q48` beats the current LE prediction by about
  `2.47x` to `7.30x` in relative error.
- `B_prior @ q48` alone is not a solved LE predictor for these low-LE held-out
  cases because it remains worse than the zero baseline.
- The next design question is therefore a value-anchor question, not another
  data-contract or training-schedule question: consider an anchored form such as
  `LE_hat = B_prior @ q + residual` with explicit zero-q and low-alpha residual
  constraints.

Validation boundary:

- No new model was trained for this diagnostic.
- No old TRUE176 `LE/B` labels were used as v1.2 labels.
- No model, loss, learning-rate, or epoch settings were changed.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 prototype - 2026-06-23 - Anchored LE head

Purpose:

- Move from v1.2 diagnostics to a structural LE value-anchor prototype.
- Keep the old `query-fe-linear-residual` model style unchanged.
- Add an opt-in anchored model style that enforces raw `q=0 -> LE=0`.

Included:

- Added `QueryFEAnchoredLinearResidualDeepONet`.
- Added `--model-style query-fe-linear-residual-anchored`.
- The anchored model computes:
  `LE_hat_norm = LE_zero_norm + B_prior_norm(point) @ (q_norm - q0_norm)
  + gate(||q_raw||) * (R_raw(q, point) - R_raw(q0, point))`.
- Added evaluation metrics:
  `zero_q_LE_pred_rms`, `zero_q_LE_pred_max_abs`,
  `zero_q_residual_rms`, `Bprior_q_LE_rel`, and
  `model_minus_Bprior_offset_rms`.
- Added smoke coverage for forward shape, raw zero-q anchor,
  residual zero-subtraction, AD-B, old model style compatibility, and a tiny
  1-epoch anchored training smoke.
- Added `docs/query_point_v1_3_anchored_le_head_design.md`.

Current interpretation:

- v1.3 is not a hyperparameter tuning step.  It is motivated by v1.2 evidence
  that the LE value branch has a wrong zero/value anchor while the fresh
  `q/LE/B` data contract and query B prior remain credible.
- This prototype only establishes the optional structure and smoke behavior.
  It does not claim improved formal case-level performance.

Validation boundary:

- No formal 50-epoch v1.3 training audit was run in this commit.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LE head split_A

Purpose:

- Run the first fixed-strategy formal audit of the v1.3 anchored LE head.
- Compare directly against the v1.2 split_A baseline.
- Keep the only intended change as:
  `model_style=query-fe-linear-residual-anchored`.
- Do not change loss, learning rate, epoch count, split, or training data.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Cases:
  `case019`, `case025`, `case031`, `case041`, `case043`,
  `case044`, `case045`, `case046`, `case049`, `case050`.
- Validation split:
  `train_cases=[19,25,31,43,44,45,46,50]`,
  `val_cases=[41,49]`.
- `validation_is_overlapping=false`, `train_frames=80`, `val_frames=20`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_val41_49_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_bq_anchor_oracle\split_A_val41_49_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_A_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_split_A_audit.md`

Key split_A comparison:

- v1.2 split_A best/latest:
  `val_LE_rel=5.137640845816213`,
  `val_AD_B_rel=0.5317644562295766`,
  `b_prior_current_val_evalcols_B_rel=0.5306775207214273`,
  prior zero-q diagnostic `zero_q_pred_rms ~= 7.81e-04`.
- v1.3 anchored best checkpoint, epoch 10:
  `val_LE_rel=2.1796218548207182`,
  `val_AD_B_rel=0.5322959397919018`,
  `val_AD_B_cos=0.8718914233087541`,
  `b_prior_current_val_evalcols_B_rel=0.532185230457769`,
  `val_zero_q_LE_pred_rms=1.1484941768098402e-11`,
  `val_model_minus_Bprior_offset_rms=0.00010172704191546938`.
- v1.3 anchored latest checkpoint, epoch 50:
  `val_LE_rel=3.0012689077205144`,
  `val_AD_B_rel=0.5329800420018063`,
  `val_zero_q_LE_pred_rms=1.1484941768098402e-11`,
  `val_model_minus_Bprior_offset_rms=0.0003313721329316278`.

Per-case best attribution:

- `case041`: `LE_rel=3.609960155721904`,
  `AD_B_rel=0.5321915350416156`,
  `B_prior_rel=0.5321483814179929`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=3.8535600056709796`,
  `model_minus_Bprior_offset_rms=0.00010339966315563473`.
- `case049`: `LE_rel=1.8090445751604918`,
  `AD_B_rel=0.5324002526523295`,
  `B_prior_rel=0.532222051735648`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=1.8501516802362072`,
  `model_minus_Bprior_offset_rms=0.00010005442067530404`.

Current interpretation:

- The zero-q ghost LE field is structurally removed in split_A.
- The best split_A `val_LE_rel` improves materially relative to v1.2, while
  AD-B remains stable.
- The latest checkpoint is worse than the best checkpoint, so the result is
  still epoch-sensitive and should be judged by best checkpoint for this audit.
- This is enough evidence to run split_B next with the same fixed strategy.
- It is not yet evidence that v1.3 broadly solves LE generalization; split_B
  and LOO remain required.

Validation boundary:

- No model, loss, learning-rate, epoch, or split changes were made beyond the
  already committed opt-in anchored model style.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LE head split_B

Purpose:

- Run the second fixed-strategy formal audit of the v1.3 anchored LE head.
- Compare directly against the v1.2 split_B baseline.
- Keep the only intended model change relative to v1.2 as:
  `model_style=query-fe-linear-residual-anchored`.
- Change only `val_cases` relative to v1.3 split_A; do not change loss,
  learning rate, epoch count, split policy, or data pool.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Cases:
  `case019`, `case025`, `case031`, `case041`, `case043`,
  `case044`, `case045`, `case046`, `case049`, `case050`.
- Validation split:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`.
- `validation_is_overlapping=false`, `train_frames=80`, `val_frames=20`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_val44_46_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_bq_anchor_oracle\split_B_val44_46_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_split_B_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_split_B_audit.md`

Key split_B comparison:

- v1.2 split_B best/latest:
  `val_LE_rel=8.515458607946384`,
  `val_AD_B_rel=0.5103261083920874`,
  `val_AD_B_cos=0.8833993891072419`,
  `b_prior_current_val_evalcols_B_rel=0.5093552486612618`.
- v1.3 anchored best checkpoint, epoch 1:
  `val_LE_rel=1.5595633647976805`,
  `val_AD_B_rel=0.5125447167349023`,
  `val_AD_B_cos=0.8790831126173732`,
  `b_prior_current_val_evalcols_B_rel=0.5125437997617942`,
  `val_zero_q_LE_pred_rms=1.157102562071606e-11`,
  `val_model_minus_Bprior_offset_rms=1.7732607370311359e-06`.
- v1.3 anchored latest checkpoint, epoch 50:
  `val_LE_rel=1.7711132238745921`,
  `val_AD_B_rel=0.5114605897630667`,
  `val_zero_q_LE_pred_rms=1.157102562071606e-11`,
  `val_model_minus_Bprior_offset_rms=9.181341274153851e-05`.

Per-case best attribution:

- `case044`: `LE_rel=1.2623462866926172`,
  `AD_B_rel=0.512531727633403`,
  `B_prior_rel=0.5125308365182847`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.2603167162795477`,
  `model_minus_Bprior_offset_rms=1.612453271090318e-06`.
- `case046`: `LE_rel=1.7080236136717677`,
  `AD_B_rel=0.5125577066694196`,
  `B_prior_rel=0.5125567638373174`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.706468997850503`,
  `model_minus_Bprior_offset_rms=1.9340682029719545e-06`.

Current interpretation:

- The zero-q ghost LE field remains structurally removed in split_B.
- The best split_B `val_LE_rel` improves materially relative to v1.2, while
  AD-B remains stable.
- The latest checkpoint remains much better than v1.2 split_B, but is slightly
  worse than best, so epoch sensitivity remains visible.
- Together with split_A, v1.3 anchored head now shows LE improvement on two
  independent validation splits without breaking B.
- Broad performance is still not fully proven; LOO or a formal v1.3 summary is
  the next audit step before any new tuning.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made beyond the already committed opt-in anchored model style and the requested
  split_B validation cases.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 summary - 2026-06-23 - Two-split formal evidence

Purpose:

- Freeze the v1.3 evidence from split_A and split_B before running additional
  LOO audits.
- Record the current interpretation conservatively: two non-overlapping case
  splits improved, but full case-level generalization is not yet proven.
- Define a risk-prioritized LOO plan instead of immediately running full LOO or
  changing hyperparameters.

Included:

- Added `docs/query_point_v1_3_two_split_formal_summary.md`.
- The summary records the fixed strategy, split_A/split_B metrics, per-case
  attribution, and LOO priority plan.

Two-split result:

- split_A:
  `train_cases=[19,25,31,43,44,45,46,50]`,
  `val_cases=[41,49]`,
  `validation_is_overlapping=false`.
  v1.2 `val_LE_rel=5.137640845816213`;
  v1.3 best `val_LE_rel=2.1796218548207182`,
  `val_AD_B_rel=0.5322959397919018`,
  `val_zero_q_LE_pred_rms=1.1484941768098402e-11`.
- split_B:
  `train_cases=[19,25,31,41,43,45,49,50]`,
  `val_cases=[44,46]`,
  `validation_is_overlapping=false`.
  v1.2 `val_LE_rel=8.515458607946384`;
  v1.3 best `val_LE_rel=1.5595633647976805`,
  `val_AD_B_rel=0.5125447167349023`,
  `val_zero_q_LE_pred_rms=1.157102562071606e-11`.

Per-case v1.3 attribution:

- split_A `case041`: `LE_rel=3.609960155721904`,
  `AD_B_rel=0.5321915350416156`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=3.8535600056709796`.
- split_A `case049`: `LE_rel=1.8090445751604918`,
  `AD_B_rel=0.5324002526523295`,
  `zero_q_LE_pred_rms=1.1484941768098398e-11`,
  `Bprior_q_LE_rel=1.8501516802362072`.
- split_B `case044`: `LE_rel=1.2623462866926172`,
  `AD_B_rel=0.512531727633403`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.2603167162795477`.
- split_B `case046`: `LE_rel=1.7080236136717677`,
  `AD_B_rel=0.5125577066694196`,
  `zero_q_LE_pred_rms=1.157102562071606e-11`,
  `Bprior_q_LE_rel=1.706468997850503`.

Current interpretation:

- v1.3 anchored head materially improves held-out `LE` on split_A and split_B.
- The zero-q ghost LE field is structurally removed.
- AD-B is not broken: validation AD-B metrics remain close to v1.2.
- Best epochs are early (`split_A` epoch 10, `split_B` epoch 1), so epoch
  sensitivity remains part of the evidence.
- Full case-level generalization is not yet proven because `case019`,
  `case025`, `case031`, `case043`, `case045`, and `case050` have not yet been
  held out under v1.3.

LOO priority plan:

- Priority 1: `LOO_case031`, because `case031` is the high-amplitude / high-LE
  case and tests whether the anchored head can extrapolate without that training
  anchor.
- Priority 2: `LOO_case050`, because it is a mixed / larger fresh direction and
  tests non-low-amplitude mixed-direction generalization.
- Priority 3: `LOO_case043` or `LOO_case045`, to add medium fresh-direction
  evidence.
- Priority 4: full 10-case LOO for a final formal generalization claim.

Validation boundary:

- This step did not run additional training.
- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LOO case031

Purpose:

- Run the first risk-prioritized LOO audit after the two-split summary.
- Hold out `case031`, the high-amplitude / high-LE case, to test whether v1.3
  can extrapolate without that case as a training anchor.
- Keep the fixed v1.3 strategy unchanged.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Validation split:
  `train_cases=[19,25,41,43,44,45,46,49,50]`,
  `val_cases=[31]`.
- `validation_is_overlapping=false`, `train_frames=90`, `val_frames=10`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_bq_anchor_oracle\LOO_case031_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case031_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_LOO_case031_audit.md`

Key result:

- Best and latest are both epoch 50:
  `val_LE_rel=0.9255415441048325`,
  `train_LE_rel=0.4854010592101344`,
  `val_AD_B_rel=0.7236396387001852`,
  `val_AD_B_cos=0.7363247525021522`,
  `b_prior_current_val_evalcols_B_rel=0.7239878680693814`,
  `val_zero_q_LE_pred_rms=2.3911419659618957e-13`,
  `val_Bprior_q_LE_rel=0.9346750750874383`,
  `val_model_minus_Bprior_offset_rms=0.0009073160436565322`.
- Per-case attribution for `case031`:
  `LE_rel=0.9255415441048325`,
  `AD_B_rel=0.7236396387001852`,
  `B_prior_rel=0.7239878680693814`,
  `zero_q_LE_pred_rms=2.3911419659618957e-13`,
  `Bprior_q_LE_rel=0.9346750750874383`,
  `model_minus_Bprior_offset_rms=0.0009073160436565322`.
- B@q diagnostic:
  `Btrue_q_LE_rel=0.6522641115009261`,
  `Bprior_q_LE_rel=0.967480304420319`,
  `model_LE_rel=0.9255415441048325`,
  `zero_q_pred_rms=3.720241450096156e-13`,
  `current_offset_rms=0.0009074376473615923`,
  `true_offset_rms=0.008277004439016715`.

Current interpretation:

- `LOO_case031` does not collapse on held-out high-amplitude LE:
  `val_LE_rel` is below 1.0.
- The zero-q ghost remains structurally removed.
- AD-B and the B prior are substantially harder than in split_A/split_B:
  `val_AD_B_rel=0.7236` versus about `0.53` and `0.51` in split_A/split_B.
- `Btrue @ q` is not a near-perfect value oracle on `case031`
  (`Btrue_q_LE_rel=0.6523`), which is consistent with stronger high-amplitude
  nonlinearity or state/path effects.
- This supports the anchored head as a useful structural fix while exposing a
  remaining high-amplitude B coverage / state-dependence risk.
- Recommended next audit: run `LOO_case050` with the same fixed strategy before
  changing model, loss, learning rate, or epoch count.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v2 planning - 2026-06-23 - Coordinate-consistent route contract

Purpose:

- Freeze the conceptual transition from v1.3 raw-q diagnostics to a v2
  coordinate-consistent finite-element operator route.
- Record that the final route should learn a standard/local macro-element
  operator, not only a raw Abaqus-coordinate black-box map.
- Add a read-only contract audit that identifies which current/future compacts
  contain the fields needed for v2 chain-rule-consistent B supervision.

Key route change:

```text
q48_raw
  -> q_useful = T_q_raw_to_useful @ q48_raw
  -> NN(q_useful, xi_standard, geometry_features)
  -> strain_standard_or_local
  -> LE_Abaqus
```

The v2 B comparison must use:

```text
B_raw_hat = T_eps_to_abq @ d(strain_standard)/d(q_useful) @ T_q_raw_to_useful
```

and may not directly compare `d(strain_standard)/d(q_useful)` against the
current raw Abaqus/Sobolev label:

```text
B_LE128_forward = d(LE_Abaqus)/d(q48_raw)
```

Added:

- `docs/query_point_v2_coordinate_consistent_route.md`
- `scripts/audit_v2_coordinate_consistent_contract.py`
- v2 gate wording in `docs/query_point_abaqus_workflow.md`

Probe result on the current 10-case strict fresh pool:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact-list D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\v1_2_min10_probe
```

- `compact_count=10`
- `strict_v2_coordinate_pass=false`
- `strict_v2_coordinate_pass_count=0`
- The existing v1.x compacts already have valid standard/IP geometry fields:
  `ip_xi_ok=true`, `ip_J_ok=true`, `ip_invJ_ok=true`, `ip_detJ_ok=true`.
- All 10 fail the new v2 coordinate-contract fields:
  `missing_q_useful`, `missing_T_q_raw_to_useful`,
  `missing_strain_output_coordinate`, `missing_q_useful_coordinate`,
  `missing_B_label_q_coordinate`, `missing_B_label_output_coordinate`,
  `missing_T_eps_to_abq_or_B_standard_useful`, and
  `missing_B_chain_rule_metadata`.
- `--strict-v2` was checked on one current compact and failed nonzero as
  expected.

Current interpretation:

- v1.3 remains useful diagnostic evidence: strict compacts, case splits,
  anchored zero-q behavior, and AD-B stability are not discarded.
- v1.3 is not the final clean operator contract because the branch coordinate,
  strain coordinate, and B-label coordinate are still raw-coordinate coupled.
- The next scientific step is a one-case v2 coordinate-chain audit, not another
  long v1.3 training run.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or tag changes were made.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- The new audit script is read-only and does not train or transform labels.
- Existing v1.x compacts are expected to fail strict v2 until `q_useful`,
  `T_q_raw_to_useful`, strain-coordinate metadata, and chain-rule metadata are
  generated.

## v2a pilot - 2026-06-23 - One-case q-coordinate consistency

Purpose:

- Build a one-case q-coordinate pilot for the v2 route without training.
- Isolate the branch-coordinate transform before attempting standard/local
  strain labels.
- Use `case050` because prior v1.3 diagnostics showed strong data-side
  `Btrue @ q` while still exposing value-field generalization limits.

Input:

- Source compact:
  `D:\IS-FEM\outputs\query_point_v1_2_fresh_cases\case050\complete\complete_case050_training_ready.npz`
- q48 node coordinate source:
  `X_keep_ref [16,3]`

Added:

- `scripts/build_v2_q_useful_pilot_compact.py`
- `docs/query_point_v2_one_case_q_useful_pilot.md`
- The v2 audit now recognizes `B_useful_abq` as the v2a useful-B field name.

Generated output, not committed:

- `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz`
- `D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.summary.json`

Method:

```text
R = six rigid modes from X_keep_ref:
    translations x/y/z
    small rotations x/y/z with u_i = omega cross (X_i - centroid)
N = orthogonal complement of R, shape [48,42]
T_q_raw_to_useful = N.T
q_useful = q48_raw @ T_q_raw_to_useful.T
P_useful = N @ N.T
```

This pilot keeps output strain in Abaqus global coordinates:

```text
strain_output_coordinate = abaqus_global
T_eps_to_abq = I_6
B_useful_abq = B_LE128_forward @ N
B_raw_hat = B_useful_abq @ T_q_raw_to_useful
```

Key result:

- `q48_shape=[10,48]`
- `q_useful_shape=[10,42]`
- `T_q_raw_to_useful_shape=[42,48]`
- `B_useful_abq_shape=[10,128,6,42]`
- `rigid_mode_rank=6`
- `rigid_annihilation_max=1.371671448566352e-16`
- `useful_basis_orthonormal_max=4.440892098500626e-16`
- `q_useful_reconstruction_rel=0.33576244788862675`
- `q_useful_removed_rigid_rel=0.33576244788862675`
- `B_chain_rule_projected_rel=4.100651493362974e-16`
- `B_chain_rule_projected_max_abs=1.4210854715202004e-13`
- `B_chain_rule_raw_rel=0.0001470978954571399`
- `B_chain_rule_raw_max_abs=0.019154849670568908`
- `B_rigid_residual_rel=0.00014709789545713607`

Strict v2 audit:

```powershell
py -3 scripts\audit_v2_coordinate_consistent_contract.py `
  --compact D:\IS-FEM\outputs\query_point_v2_coordinate_pilot\case050\complete_case050_v2a_q_useful_pilot.npz `
  --out-root D:\IS-FEM\outputs\query_point_v2_coordinate_contract_audit\case050_v2a_q_useful_pilot `
  --strict-v2
```

Result:

- `strict_v2_coordinate_pass=true`
- `strict_v2_coordinate_pass_count=1`
- `failure_counts={}`

Current interpretation:

- The v2a q-coordinate chain closes for `case050`.
- About 33.6 percent of raw q is removed as rigid-body content, which confirms
  that raw-q contains non-useful motion.
- The raw Abaqus B response in removed rigid directions is tiny
  (`B_rigid_residual_rel=1.47e-4`), so the useful-q projection does not remove
  meaningful strain sensitivity for this case.
- The projected B chain rule is exact to numerical precision.
- The next step should be a separate v2b strain-coordinate/local-coordinate
  pilot, not immediate v2 training.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or tag changes were made.
- No old TRUE176 `LE/B` labels were used as v2 labels.
- No standard/local strain labels were generated; this is Abaqus-global-strain
  q-coordinate pilot only.
- Generated `.npz` output remains under `outputs` and was not committed.

## v1.3 audit - 2026-06-23 - Anchored LOO case043

Purpose:

- Run the third risk-prioritized LOO audit after `LOO_case031` and
  `LOO_case050`.
- Hold out `case043`, a medium fresh direction, to test whether the anchored
  strategy remains stable between the high-amplitude and mixed/larger checks.
- Keep the fixed v1.3 strategy unchanged.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Validation split:
  `train_cases=[19,25,31,41,44,45,46,49,50]`,
  `val_cases=[43]`.
- `validation_is_overlapping=false`, `train_frames=90`, `val_frames=10`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_bq_anchor_oracle\LOO_case043_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_LOO_case043_audit.md`

Key result:

- Best checkpoint, epoch 10:
  `val_LE_rel=1.8380485229166579`,
  `train_LE_rel=0.74541650372748`,
  `val_AD_B_rel=0.5036482686290428`,
  `val_AD_B_cos=0.8830602940469523`,
  `b_prior_current_val_evalcols_B_rel=0.503501472353476`,
  `val_zero_q_LE_pred_rms=3.863962753320447e-12`,
  `val_Bprior_q_LE_rel=1.9000499821060757`,
  `val_model_minus_Bprior_offset_rms=0.00016322065296508765`.
- Latest checkpoint, epoch 50:
  `val_LE_rel=1.9656412310345146`,
  `val_AD_B_rel=0.5016945264798282`,
  `b_prior_current_val_evalcols_B_rel=0.5007423836646588`,
  `val_zero_q_LE_pred_rms=3.863962753320447e-12`.
- Per-case attribution for `case043`:
  `LE_rel=1.8380485229166579`,
  `AD_B_rel=0.5036482686290428`,
  `B_prior_rel=0.503501472353476`,
  `zero_q_LE_pred_rms=3.863962753320447e-12`,
  `Bprior_q_LE_rel=1.9000499821060757`,
  `model_minus_Bprior_offset_rms=0.00016322065296508765`.
- B@q diagnostic:
  `Btrue_q_LE_rel=0.10034757778215131`,
  `Bprior_q_LE_rel=1.8931336235552196`,
  `model_LE_rel=1.8380485229166579`,
  `zero_q_pred_rms=0.0`,
  `current_offset_rms=0.00021721550419227364`,
  `true_offset_rms=2.075807588384023e-05`.

Current interpretation:

- `LOO_case043` keeps the zero-q ghost structurally removed.
- AD-B and the query B prior remain stable:
  `val_AD_B_rel=0.5036`, `b_prior_current_val_evalcols_B_rel=0.5035`.
- `val_LE_rel=1.8380` remains high, so medium fresh-direction LE value
  generalization is not solved.
- `Btrue @ q` is strong on `case043` (`Btrue_q_LE_rel=0.1003`), so the data-side
  q/LE/B contract is not the bottleneck; the remaining issue is value-field
  generalization.
- Best is epoch 10 and latest is worse, so epoch sensitivity remains visible.
- Recommended next audit: run `LOO_case045` with the same fixed strategy, or
  write a partial LOO summary before deciding whether to expand data or revisit
  the value branch.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.

## v1.3 audit - 2026-06-23 - Anchored LOO case050

Purpose:

- Run the second risk-prioritized LOO audit after `LOO_case031`.
- Hold out `case050`, a mixed / larger fresh direction, to test mixed-direction
  held-out generalization.
- Keep the fixed v1.3 strategy unchanged.

Inputs:

- Compact pool:
  `D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`
- Validation split:
  `train_cases=[19,25,31,41,43,44,45,46,49]`,
  `val_cases=[50]`.
- `validation_is_overlapping=false`, `train_frames=90`, `val_frames=10`.

Outputs:

- Training:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050`
- Case attribution:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_case_attribution`
- B@q diagnostic:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_bq_anchor_oracle\LOO_case050_best`
- Comparison:
  `D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_comparison`
- Documentation:
  `docs/query_point_v1_3_anchored_LOO_case050_audit.md`

Key result:

- Best checkpoint, epoch 1:
  `val_LE_rel=1.5294014338724993`,
  `train_LE_rel=1.6161069592630004`,
  `val_AD_B_rel=0.517000373997325`,
  `val_AD_B_cos=0.8751722384624953`,
  `b_prior_current_val_evalcols_B_rel=0.5169978402891399`,
  `val_zero_q_LE_pred_rms=1.1676963344145722e-11`,
  `val_Bprior_q_LE_rel=1.5326505786455733`,
  `val_model_minus_Bprior_offset_rms=3.92915279653482e-06`.
- Latest checkpoint, epoch 50:
  `val_LE_rel=1.7229821122212494`,
  `val_AD_B_rel=0.5145903038519773`,
  `b_prior_current_val_evalcols_B_rel=0.5137761146056249`,
  `val_zero_q_LE_pred_rms=1.1676963344145722e-11`.
- Per-case attribution for `case050`:
  `LE_rel=1.5294014338724993`,
  `AD_B_rel=0.517000373997325`,
  `B_prior_rel=0.5169978402891399`,
  `zero_q_LE_pred_rms=1.1676963344145722e-11`,
  `Bprior_q_LE_rel=1.5326505786455733`,
  `model_minus_Bprior_offset_rms=3.92915279653482e-06`.
- B@q diagnostic:
  `Btrue_q_LE_rel=0.06644355125974967`,
  `Bprior_q_LE_rel=1.5310648298695695`,
  `model_LE_rel=1.5294014338724993`,
  `zero_q_pred_rms=0.0`,
  `current_offset_rms=4.327757044449705e-06`,
  `true_offset_rms=1.843403091453214e-05`.

Current interpretation:

- `LOO_case050` does not collapse on the held-out mixed / larger fresh direction.
- The zero-q ghost remains structurally removed.
- AD-B and the query B prior remain stable and close to split_B:
  `val_AD_B_rel=0.5170`, `b_prior_current_val_evalcols_B_rel=0.5170`.
- `val_LE_rel=1.5294` is comparable to split_B and better than split_A, but it
  is still above 1.0, so mixed-direction value generalization is not solved.
- `Btrue @ q` is strong on `case050` (`Btrue_q_LE_rel=0.0664`), so the data-side
  q/LE/B contract is not the bottleneck; the remaining issue is value-field
  generalization.
- Best is early (epoch 1) and latest is worse, so epoch sensitivity remains
  visible.
- Recommended next audit: run `LOO_case043` or `LOO_case045` before full LOO.

Validation boundary:

- No model, loss, learning-rate, epoch, split-policy, or data-pool changes were
  made.
- No old TRUE176 `LE/B` labels were used as v1.2/v1.3 labels.
- No tag was moved.
- Large generated artifacts remain outside git and must not be committed.
