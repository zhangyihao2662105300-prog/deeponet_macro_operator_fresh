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
