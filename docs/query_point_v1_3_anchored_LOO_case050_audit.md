# v1.3 Anchored LE Head LOO Case050 Audit

Date: 2026-06-23

## Purpose

Run the second risk-prioritized leave-one-case-out audit for v1.3 anchored LE.

`case050` was selected because it is a mixed / larger fresh direction.  It
complements `LOO_case031`:

- `LOO_case031` tests high-amplitude / high-LE extrapolation.
- `LOO_case050` tests held-out mixed-direction value generalization.

This audit only runs `LOO_case050`.  It does not run `case043`, `case045`, or
full LOO.

## Boundary

No model, loss, learning-rate, epoch, data-pool, or tag changes were made.

No old TRUE176 `LE/B` labels were used.  The training data remains the strict
fresh compact pool listed in:

`D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`

Large generated artifacts remain outside git.

## Split

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[19,25,31,41,43,44,45,46,49]`
- `val_cases=[50]`
- `train_frames=90`
- `val_frames=10`

Output directory:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050`

## Fixed Strategy

- `model_style=query-fe-linear-residual-anchored`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=128`
- `split_mode=case`
- `allow_overlap_val=false`
- `val_cases=50`
- `epochs=50`
- `b_baseline_warmstart_source=train-only`
- `b_baseline_warmstart_steps=500`
- `j_loss_mode=physical`
- `global_b_lr_scale=0.05`
- `point_b_lr_scale=0.10`
- `le_loss_weight=1.0`
- `baseline_j_loss_mode=norm-plus-physical`
- `baseline_jacobian_weight=1.0`
- `initial_jacobian_weight=1.0`

## Best / Latest Metrics

Best checkpoint:

- `best_epoch=1`
- `best_score=2.0464018078698243`
- `train_LE_rel=1.6161069592630004`
- `val_LE_rel=1.5294014338724993`
- `train_AD_B_rel=0.5303422908098847`
- `val_AD_B_rel=0.517000373997325`
- `train_AD_B_cos=0.8505623537821514`
- `val_AD_B_cos=0.8751722384624953`
- `train_phys_rand_dir_B_rel=0.5312991834758255`
- `val_phys_rand_dir_B_rel=0.5183517146771989`
- `b_prior_current_train_evalcols_B_rel=0.5303368426324562`
- `b_prior_current_val_evalcols_B_rel=0.5169978402891399`
- `train_AD_B_ip_rel_max=4.291026400388988`
- `val_AD_B_ip_rel_max=3.7383072999302294`
- `val_AD_B_col_rel_max=0.770552743059488`
- `train_zero_q_LE_pred_rms=1.1676963344145718e-11`
- `val_zero_q_LE_pred_rms=1.1676963344145722e-11`
- `train_zero_q_LE_pred_max_abs=2.6126663332792788e-11`
- `val_zero_q_LE_pred_max_abs=2.6126663332792788e-11`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=1.927661421112666`
- `val_Bprior_q_LE_rel=1.5326505786455733`
- `train_model_minus_Bprior_offset_rms=2.738573373448194e-05`
- `val_model_minus_Bprior_offset_rms=3.92915279653482e-06`

Latest checkpoint:

- `latest_epoch=50`
- `train_LE_rel=0.29580473706175486`
- `val_LE_rel=1.7229821122212494`
- `train_AD_B_rel=0.5222220480095185`
- `val_AD_B_rel=0.5145903038519773`
- `train_AD_B_cos=0.8582407000463297`
- `val_AD_B_cos=0.8788987965212188`
- `b_prior_current_val_evalcols_B_rel=0.5137761146056249`
- `val_zero_q_LE_pred_rms=1.1676963344145722e-11`
- `val_Bprior_q_LE_rel=1.6379922447049655`
- `val_model_minus_Bprior_offset_rms=0.0001240665532517473`

## Comparison With Previous Audits

| Run | Held-out cases | `val_LE_rel` | `val_AD_B_rel` | `zero_q_LE_pred_rms` |
|---|---|---:|---:|---:|
| split_A best | `[41,49]` | 2.1796218548207182 | 0.5322959397919018 | 1.1484941768098402e-11 |
| split_B best | `[44,46]` | 1.5595633647976805 | 0.5125447167349023 | 1.157102562071606e-11 |
| LOO_case031 best | `[31]` | 0.9255415441048325 | 0.7236396387001852 | 2.3911419659618957e-13 |
| LOO_case050 best | `[50]` | 1.5294014338724993 | 0.517000373997325 | 1.1676963344145722e-11 |

`LOO_case050` does not collapse: `val_LE_rel=1.5294014338724993`, comparable
to split_B and better than split_A.  Unlike `LOO_case031`, AD-B remains stable
and close to split_B.

## Per-Case Attribution

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_case_attribution`

| Case | `LE_rel` | `AD_B_rel` | `B_prior_rel` | `zero_q_LE_pred_rms` | `Bprior_q_LE_rel` | `model_minus_Bprior_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.5294014338724993 | 0.517000373997325 | 0.5169978402891399 | 1.1676963344145722e-11 | 1.5326505786455733 | 3.92915279653482e-06 |

## B@q Diagnostic

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case050_bq_anchor_oracle\LOO_case050_best`

| Case | `Btrue_q_LE_rel` | `Bprior_q_LE_rel` | `model_LE_rel` | `zero_q_pred_rms` | `current_offset_rms` | `true_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.06644355125974967 | 1.5310648298695695 | 1.5294014338724993 | 0.0 | 4.327757044449705e-06 | 1.843403091453214e-05 |

The data-side `Btrue @ q` oracle is strong on `case050`
(`Btrue_q_LE_rel=0.06644355125974967`).  This means the q/LE/B contract is not
the bottleneck for this case.  The remaining error is primarily a mixed-direction
value-generalization issue rather than a data-contract or B-label failure.

As before, B@q diagnostic values are evaluation-only.  If raw `q_coord` oracle
values differ from trainer/evaluate metrics, the trainer/evaluate metrics are
the primary audit source.

## Interpretation

What is established:

- `LOO_case050` completed with a clean non-overlapping case split.
- The zero-q LE ghost remains structurally fixed.
- AD-B and query B prior remain stable:
  `val_AD_B_rel=0.517000373997325` and
  `b_prior_current_val_evalcols_B_rel=0.5169978402891399`.
- The held-out mixed/larger case does not collapse:
  `val_LE_rel=1.5294014338724993`.

What remains limited:

- `val_LE_rel` remains above 1.0, so mixed-direction value generalization is
  not solved.
- Best is epoch 1 and latest is worse (`val_LE_rel=1.7229821122212494`), so
  epoch sensitivity remains visible.
- Because `Btrue @ q` is strong, this case points more toward value/generalizer
  limitations than toward bad labels or missing B supervision.

Current conclusion:

`LOO_case050` adds positive evidence that v1.3 anchored head handles a held-out
mixed / larger fresh direction without collapse, while also showing that mixed
direction LE value generalization is still imperfect.

Recommended next step:

- Run `LOO_case043` or `LOO_case045` before full LOO.
- Do not change model, loss, learning rate, or epoch count before that next
  case-level evidence is recorded.
