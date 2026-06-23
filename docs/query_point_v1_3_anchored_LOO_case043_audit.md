# v1.3 Anchored LE Head LOO Case043 Audit

Date: 2026-06-23

## Purpose

Run the third risk-prioritized leave-one-case-out audit for v1.3 anchored LE.

`case043` was selected as a medium fresh-direction check after:

- `LOO_case031`, which tested high-amplitude / high-LE extrapolation.
- `LOO_case050`, which tested a mixed / larger fresh direction.

This audit only runs `LOO_case043`.  It does not run `case045` or full LOO.

## Boundary

No model, loss, learning-rate, epoch, data-pool, or tag changes were made.

No old TRUE176 `LE/B` labels were used.  The training data remains the strict
fresh compact pool listed in:

`D:\IS-FEM\outputs\query_point_v1_2_training_audit\pool_v1_2_min10_manifest\compact_list.txt`

Large generated artifacts remain outside git.

## Split

- `validation_split_mode=case`
- `validation_is_overlapping=false`
- `train_cases=[19,25,31,41,44,45,46,49,50]`
- `val_cases=[43]`
- `train_frames=90`
- `val_frames=10`

Output directory:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043`

## Fixed Strategy

- `model_style=query-fe-linear-residual-anchored`
- `point_feature_source=data`
- `branch_feature_mode=xkeep-qraw`
- `le_normalization=global-component`
- `train_point_sample_count=128`
- `split_mode=case`
- `allow_overlap_val=false`
- `val_cases=43`
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

- `best_epoch=10`
- `best_score=2.3416967915457008`
- `train_LE_rel=0.74541650372748`
- `val_LE_rel=1.8380485229166579`
- `train_AD_B_rel=0.5160809789254196`
- `val_AD_B_rel=0.5036482686290428`
- `train_AD_B_cos=0.8596124251563153`
- `val_AD_B_cos=0.8830602940469523`
- `train_phys_rand_dir_B_rel=0.5139457097929578`
- `val_phys_rand_dir_B_rel=0.5052459713833004`
- `b_prior_current_train_evalcols_B_rel=0.5158730652973683`
- `b_prior_current_val_evalcols_B_rel=0.503501472353476`
- `train_AD_B_ip_rel_max=4.105253280167911`
- `val_AD_B_ip_rel_max=3.5818628660027882`
- `val_AD_B_col_rel_max=0.7726722643272572`
- `train_zero_q_LE_pred_rms=3.863962753320447e-12`
- `val_zero_q_LE_pred_rms=3.863962753320447e-12`
- `train_zero_q_LE_pred_max_abs=5.3314557629802195e-12`
- `val_zero_q_LE_pred_max_abs=5.3314557629802195e-12`
- `train_zero_q_residual_rms=0.0`
- `val_zero_q_residual_rms=0.0`
- `train_Bprior_q_LE_rel=1.6849670367376421`
- `val_Bprior_q_LE_rel=1.9000499821060757`
- `train_model_minus_Bprior_offset_rms=0.0010253476075768628`
- `val_model_minus_Bprior_offset_rms=0.00016322065296508765`

Latest checkpoint:

- `latest_epoch=50`
- `train_LE_rel=0.3117178897208297`
- `val_LE_rel=1.9656412310345146`
- `train_AD_B_rel=0.5089171355331024`
- `val_AD_B_rel=0.5016945264798282`
- `train_AD_B_cos=0.8663802937955818`
- `val_AD_B_cos=0.8861545507984363`
- `b_prior_current_val_evalcols_B_rel=0.5007423836646588`
- `val_zero_q_LE_pred_rms=3.863962753320447e-12`
- `val_Bprior_q_LE_rel=1.9688165620634035`
- `val_model_minus_Bprior_offset_rms=0.00010718746720676762`

## Comparison With Previous Audits

| Run | Held-out cases | `val_LE_rel` | `val_AD_B_rel` | `zero_q_LE_pred_rms` |
|---|---|---:|---:|---:|
| split_A best | `[41,49]` | 2.1796218548207182 | 0.5322959397919018 | 1.1484941768098402e-11 |
| split_B best | `[44,46]` | 1.5595633647976805 | 0.5125447167349023 | 1.157102562071606e-11 |
| LOO_case031 best | `[31]` | 0.9255415441048325 | 0.7236396387001852 | 2.3911419659618957e-13 |
| LOO_case050 best | `[50]` | 1.5294014338724993 | 0.517000373997325 | 1.1676963344145722e-11 |
| LOO_case043 best | `[43]` | 1.8380485229166579 | 0.5036482686290428 | 3.863962753320447e-12 |

`LOO_case043` keeps AD-B stable and slightly better than split_A/split_B, but
held-out LE remains high.  This is not a zero-q failure: the anchored zero-q
prediction stays near numerical zero.

## Per-Case Attribution

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_case_attribution`

| Case | `LE_rel` | `AD_B_rel` | `B_prior_rel` | `zero_q_LE_pred_rms` | `Bprior_q_LE_rel` | `model_minus_Bprior_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 43 | 1.8380485229166579 | 0.5036482686290428 | 0.503501472353476 | 3.863962753320447e-12 | 1.9000499821060757 | 0.00016322065296508765 |

## B@q Diagnostic

Output:

`D:\IS-FEM\outputs\query_point_v1_3_training_audit\anchored_LOO_case043_bq_anchor_oracle\LOO_case043_best`

| Case | `Btrue_q_LE_rel` | `Bprior_q_LE_rel` | `model_LE_rel` | `zero_q_pred_rms` | `current_offset_rms` | `true_offset_rms` |
|---:|---:|---:|---:|---:|---:|---:|
| 43 | 0.10034757778215131 | 1.8931336235552196 | 1.8380485229166579 | 0.0 | 0.00021721550419227364 | 2.075807588384023e-05 |

The data-side `Btrue @ q` oracle is strong on `case043`
(`Btrue_q_LE_rel=0.10034757778215131`).  This means the q/LE/B contract is not
the bottleneck for this case.  The remaining error is primarily a medium
fresh-direction value-generalization issue.

As before, B@q diagnostic values are evaluation-only.  If raw `q_coord` oracle
values differ from trainer/evaluate metrics, the trainer/evaluate metrics are
the primary audit source.

## Interpretation

What is established:

- `LOO_case043` completed with a clean non-overlapping case split.
- The zero-q LE ghost remains structurally fixed.
- AD-B and query B prior remain stable:
  `val_AD_B_rel=0.5036482686290428` and
  `b_prior_current_val_evalcols_B_rel=0.503501472353476`.
- The data-side `Btrue @ q` oracle is strong:
  `Btrue_q_LE_rel=0.10034757778215131`.

What remains limited:

- `val_LE_rel=1.8380485229166579` is still high, worse than split_B and
  `LOO_case050`, though better than split_A.
- Best is epoch 10 and latest is worse (`val_LE_rel=1.9656412310345146`), so
  epoch sensitivity remains visible.
- Because `Btrue @ q` is strong, this case points more toward value/generalizer
  limitations than toward bad labels or missing B supervision.

Current conclusion:

`LOO_case043` adds medium fresh-direction evidence that v1.3 anchored head keeps
zero-q and AD-B stable, but it does not prove LE generalization is solved.  The
case exposes value-field generalization limits under a fresh held-out direction.

Recommended next step:

- Run `LOO_case045` next with the same fixed strategy, or write a partial LOO
  summary covering split_A, split_B, `LOO_case031`, `LOO_case050`, and
  `LOO_case043`.
- Do not change model, loss, learning rate, or epoch count before that decision
  is recorded.
